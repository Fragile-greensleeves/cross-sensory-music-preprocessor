"""Automatic melody, chord and beat analysis with explicit degraded backends."""
from __future__ import annotations

import hashlib
import importlib.metadata
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load(path, sr=16000):
    import librosa
    y, original_sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if original_sr != sr:
        y = librosa.resample(y, orig_sr=original_sr, target_sr=sr)
    return np.nan_to_num(y), sr


def _stems(ctx):
    dependencies = ctx.get("dependencies", {})
    separation = dependencies.get("separation", dependencies.get("separate", {}))
    return separation.get("stems", {})


def _source(ctx):
    stems = _stems(ctx)
    for key in ("vocals", "harmonic"):
        path = stems.get(key)
        if path and Path(path).exists():
            return path, key
    return ctx["audio_path"], "mixed_audio"


def _pitch_track(path, config, duration, backend="yin"):
    import librosa
    from scipy.ndimage import median_filter

    sr = 16000
    hop = max(1, round(sr * float(config.get("hop_seconds", 0.02))))
    frame_length = 2048
    y, _ = _load(path, sr)
    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)))
    fmin = float(config.get("fmin", 65.0))
    fmax = float(config.get("fmax", 1000.0))
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
    if backend == "torchcrepe":
        import torch
        import torchcrepe
        with torch.inference_mode():
            pitch, periodicity = torchcrepe.predict(
                torch.from_numpy(y)[None], sr, hop, fmin, fmax,
                model=config.get("crepe_model", "tiny"), batch_size=int(config.get("batch_size", 512)),
                device=config.get("device", "cpu"), return_periodicity=True,
            )
        f0 = pitch.detach().cpu().numpy().reshape(-1)
        quality = periodicity.detach().cpu().numpy().reshape(-1)
    else:
        f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length, hop_length=hop)
        # Normalized autocorrelation at the estimated lag, processed in small batches.
        frames = librosa.util.frame(np.pad(y, frame_length // 2), frame_length=frame_length, hop_length=hop)
        quality = np.zeros(len(f0), dtype=np.float32)
        for begin in range(0, len(f0), 256):
            batch = frames[:, begin:begin + 256].T.copy()
            batch -= batch.mean(axis=1, keepdims=True)
            ac = np.fft.irfft(np.abs(np.fft.rfft(batch, n=2 * frame_length, axis=1)) ** 2, axis=1)
            lags = np.clip(np.rint(sr / f0[begin:begin + len(batch)]).astype(int), 1, frame_length - 1)
            quality[begin:begin + len(batch)] = ac[np.arange(len(batch)), lags] / np.maximum(ac[:, 0], 1e-12)
    n = min(len(f0), len(quality), len(rms))
    times = np.arange(n) * hop / sr
    keep = times < duration
    times, f0, quality, rms = times[keep], f0[:n][keep], quality[:n][keep], rms[:n][keep]
    threshold = float(config.get("periodicity_threshold", 0.55 if backend == "yin" else 0.35))
    quality = median_filter(quality, size=3, mode="nearest")
    energy_gate = max(float(config.get("silence_rms", 0.0001)), float(np.max(rms, initial=0)) * float(config.get("relative_energy_gate", 0.025)))
    valid = (rms >= energy_gate) & (quality >= threshold) & np.isfinite(f0)
    midi = librosa.hz_to_midi(np.maximum(f0, 1e-9))
    return {"times": times, "f0": f0, "midi": midi, "quality": quality, "valid": valid, "y": y, "sr": sr, "hop": hop}


def _notes_from_track(track, duration, source, config):
    import librosa
    from scipy.ndimage import median_filter
    times, valid = track["times"], track["valid"]
    if not len(times) or not valid.any():
        return []
    hop, sr = track["hop"], track["sr"]
    midi = median_filter(track["midi"], size=3, mode="nearest")
    onset_frames = set(librosa.onset.onset_detect(
        y=track["y"], sr=sr, hop_length=hop, backtrack=False, units="frames"
    ).tolist())
    minimum = float(config.get("minimum_note_seconds", 0.08))
    note_frames = []
    begin = None
    for index in range(len(times)):
        if not valid[index]:
            if begin is not None:
                note_frames.append((begin, index))
                begin = None
            continue
        if begin is None:
            begin = index
            continue
        previous = float(np.median(midi[max(begin, index - 3):index]))
        pitch_change = abs(midi[index] - previous) >= float(config.get("note_change_semitones", 0.9))
        repeated_onset = index in onset_frames and times[index] - times[begin] >= minimum
        if (pitch_change or repeated_onset) and times[index] - times[begin] >= minimum:
            note_frames.append((begin, index))
            begin = index
    if begin is not None:
        note_frames.append((begin, len(times)))
    notes = []
    for begin, end in note_frames:
        start_time = float(times[begin])
        end_time = min(duration, float(times[end]) if end < len(times) else float(times[-1] + hop / sr))
        if end_time - start_time < minimum:
            continue
        notes.append({"start": round(start_time, 6), "end": round(end_time, 6), "midi": int(round(float(np.median(midi[begin:end])))), "source": source, "status": "estimated", "quality": {"periodicity": float(np.mean(track["quality"][begin:end])), "calibrated": False}})
    return notes


def _contour(track, source):
    output = []
    for time, f0, midi, quality, valid in zip(track["times"], track["f0"], track["midi"], track["quality"], track["valid"]):
        output.append({"time": round(float(time), 6), "f0_hz": float(f0) if valid else None,
                       "midi_float": float(midi) if valid else None, "valid": bool(valid),
                       "source": source, "quality": {"periodicity": float(quality), "calibrated": False}})
    return output


def _baseline_melody(ctx):
    config = ctx.get("config", {})
    path, source = _source(ctx)
    track = _pitch_track(path, config, float(ctx["duration"]))
    result = {"status": "partial", "data": {"notes": _notes_from_track(track, float(ctx["duration"]), source, config), "contour": _contour(track, source)},
            "warnings": ["Baseline YIN melody is a monophonic estimate; polyphonic accompaniment and backing vocals can cause wrong melody/octave selection."],
            "provenance": {"engine": "librosa-yin-onset", "version": _version("librosa"), "model_id": None, "backend": "baseline", "source": source}}
    return _add_instrumental_fallback(ctx, result)


def _add_instrumental_fallback(ctx, result):
    """Use an explicitly uncertain other-stem candidate only in long unvoiced gaps.

    This is temporal routing, not a claim to recognize the composer's lead instrument.
    Confidence rejection still applies and unknown frames remain unknown.
    """
    config = ctx.get("config", {})
    other = _stems(ctx).get("other")
    contour = result["data"]["contour"]
    if not config.get("instrumental_fallback", True) or result["provenance"].get("source") != "vocals" or not other or not Path(other).is_file() or not contour:
        return result
    gap_mask = np.zeros(len(contour), dtype=bool)
    minimum_gap = float(config.get("instrumental_gap_seconds", 1.0))
    begin = None
    duration = float(ctx["duration"])
    for i in range(len(contour) + 1):
        if i < len(contour) and not contour[i]["valid"]:
            if begin is None:
                begin = i
        elif begin is not None:
            end_time = contour[i]["time"] if i < len(contour) else duration
            if end_time - contour[begin]["time"] >= minimum_gap:
                gap_mask[begin:i] = True
            begin = None
    if not gap_mask.any():
        return result
    other_track = _pitch_track(other, config, duration)
    n = min(len(gap_mask), len(other_track["valid"]))
    routed = np.zeros(len(other_track["valid"]), dtype=bool)
    routed[:n] = gap_mask[:n]
    other_track["valid"] &= routed
    candidate_source = "other_instrumental_candidate"
    candidate_contour = _contour(other_track, candidate_source)
    changed = 0
    for i in range(n):
        if candidate_contour[i]["valid"]:
            contour[i] = candidate_contour[i]
            changed += 1
    result["data"]["notes"].extend(_notes_from_track(other_track, duration, candidate_source, config))
    result["data"]["notes"].sort(key=lambda note: (note["start"], note["end"]))
    result["provenance"]["instrumental_fallback"] = {"engine": "librosa-yin-onset", "minimum_unvoiced_gap_seconds": minimum_gap, "candidate_frames": changed}
    if changed:
        result["status"] = "partial"
        result["warnings"].append("Long vocal gaps use a periodic other-stem candidate; this does not establish which instrument carries the intended melody.")
    return result


def _basic_pitch_melody(ctx):
    from basic_pitch import build_icassp_2022_model_path, FilenameSuffix
    from basic_pitch.inference import predict
    config = ctx.get("config", {})
    path, source = _source(ctx)
    duration = float(ctx["duration"])
    pitch_backend = config.get("pitch_backend", "yin")
    track = _pitch_track(path, config, duration, backend=pitch_backend)
    model_path = config.get("basic_pitch_model") or str(build_icassp_2022_model_path(FilenameSuffix.onnx))
    _, _, candidates = predict(
        path, model_or_model_path=model_path,
        onset_threshold=float(config.get("onset_threshold", 0.5)),
        frame_threshold=float(config.get("frame_threshold", 0.3)),
        minimum_note_length=1000 * float(config.get("minimum_note_seconds", 0.08)),
        minimum_frequency=float(config.get("fmin", 65)), maximum_frequency=float(config.get("fmax", 1000)),
    )
    accepted = []
    for candidate in candidates:
        start, end, pitch, amplitude = map(float, candidate[:4])
        start, end = max(0.0, start), min(duration, end)
        if end <= start:
            continue
        in_note = (track["times"] >= start) & (track["times"] < end)
        voiced = in_note & track["valid"]
        if not voiced.any():
            continue
        agreement = float(np.mean(np.abs(track["midi"][voiced] - pitch) <= 1.5))
        coverage = float(np.sum(voiced) / max(1, np.sum(in_note)))
        if agreement < float(config.get("note_pitch_agreement", 0.5)) or coverage < 0.25:
            continue
        accepted.append({"start": round(start, 6), "end": round(end, 6), "midi": int(pitch), "source": source, "status": "estimated", "quality": {"pitch_agreement": agreement, "voiced_coverage": coverage, "note_amplitude": amplitude, "calibrated": False}})
    # Keep the most consistent candidate where a model returns overlapping pitches.
    accepted.sort(key=lambda n: (n["start"], -n["quality"]["pitch_agreement"]))
    notes = []
    for note in accepted:
        if notes and note["start"] < notes[-1]["end"] and note["midi"] != notes[-1]["midi"]:
            if note["start"] > notes[-1]["start"] + 0.06:
                notes[-1]["end"] = note["start"]
            else:
                continue
        notes.append(note)
    messages = ["Automatic vocal-priority melody selection is an estimate, not a guaranteed lead-vocal transcription."]
    if source != "vocals":
        messages.append("No separated vocals available; note candidates were estimated from harmonic/mixed audio.")
    result = {"status": "succeeded" if source == "vocals" else "partial", "data": {"notes": notes, "contour": _contour(track, source)}, "warnings": messages,
              "provenance": {"engine": "basic-pitch+" + pitch_backend, "model_id": "icassp_2022", "version": _version("basic-pitch"), "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest() if Path(model_path).is_file() else None, "source": source}}
    return _add_instrumental_fallback(ctx, result)


def run_melody(ctx):
    config = ctx.get("config", {})
    backend = config.get("backend", "baseline")
    if backend in {"baseline", "yin"}:
        return _baseline_melody(ctx)
    if backend not in {"auto", "basic_pitch", "basic-pitch", "torchcrepe"}:
        raise ValueError(f"Unsupported melody backend: {backend}")
    try:
        if backend == "torchcrepe":
            ctx = {**ctx, "config": {**config, "pitch_backend": "torchcrepe"}}
        return _basic_pitch_melody(ctx)
    except Exception as exc:
        if not config.get("fallback", True):
            raise
        result = _baseline_melody(ctx)
        result["warnings"].insert(0, f"Neural melody backend unavailable/failed; using YIN fallback: {type(exc).__name__}: {exc}")
        result["provenance"]["requested_backend"] = backend
        return result


def _baseline_harmony(ctx):
    import librosa
    from scipy.ndimage import median_filter
    config = ctx.get("config", {})
    y, sr = _load(ctx["audio_path"], 22050)
    duration = float(ctx["duration"])
    if len(y) < 4096:
        y = np.pad(y, (0, 4096 - len(y)))
    hop = 1024
    spectrum = np.abs(librosa.stft(y, n_fft=4096, hop_length=hop))
    chroma = librosa.feature.chroma_stft(S=spectrum ** 2, sr=sr, n_fft=4096, hop_length=hop)
    chroma = median_filter(chroma, size=(1, 5), mode="nearest")
    chroma /= np.maximum(np.linalg.norm(chroma, axis=0, keepdims=True), 1e-9)
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    templates, labels, pitch_classes = [], [], []
    for root in range(12):
        for quality, intervals in (("maj", (0, 4, 7)), ("min", (0, 3, 7))):
            pcs = sorted((root + interval) % 12 for interval in intervals)
            template = np.zeros(12)
            template[pcs] = 1 / math.sqrt(3)
            templates.append(template)
            labels.append(names[root] + ":" + quality)
            pitch_classes.append(pcs)
    scores = np.array(templates) @ chroma
    order = np.argsort(scores, axis=0)
    best = order[-1]
    maxima = scores[best, np.arange(scores.shape[1])]
    margins = maxima - scores[order[-2], np.arange(scores.shape[1])]
    rms = librosa.feature.rms(S=spectrum, frame_length=4096, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    codes = np.where((maxima >= float(config.get("minimum_similarity", 0.65))) & (margins >= float(config.get("minimum_margin", 0.015))), best, -1)
    codes[rms < max(1e-5, np.max(rms, initial=0) * 0.005)] = -2
    # Chord IDs are nominal categories: a numeric median would invent another chord.
    smoothed = codes.copy()
    for i in range(1, len(codes) - 1):
        if codes[i - 1] == codes[i + 1]:
            smoothed[i] = codes[i - 1]
    codes = smoothed
    output = []
    begin = 0
    for end in range(1, len(codes) + 1):
        if end < len(codes) and codes[end] == codes[begin]:
            continue
        start_time = float(times[begin])
        end_time = min(duration, float(times[end]) if end < len(times) else duration)
        code = int(codes[begin])
        if end_time > start_time:
            output.append({"start": round(start_time, 6), "end": round(end_time, 6),
                           "label": labels[code] if code >= 0 else ("N" if code == -2 else None),
                           "pitch_classes": pitch_classes[code] if code >= 0 else [],
                           "status": "estimated" if code >= 0 else ("no_chord" if code == -2 else "unknown"),
                           "quality": {"template_similarity": float(np.mean(maxima[begin:end])), "margin": float(np.mean(margins[begin:end])), "calibrated": False}})
        begin = end
    return {"status": "partial", "data": {"segments": output}, "warnings": ["Chroma-template baseline recognizes major/minor triads only; extensions, inversions and polyphonic mixtures require the BTC backend and evaluation."],
            "provenance": {"engine": "librosa-chroma-template", "version": _version("librosa"), "model_id": None, "vocabulary": "24_major_minor_plus_unknown_silence", "backend": "baseline"}}


def _btc_harmony(ctx):
    import mir_eval
    config = ctx.get("config", {})
    repo = Path(config.get("btc_repo", "models/BTC-ISMIR19")).resolve()
    checkpoint = repo / "test" / "btc_model_large_voca.pt"
    if not (repo / "btc_model.py").is_file() or not checkpoint.is_file():
        raise FileNotFoundError("BTC requires btc_repo with btc_model.py and test/btc_model_large_voca.pt; run model setup first.")
    work = Path(ctx["work_dir"]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    output = work / "btc-segments.json"
    script = Path(__file__).resolve().parents[1] / "btc_worker.py"
    process = subprocess.run([config.get("python_executable", sys.executable), str(script), str(repo), str(Path(ctx["audio_path"]).resolve()), str(output)], cwd=work, capture_output=True, text=True, timeout=float(config.get("timeout_seconds", 1800)))
    (work / "btc.log").write_text(process.stdout + "\n" + process.stderr)
    if process.returncode:
        raise RuntimeError("BTC subprocess failed; inspect btc.log: " + process.stderr[-600:])
    if not output.exists():
        raise RuntimeError("BTC did not produce chord segments")
    import json
    segments = []
    for item in json.loads(output.read_text()):
        start, end = max(0, float(item["start"])), min(float(ctx["duration"]), float(item["end"]))
        label = item["label"]
        if end <= start:
            continue
        if label in {"N", "X"}:
            pcs, state = [], "no_chord" if label == "N" else "unknown"
        else:
            root, bitmap, _ = mir_eval.chord.encode(label, reduce_extended_chords=True)
            pcs = np.flatnonzero(mir_eval.chord.rotate_bitmap_to_root(bitmap, root)).tolist()
            state = "estimated"
        segments.append({"start": start, "end": end, "label": label, "pitch_classes": pcs, "status": state})
    return {"status": "succeeded", "data": {"segments": segments}, "warnings": [], "provenance": {"engine": "BTC-ISMIR19", "model_id": "btc_model_large_voca", "model_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "vocabulary": 170, "time_mapping": "sample_based_chunk_offsets"}}


def run_harmony(ctx):
    config = ctx.get("config", {})
    backend = config.get("backend", "baseline")
    if backend in {"baseline", "chroma"}:
        return _baseline_harmony(ctx)
    if backend not in {"auto", "btc"}:
        raise ValueError(f"Unsupported harmony backend: {backend}")
    try:
        return _btc_harmony(ctx)
    except Exception as exc:
        if not config.get("fallback", True):
            raise
        result = _baseline_harmony(ctx)
        result["warnings"].insert(0, f"BTC unavailable/failed; using triad-template fallback: {type(exc).__name__}: {exc}")
        result["provenance"]["requested_backend"] = backend
        return result


def _rhythm_data(beats, downbeats, duration, config):
    beats = np.array(sorted(set(float(b) for b in beats if math.isfinite(float(b)) and 0 <= float(b) < duration)))
    downbeats = np.array([float(b) for b in downbeats if math.isfinite(float(b)) and 0 <= float(b) < duration])
    marked = set()
    for downbeat in downbeats:
        if len(beats):
            index = int(np.argmin(np.abs(beats - downbeat)))
            if abs(beats[index] - downbeat) <= float(config.get("downbeat_tolerance_seconds", 0.07)):
                marked.add(index)
    output = []
    bar, position = None, None
    for index, time in enumerate(beats):
        if index in marked:
            bar = 0 if bar is None else bar + 1
            position = 1
        elif position is not None:
            position += 1
        output.append({"time": round(float(time), 6), "is_downbeat": (index in marked) if len(downbeats) else None, "bar_index": bar, "beat_in_bar": position})
    tempo = []
    for i in range(max(0, len(beats) - 1)):
        interval = beats[i + 1] - beats[i]
        if interval > 0:
            tempo.append({"start": float(beats[i]), "end": float(beats[i + 1]), "bpm": float(60 / interval), "beat_unit": "estimated_beat"})
    return {"beats": output, "tempo": tempo, "meter": []}


def _baseline_rhythm(ctx):
    import librosa
    y, sr = _load(ctx["audio_path"], 22050)
    if len(y) < 2048 or float(np.max(np.abs(y), initial=0)) < 1e-5:
        beats = []
    else:
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        if np.max(onset, initial=0) < 1e-6:
            beats = []
        else:
            _, frames = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=512, trim=True)
            beats = librosa.frames_to_time(frames, sr=sr, hop_length=512)
    return {"status": "partial", "data": _rhythm_data(beats, [], float(ctx["duration"]), ctx.get("config", {})),
            "warnings": ["Librosa baseline estimates beats only; first beats, bar positions and time signature remain unknown."],
            "provenance": {"engine": "librosa-beat-track", "version": _version("librosa"), "model_id": None, "backend": "baseline"}}


def _beat_this(ctx):
    from beat_this.inference import Audio2Beats
    import torch
    config = ctx.get("config", {})
    checkpoint = config.get("checkpoint", config.get("model", "final0"))
    checkpoint_path = Path(str(checkpoint)).expanduser()
    if not checkpoint_path.is_file() and "://" not in str(checkpoint):
        cached = Path(torch.hub.get_dir()) / "checkpoints" / f"beat_this-{checkpoint}.ckpt"
        if cached.is_file():
            checkpoint_path = cached
    resolved_checkpoint = str(checkpoint_path.resolve()) if checkpoint_path.is_file() else checkpoint
    # Avoid torchaudio audio-backend/torchcodec variability: decode through soundfile.
    y, sr = _load(ctx["audio_path"], 22050)
    if float(np.max(np.abs(y), initial=0)) < 1e-5:
        beats, downbeats = [], []
    else:
        model = Audio2Beats(checkpoint_path=resolved_checkpoint, device=config.get("device", "cpu"), dbn=False)
        with torch.inference_mode():
            beats, downbeats = model(y, sr)
    return {"status": "succeeded", "data": _rhythm_data(beats, downbeats, float(ctx["duration"]), config), "warnings": ["Beat grouping is estimated; a full notated time signature is not inferred from beat count alone."],
            "provenance": {"engine": "beat_this", "model_id": config.get("model", "final0"), "version": _version("beat-this"), "device": config.get("device", "cpu"), "dbn": False,
                           "model_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() if checkpoint_path.is_file() else None}}


def run_rhythm(ctx):
    config = ctx.get("config", {})
    backend = config.get("backend", "baseline")
    if backend in {"baseline", "librosa"}:
        return _baseline_rhythm(ctx)
    if backend not in {"auto", "beat_this", "beat-this"}:
        raise ValueError(f"Unsupported rhythm backend: {backend}")
    try:
        return _beat_this(ctx)
    except Exception as exc:
        if not config.get("fallback", True):
            raise
        result = _baseline_rhythm(ctx)
        result["warnings"].insert(0, f"Beat This unavailable/failed; using librosa beat fallback: {type(exc).__name__}: {exc}")
        result["provenance"]["requested_backend"] = backend
        return result
