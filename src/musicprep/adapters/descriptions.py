"""Section-level acoustic descriptions and explicitly configured model inference.

The dependency-light baseline intentionally does not guess genre, instruments or
emotion from loudness. Model inference runs in an isolated subprocess because
some upstream research implementations write/remove files in their cwd.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from scipy import signal
import soundfile as sf


def _finite(value, name):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _sections(ctx):
    duration = _finite(ctx["duration"], "duration")
    if duration <= 0:
        raise ValueError("duration must be positive")
    source = ctx.get("dependencies", {}).get("sections", {}).get("segments", [])
    warnings = []
    sections = []
    for index, item in enumerate(source):
        try:
            start = _finite(item.get("start", item.get("applies_start")), "start")
            end = _finite(item.get("end", item.get("applies_end")), "end")
            if not 0 <= start < end <= duration + 0.02:
                raise ValueError("section outside audio timeline")
            sections.append({"id": str(item.get("id", f"section_{index:03d}")),
                             "start": start, "end": min(end, duration)})
        except (ValueError, TypeError, AttributeError) as exc:
            warnings.append(f"section {index} ignored: {exc}")
    if not sections:
        window = _finite(ctx.get("config", {}).get("window_seconds", 30), "window_seconds")
        if window <= 0:
            raise ValueError("window_seconds must be positive")
        sections = [{"id": None, "start": float(start),
                     "end": min(float(start + window), duration)}
                    for i, start in enumerate(np.arange(0, duration, window))]
        warnings.append("No usable structural sections; descriptions use fixed time windows, not inferred verse/chorus labels.")
    return sections, warnings


def _read_clip(path, start, end, target_sr=22050):
    with sf.SoundFile(path) as audio:
        sr = audio.samplerate
        audio.seek(min(round(start * sr), len(audio)))
        clip = audio.read(max(0, round((end - start) * sr)), dtype="float32", always_2d=True)
    if not len(clip) or not np.isfinite(clip).all():
        raise ValueError("empty or non-finite audio segment")
    mono = clip.mean(axis=1)
    if sr != target_sr:
        divisor = math.gcd(sr, target_sr)
        mono = signal.resample_poly(mono, target_sr // divisor, sr // divisor)
    return np.asarray(mono, dtype=np.float32), target_sr


def acoustic_features(audio, sr):
    """Physical summaries; thresholds describe candidate timbre, not emotion."""
    rms = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2)))
    features = {"rms_dbfs": round(20 * math.log10(max(rms, 1e-8)), 4),
                "silent": rms < 1e-5}
    if features["silent"]:
        features.update(spectral_centroid_hz=0.0, spectral_flatness=0.0,
                        envelope_variation=0.0, onset_density_per_second=0.0)
        return features, []
    size = min(2048, len(audio))
    frequencies, _, spectrum = signal.stft(audio, sr, nperseg=size,
                                           noverlap=size // 2, boundary=None)
    magnitude = np.abs(spectrum).astype(np.float64)
    energy = magnitude.sum(axis=0)
    valid = energy > max(float(energy.max()) * 0.001, 1e-10)
    centroid = (magnitude * frequencies[:, None]).sum(axis=0) / np.maximum(energy, 1e-12)
    power = magnitude ** 2 + 1e-20
    flatness = np.exp(np.log(power).mean(axis=0)) / power.mean(axis=0)
    envelope = np.sqrt((magnitude ** 2).sum(axis=0))
    flux = np.maximum(np.diff(magnitude, axis=1), 0).sum(axis=0)
    hop_seconds = (size - size // 2) / sr
    threshold = float(np.median(flux) + 2 * np.median(np.abs(flux - np.median(flux)))) if len(flux) else 0
    peaks = signal.find_peaks(flux, height=max(threshold, 1e-8),
                              distance=max(1, round(0.08 / hop_seconds)))[0] if len(flux) else []
    features.update(spectral_centroid_hz=round(float(np.median(centroid[valid])), 4),
                    spectral_flatness=round(float(np.median(flatness[valid])), 6),
                    envelope_variation=round(float(envelope.std() / max(envelope.mean(), 1e-12)), 6),
                    onset_density_per_second=round(len(peaks) / (len(audio) / sr), 4))
    labels = []
    if features["spectral_centroid_hz"] > 2800:
        labels.append("bright")
    elif features["spectral_centroid_hz"] < 1100:
        labels.append("dark")
    if features["spectral_flatness"] > 0.15:
        labels.append("noise_like")
    if features["envelope_variation"] < 0.12:
        labels.append("steady_envelope")
    return features, [{"label": label, "status": "candidate", "source": "acoustic_heuristic"}
                      for label in labels]


def _tags(value, field):
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    result = []
    seen = set()
    for entry in value:
        if isinstance(entry, str):
            entry = {"label": entry}
        if not isinstance(entry, dict) or not isinstance(entry.get("label"), str) or not entry["label"].strip():
            raise ValueError(f"invalid {field} label")
        clean = {"label": entry["label"].strip(), "status": "candidate"}
        if "score" in entry:
            score = _finite(entry["score"], f"{field} score")
            if not 0 <= score <= 1:
                raise ValueError(f"{field} score outside [0,1]")
            clean["score"] = score
        if clean["label"] not in seen:
            result.append(clean)
            seen.add(clean["label"])
    return result


def _validate_model_result(result, engine):
    if not isinstance(result, dict):
        raise ValueError("model result must be an object")
    clean = {}
    for field in ("genre", "instruments", "textures"):
        if field in result:
            clean[field] = [dict(tag, source=engine) for tag in _tags(result[field], field)]
    if "emotion" in result:
        emotion = result["emotion"]
        if not isinstance(emotion, dict):
            raise ValueError("emotion must be an object")
        scale = emotion.get("scale", [1, 9])
        if not isinstance(scale, list) or len(scale) != 2:
            raise ValueError("emotion scale must contain lower and upper bounds")
        lower, upper = [_finite(x, "emotion scale") for x in scale]
        if lower >= upper:
            raise ValueError("invalid emotion scale")
        values = {}
        for key in ("valence", "arousal"):
            value = _finite(emotion.get(key), key)
            if not lower <= value <= upper:
                raise ValueError(f"{key} outside model scale")
            values[key] = value
        clean["emotion"] = dict(values, scale=[lower, upper], status="estimated", source=engine,
                                 moods=[dict(tag, source=engine) for tag in _tags(emotion.get("moods", []), "moods")])
    if not clean:
        raise ValueError("model returned no supported description fields")
    return clean


def _run_model_batch(engine, cfg, entries, work):
    cfg = copy.deepcopy(cfg)
    # Resolve caller-relative configuration before the subprocess changes cwd.
    for name in ("repo", "weights", "embedding"):
        if cfg.get(name):
            cfg[name] = str(Path(cfg[name]).expanduser().resolve())
    if cfg.get("python") and ("/" in cfg["python"] or "\\" in cfg["python"]):
        # Resolving this symlink would bypass the selected virtualenv.
        cfg["python"] = str(Path(cfg["python"]).expanduser().absolute())
    for name in ("genre", "instruments"):
        for field in ("graph", "metadata"):
            if isinstance(cfg.get(name), dict) and cfg[name].get(field):
                cfg[name][field] = str(Path(cfg[name][field]).expanduser().resolve())
    if cfg.get("model_id") and Path(cfg["model_id"]).expanduser().exists():
        cfg["model_id"] = str(Path(cfg["model_id"]).expanduser().resolve())
    directory = Path(tempfile.mkdtemp(prefix=f"{engine}-", dir=work)).resolve()
    request = directory / "request.json"
    output = directory / "result.json"
    request.write_text(json.dumps({"config": cfg, "entries": entries}, ensure_ascii=False), encoding="utf-8")
    log_path = directory / "worker.log"
    args = [str(cfg.get("python", sys.executable)), str(Path(__file__).resolve()),
            "--worker", engine, "--request", str(request), "--output", str(output)]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", str(cfg.get("threads", 2)))
    env.setdefault("TF_NUM_INTRAOP_THREADS", str(cfg.get("threads", 2)))
    env.setdefault("TF_NUM_INTEROP_THREADS", "1")
    if not cfg.get("allow_download", False):
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(args, cwd=directory, env=env, stdout=log, stderr=log, check=True,
                       timeout=float(cfg.get("timeout_seconds", 1800)))
    result = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or not isinstance(result.get("results"), list):
        raise ValueError("model worker result must contain a results array")
    return result, str(log_path)


def _model_provenance(engine, cfg):
    ids = {"music2emo": "AMAAI-Lab/Music2Emotion:J_all+m-a-p/MERT-v1-95M",
           "essentia": "MTG/Discogs-EffNet+configured-classification-heads",
           "qwen2_audio": "Qwen/Qwen2-Audio-7B-Instruct"}
    result = {"model_id": cfg.get("model_id", ids[engine]), "files": []}
    files = [cfg.get("weights"), cfg.get("embedding")]
    if engine == "music2emo" and cfg.get("repo"):
        repo = Path(cfg["repo"]).expanduser().resolve()
        files.append(cfg.get("weights", repo / "saved_models/J_all.ckpt"))
        try:
            result["source_revision"] = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            result["source_revision"] = None
    for field in ("genre", "instruments"):
        if isinstance(cfg.get(field), dict):
            files.extend([cfg[field].get("graph"), cfg[field].get("metadata")])
    for filename in dict.fromkeys(str(value) for value in files if value):
        path = Path(filename).expanduser().resolve()
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            result["files"].append({"path": str(path), "sha256": digest.hexdigest()})
    return result


def run_descriptions(ctx):
    cfg = ctx.get("config", {})
    backend = cfg.get("backend", "auto")
    allowed = {"auto", "baseline", "music2emo", "essentia", "qwen2_audio", "models"}
    if backend not in allowed:
        raise ValueError(f"unsupported descriptions backend: {backend}")
    sections, warnings = _sections(ctx)
    work = Path(ctx["work_dir"]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    segments, entries = [], []
    for index, section in enumerate(sections):
        record = {"section_id": section["id"], "analysis_start": section["start"],
                  "analysis_end": section["end"], "applies_start": section["start"],
                  "applies_end": section["end"], "genre": [], "instruments": [], "textures": [],
                  "emotion": {"valence": None, "arousal": None, "scale": None,
                              "moods": [], "status": "unknown"}, "status": "partial"}
        try:
            audio, sr = _read_clip(ctx["audio_path"], section["start"], section["end"])
            record["acoustic_features"], record["textures"] = acoustic_features(audio, sr)
            clip_path = work / f"segment_{index:04d}.wav"
            sf.write(clip_path, audio, sr, subtype="FLOAT")
            entries.append({"index": index, "audio_path": str(clip_path)})
        except (OSError, ValueError, RuntimeError) as exc:
            warnings.append(f"section {section['id']}: acoustic analysis failed: {exc}")
            record["status"] = "failed"
        segments.append(record)
    configured = cfg.get("model_backends", {})
    if not isinstance(configured, dict):
        raise ValueError("model_backends must be an object")
    engines = [] if backend == "baseline" else [name for name in ("essentia", "music2emo", "qwen2_audio")
                                               if name in configured and (backend in {"auto", "models"} or name == backend)]
    if backend not in {"auto", "baseline", "models"} and backend not in configured:
        if not cfg.get("fallback", True):
            raise ValueError(f"{backend} configuration is missing")
        warnings.append(f"{backend} is not configured; acoustic-only fallback used.")
    provenance = {"engine": "scipy_acoustic_features", "model_id": None, "method_version": "1.0",
                  "analysis_sample_rate": 22050, "models": {},
                  "limitations": ["Acoustic texture tags are heuristic candidates; model labels and emotions are estimates, not perceptual ground truth."],
                  "texture_thresholds": {"bright_centroid_hz": 2800, "dark_centroid_hz": 1100,
                  "noise_flatness": 0.15, "steady_envelope_cv": 0.12}}
    for engine in engines:
        try:
            results, log = _run_model_batch(engine, configured[engine], entries, work)
            provenance["models"][engine] = dict(_model_provenance(engine, configured[engine]),
                                                status="executed", log_path=log)
            seen = set()
            for item in results.get("results", []):
                try:
                    index = item["index"]
                    if not isinstance(index, int) or index not in {e["index"] for e in entries} or index in seen:
                        raise ValueError("invalid or duplicate segment index")
                    seen.add(index)
                    if "error" in item:
                        raise ValueError(str(item["error"]))
                    clean = _validate_model_result(item["data"], engine)
                    if "textures" in clean:
                        clean["textures"] = segments[index]["textures"] + clean["textures"]
                    segments[index].update(clean)
                except (ValueError, KeyError, TypeError) as exc:
                    warnings.append(f"{engine} segment result rejected: {exc}")
            missing = {e["index"] for e in entries} - seen
            if missing:
                warnings.append(f"{engine} did not return {len(missing)} segment(s).")
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            if not cfg.get("fallback", True):
                raise
            warnings.append(f"{engine} unavailable/failed: {type(exc).__name__}: {exc}")
            provenance["models"][engine] = {"status": "failed"}
    for segment in segments:
        if segment["genre"] and segment["instruments"] and segment["emotion"]["status"] == "estimated":
            segment["status"] = "succeeded"
    complete = bool(segments) and all(s["status"] == "succeeded" for s in segments)
    missing = []
    for field in ("genre", "instruments", "emotion"):
        count = sum((segment[field].get("status") != "estimated" if field == "emotion" else not segment[field])
                    for segment in segments)
        if count:
            missing.append(f"{field} ({count}/{len(segments)} segments)")
    if missing:
        warnings.append("Unresolved fields remain unknown: " + ", ".join(missing) + ".")
    executed = [name for name, model in provenance["models"].items() if model["status"] == "executed"]
    if executed:
        provenance["engine"] = "+".join(["scipy_acoustic_features", *executed])
        provenance["model_id"] = "+".join(provenance["models"][name]["model_id"] for name in executed)
    return {"status": "succeeded" if complete else "partial", "data": {"segments": segments},
            "warnings": warnings, "provenance": provenance}


def _music2emo_predictor(cfg, root):
    repo = Path(cfg["repo"]).expanduser().resolve()
    weights = Path(cfg.get("weights", repo / "saved_models/J_all.ckpt")).expanduser().resolve()
    if not weights.is_file() or not (repo / "inference/data").is_dir():
        raise FileNotFoundError("Music2Emotion checkpoint or inference/data assets missing")
    sys.path.insert(0, str(repo))
    # Upstream imports gradio at module level and resolves inference/data from cwd.
    (root / "inference").symlink_to(repo / "inference", target_is_directory=True)
    os.chdir(root)
    from music2emo import Music2emo
    import torch
    torch.set_num_threads(int(cfg.get("threads", 2)))
    model = Music2emo(model_weights=str(weights))

    def predict(path):
        with torch.inference_mode():
            output = model.predict(path, threshold=float(cfg.get("threshold", 0.5)))
        return {"emotion": {"valence": output["valence"], "arousal": output["arousal"],
                            "scale": [1, 9], "moods": output.get("predicted_moods", [])}}
    return predict


def _essentia_predictor(cfg, root):
    from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D
    embedding = TensorflowPredictEffnetDiscogs(graphFilename=str(Path(cfg["embedding"]).resolve()),
                                               output="PartitionedCall:1")
    heads = {}
    for field in ("genre", "instruments"):
        if field not in cfg:
            continue
        spec = cfg[field]
        metadata = json.loads(Path(spec["metadata"]).read_text(encoding="utf-8"))
        classes = metadata["classes"]
        schema = metadata.get("schema", {})
        inputs = schema.get("inputs", [{}])
        outputs = schema.get("outputs", [{}])
        head = TensorflowPredict2D(graphFilename=str(Path(spec["graph"]).resolve()),
                                   input=spec.get("input", inputs[0].get("name", "serving_default_model_Placeholder")),
                                   output=spec.get("output", outputs[0].get("name", "PartitionedCall:0")))
        heads[field] = (head, classes, spec)
    if not heads:
        raise ValueError("Essentia requires a genre or instruments classification head")

    def predict(path):
        audio = MonoLoader(filename=path, sampleRate=16000, resampleQuality=4)()
        if len(audio) < 16000 * 6:
            audio = np.pad(audio, (0, 16000 * 6 - len(audio)))
        features = embedding(audio)
        result = {}
        for field, (head, classes, spec) in heads.items():
            values = np.asarray(head(features))
            mean = values.mean(axis=0) if values.ndim > 1 else values
            if len(mean) != len(classes):
                raise ValueError("Essentia output dimension does not match metadata classes")
            order = np.argsort(mean)[::-1][:int(spec.get("top_k", 5))]
            result[field] = [{"label": classes[int(i)], "score": float(mean[i])} for i in order
                             if mean[i] >= float(spec.get("threshold", 0.2))]
        return result
    return predict


def _qwen_predictor(cfg, root):
    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    model_id = cfg.get("model_id", "Qwen/Qwen2-Audio-7B-Instruct")
    local = not cfg.get("allow_download", False)
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=local)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id, local_files_only=local, torch_dtype="auto", device_map=cfg.get("device_map", "auto"))
    prompt = ('Describe only the audible music in this clip. Return a JSON object with three arrays of short English strings: '
              '"genre", "instruments", "textures". If uncertain use an empty array. Do not infer lyrics, artist, song identity, '
              'biography, or any facts outside the audio. Do not output valence/arousal numbers.')

    def predict(path):
        conversation = [{"role": "user", "content": [{"type": "audio", "audio_url": path},
                                                        {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audio, _ = _read_clip(path, 0, sf.info(path).duration, processor.feature_extractor.sampling_rate)
        inputs = processor(text=text, audios=[audio], return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=int(cfg.get("max_new_tokens", 256)), do_sample=False)
        answer = processor.batch_decode(outputs[:, inputs.input_ids.size(1):], skip_special_tokens=True)[0]
        start, end = answer.find("{"), answer.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Qwen2-Audio did not return a JSON object")
        return json.loads(answer[start:end + 1])
    return predict


def _worker(engine, request_path, output_path):
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    root = Path.cwd().resolve()
    factory = {"music2emo": _music2emo_predictor, "essentia": _essentia_predictor,
               "qwen2_audio": _qwen_predictor}[engine]
    predict = factory(request["config"], root)
    results = []
    for entry in request["entries"]:
        try:
            results.append({"index": entry["index"], "data": predict(entry["audio_path"])})
        except Exception as exc:  # Individual model inference errors must not drop other sections.
            results.append({"index": entry["index"], "error": f"{type(exc).__name__}: {exc}"})
        Path(output_path).write_text(json.dumps({"results": results}, ensure_ascii=False, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, choices=["music2emo", "essentia", "qwen2_audio"])
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    _worker(args.worker, args.request, args.output)
