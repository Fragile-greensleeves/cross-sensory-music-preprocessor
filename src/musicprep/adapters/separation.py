"""Source separation adapters. HPSS is explicitly not a four-stem substitute."""
from __future__ import annotations

import importlib.metadata
from pathlib import Path
import warnings

import numpy as np
import soundfile as sf


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load(path, sample_rate):
    import librosa
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != sample_rate:
        y = librosa.resample(y.T, orig_sr=sr, target_sr=sample_rate).T
    return y, sample_rate


def _baseline(ctx):
    import librosa
    sr = int(ctx.get("config", {}).get("sample_rate", 22050))
    y, _ = _load(ctx["audio_path"], sr)
    mono = y.mean(axis=1)
    work = Path(ctx["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    if len(mono) < 2048:
        mono = np.pad(mono, (0, 2048 - len(mono)))
    harmonic, percussive = librosa.effects.hpss(mono, n_fft=2048, hop_length=512)
    length = round(float(ctx["duration"]) * sr)
    paths = {}
    for name, audio in (("harmonic", harmonic[:length]), ("percussive", percussive[:length])):
        path = work / f"{name}.wav"
        sf.write(path, audio, sr, subtype="FLOAT")
        paths[name] = str(path.resolve())
    return {
        "status": "partial",
        "data": {"stems": paths, "sample_rate": sr, "separation_type": "harmonic_percussive"},
        "warnings": ["HPSS baseline extracts harmonic/percussive components, not vocals/drums/bass/other; melody source remains uncertain."],
        "provenance": {"engine": "librosa-hpss", "model_id": None, "version": _version("librosa"), "backend": "baseline"},
    }


def _demucs(ctx):
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    config = ctx.get("config", {})
    model_name = config.get("model", "htdemucs").removesuffix(".yaml")
    device = config.get("device", "cpu")
    torch.set_num_threads(int(config.get("threads", min(4, torch.get_num_threads()))))
    torch.manual_seed(int(config.get("seed", 0)))
    model = get_model(model_name).to(device).eval()
    y, sr = _load(ctx["audio_path"], int(model.samplerate))
    if y.shape[1] == 1 and model.audio_channels == 2:
        y = np.repeat(y, 2, axis=1)
    elif y.shape[1] != model.audio_channels:
        y = np.repeat(y.mean(axis=1, keepdims=True), model.audio_channels, axis=1)
    wav = torch.from_numpy(np.ascontiguousarray(y.T))
    reference = wav.mean(0)
    mean, std = reference.mean(), reference.std()
    work = Path(ctx["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    if float(std) < 1e-8:
        sources = torch.zeros((len(model.sources), *wav.shape), dtype=wav.dtype)
    else:
        with torch.inference_mode():
            sources = apply_model(
                model, ((wav - mean) / std)[None], device=device,
                shifts=int(config.get("shifts", 0)), split=True,
                overlap=float(config.get("overlap", 0.25)), progress=bool(config.get("progress", False)),
                num_workers=0,
            )[0].cpu()
        sources = sources * std + mean
    paths = {}
    for name, signal in zip(model.sources, sources):
        path = work / f"{name}.wav"
        sf.write(path, signal.numpy().T, sr, subtype="FLOAT")
        paths[name] = str(path.resolve())
    return {
        "status": "succeeded",
        "data": {"stems": paths, "sample_rate": sr, "separation_type": "four_stem" if len(paths) == 4 else "multi_stem"},
        "warnings": [],
        "provenance": {"engine": "demucs", "model_id": model_name, "version": _version("demucs"), "device": device, "shifts": int(config.get("shifts", 0)), "seed": int(config.get("seed", 0)), "independent_stem_normalization": False},
    }


def run_separation(ctx):
    config = ctx.get("config", {})
    backend = config.get("backend", "baseline")
    if backend not in {"baseline", "hpss", "auto", "demucs"}:
        raise ValueError(f"Unsupported separation backend: {backend}")
    if backend in {"baseline", "hpss"}:
        return _baseline(ctx)
    try:
        return _demucs(ctx)
    except Exception as exc:
        if not config.get("fallback", True):
            raise
        result = _baseline(ctx)
        result["warnings"].insert(0, f"Demucs unavailable/failed; using explicit HPSS fallback: {type(exc).__name__}: {exc}")
        result["provenance"]["requested_backend"] = backend
        return result
