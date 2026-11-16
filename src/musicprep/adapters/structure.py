"""Functional structure model and audibly grounded, unnamed baseline segments."""
from __future__ import annotations

from pathlib import Path
import importlib.metadata
import hashlib
import math
import numpy as np


def _baseline(ctx):
    from musicprep.adapters.events import _load
    import librosa
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    config = ctx.get("config", {})
    duration = float(ctx["duration"])
    min_length = max(1.0, float(config.get("minimum_section_seconds", 12)))
    max_length = max(min_length, float(config.get("fallback_window_seconds", 30)))
    y, sr = _load(ctx["audio_path"], 16000)
    boundaries = [0.0]
    if len(y) >= 2048 and float(np.max(np.abs(y), initial=0)) >= 1e-5:
        hop = 8000  # 0.5s frames; this is a coarse segmentation baseline, not note timing.
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=hop)
        feature = mfcc / np.maximum(np.std(mfcc, axis=1, keepdims=True), 1e-3)
        novelty = np.linalg.norm(np.diff(feature, axis=1), axis=0)
        novelty = gaussian_filter1d(novelty, 1)
        if len(novelty):
            threshold = float(np.percentile(novelty, 80))
            indices, _ = find_peaks(novelty, distance=max(1, round(min_length * 2)), prominence=max(0.1, threshold * 0.2), height=threshold)
            boundaries.extend(float((index + 1) * hop / sr) for index in indices if min_length <= (index + 1) * hop / sr <= duration - min_length)
    boundaries.append(duration)
    boundaries = sorted(set(boundaries))
    output = []
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        count = max(1, math.ceil((end - begin) / max_length))
        edges = np.linspace(begin, end, count + 1)
        for start, stop in zip(edges[:-1], edges[1:]):
            output.append({"id": f"section_{len(output) + 1:03d}", "start": float(start), "end": float(stop), "label": "unknown", "source": "baseline_change_points" if count == 1 else "fallback_window", "status": "unknown"})
    return {"status": "partial", "data": {"segments": output}, "warnings": ["Baseline splits by acoustic change and bounded windows; functional names such as verse/chorus remain unknown."], "provenance": {"engine": "librosa-mfcc-novelty", "version": importlib.metadata.version("librosa"), "model_id": None, "backend": "baseline"}}


def _allin1(ctx):
    import allin1
    from allin1.models import loaders
    config = ctx.get("config", {})
    work = Path(ctx["work_dir"]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    revision = config.get("model_revision", "379e5fd010b3fdd0ee8381ff8cbcfa51d70b5c19")
    downloaded = {}
    original_download = loaders.hf_hub_download

    def pinned_download(*args, **kwargs):
        kwargs["revision"] = revision
        path = original_download(*args, **kwargs)
        downloaded[Path(path).name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return path

    # All-In-One does not expose revision in analyze(); confine the override to
    # this invocation and restore it even when inference raises.
    loaders.hf_hub_download = pinned_download
    try:
        result = allin1.analyze(ctx["audio_path"], model=config.get("model", "harmonix-all"), device=config.get("device", "cpu"), demix_dir=work / "demix", spec_dir=work / "spec", keep_byproducts=True, multiprocess=False)
    finally:
        loaders.hf_hub_download = original_download
    duration = float(ctx["duration"])
    segments = []
    for segment in result.segments:
        start, end = max(0.0, float(segment.start)), min(duration, float(segment.end))
        if end <= start:
            continue
        segments.append({"id": f"section_{len(segments) + 1:03d}", "start": start, "end": end, "label": str(segment.label), "source": "allin1", "status": "estimated"})
    if not segments:
        raise RuntimeError("All-In-One returned no positive-duration segments")
    # Preserve unknown gaps and remove overlaps instead of stretching semantic labels.
    covered, cursor, repaired = [], 0.0, False
    for segment in sorted(segments, key=lambda row: (row["start"], row["end"])):
        if segment["start"] > cursor:
            covered.append({"start": cursor, "end": segment["start"], "label": "unknown", "source": "coverage_fill", "status": "unknown"})
            repaired = True
        if segment["end"] <= cursor:
            repaired = True
            continue
        if segment["start"] < cursor:
            segment["start"] = cursor
            repaired = True
        covered.append(segment)
        cursor = segment["end"]
    if cursor < duration:
        covered.append({"start": cursor, "end": duration, "label": "unknown", "source": "coverage_fill", "status": "unknown"})
        repaired = True
    for index, segment in enumerate(covered):
        segment["id"] = f"section_{index + 1:03d}"
    return {"status": "partial" if repaired else "succeeded", "data": {"segments": covered}, "warnings": ["Structure output had timing gaps/overlaps; unknown coverage and clipping were applied."] if repaired else [], "provenance": {"engine": "allin1", "model_id": config.get("model", "harmonix-all"), "version": importlib.metadata.version("allin1"), "model_revision": revision, "model_sha256": downloaded}}


def run_sections(ctx):
    config = ctx.get("config", {})
    backend = config.get("backend", "baseline")
    if backend in {"baseline", "novelty"}:
        return _baseline(ctx)
    if backend not in {"auto", "allin1"}:
        raise ValueError(f"Unsupported sections backend: {backend}")
    try:
        return _allin1(ctx)
    except Exception as exc:
        if not config.get("fallback", True):
            raise
        result = _baseline(ctx)
        result["warnings"].insert(0, f"All-In-One unavailable/failed; using unnamed acoustic segments: {type(exc).__name__}: {exc}")
        result["provenance"]["requested_backend"] = backend
        return result
