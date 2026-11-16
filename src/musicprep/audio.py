#!/usr/bin/env python3
"""Prepare audio without cutting, normalizing, resampling or running MIR models."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


class PreparationError(RuntimeError):
    pass


def run_command(arguments):
    try:
        return subprocess.run(arguments, check=True, capture_output=True, text=True).stdout
    except FileNotFoundError as exc:
        raise PreparationError(f"Required program not found: {arguments[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise PreparationError(f"{arguments[0]} failed: {exc.stderr.strip()}") from exc


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_audio(path):
    raw = json.loads(run_command([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)
    ]))
    streams = [stream for stream in raw.get("streams", []) if stream.get("codec_type") == "audio"]
    if len(streams) != 1:
        raise PreparationError("Expected exactly one audio stream; explicit stream selection is required otherwise.")
    stream = streams[0]
    duration_text = stream.get("duration") or raw.get("format", {}).get("duration")
    try:
        duration = float(duration_text)
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PreparationError("Missing or invalid duration, sample rate or channels.") from exc
    if not math.isfinite(duration) or duration <= 0 or sample_rate <= 0 or channels <= 0:
        raise PreparationError("Audio duration, sample rate and channels must be positive and finite.")
    return {
        "codec": stream.get("codec_name"),
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_format": stream.get("sample_fmt"),
        "bits_per_raw_sample": int(stream.get("bits_per_raw_sample") or 0) or None,
        "duration_seconds": duration,
        "size_bytes": path.stat().st_size,
    }


def prepare_audio(source, output_dir, track_id):
    source = Path(source).resolve()
    output_dir = Path(output_dir).absolute()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", track_id):
        raise PreparationError("track_id must be 1–64 ASCII letters, digits, underscores or hyphens.")
    if not source.is_file():
        raise PreparationError(f"Input is not a file: {source}")
    if output_dir.exists():
        raise PreparationError(f"Output directory already exists; choose a new directory: {output_dir}")
    source_metadata = probe_audio(source)
    source_hash = sha256_file(source)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{track_id}-", dir=output_dir.parent))
    try:
        canonical_path = staging / "canonical.wav"
        run_command([
            "ffmpeg", "-nostdin", "-v", "error", "-n", "-i", str(source),
            "-map", "0:a:0", "-vn", "-map_metadata", "-1",
            "-c:a", "pcm_f32le", "-rf64", "auto", str(canonical_path)
        ])
        canonical_metadata = probe_audio(canonical_path)
        if canonical_metadata["sample_rate"] != source_metadata["sample_rate"]:
            raise PreparationError("Unexpected sample-rate change.")
        if canonical_metadata["channels"] != source_metadata["channels"]:
            raise PreparationError("Unexpected channel-count change.")
        if canonical_metadata["codec"] != "pcm_f32le":
            raise PreparationError("Canonical audio is not float32 PCM.")
        duration_delta = canonical_metadata["duration_seconds"] - source_metadata["duration_seconds"]
        # Containers can round reported durations. Do not silently accept an audio edit.
        tolerance = 1.0 / source_metadata["sample_rate"] + 0.000002
        if abs(duration_delta) > tolerance:
            raise PreparationError(f"Duration changed by {duration_delta:.9f}s; inspect decoder/container timing.")
        manifest = {
            "schema_version": "0.1.0",
            "track_id": track_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {"filename": source.name, "sha256": source_hash, **source_metadata},
            "canonical": {"file": "canonical.wav", "sha256": sha256_file(canonical_path), **canonical_metadata},
            "timeline": {"origin": "source_start", "offset_seconds": 0, "unit": "seconds"},
            "processing": {
                "operation": "decode_to_float32_pcm",
                "resampled": False,
                "normalized": False,
                "trimmed": False,
                "duration_delta_seconds": duration_delta,
                "ffmpeg_version": run_command(["ffmpeg", "-version"]).splitlines()[0],
                "ffprobe_version": run_command(["ffprobe", "-version"]).splitlines()[0],
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        # Do not replace any prior run or expose a successful manifest before decoding finishes.
        if output_dir.exists():
            raise PreparationError("Output directory appeared during preparation; refusing to replace it.")
        staging.rename(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = prepare_audio(args.source, args.output_dir, args.track_id)
    except (PreparationError, OSError, ValueError) as exc:
        parser.exit(1, f"Audio preparation failed: {exc}\n")
    print(json.dumps({
        "status": "prepared",
        "track_id": manifest["track_id"],
        "duration_seconds": manifest["canonical"]["duration_seconds"],
        "sample_rate": manifest["canonical"]["sample_rate"],
        "channels": manifest["canonical"]["channels"],
        "manifest": str(args.output_dir / "manifest.json"),
        "music_analysis": "not_run",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
