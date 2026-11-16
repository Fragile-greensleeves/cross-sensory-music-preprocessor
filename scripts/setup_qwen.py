#!/usr/bin/env python3
"""Install isolated Apple-Silicon Qwen2-Audio runtime and pinned real weights."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

REPO = "mlx-community/Qwen2-Audio-7B-Instruct-4bit"
REVISION = "c65570002626f41b4dc08b7b54f42f99f3e82e7f"
SHA256 = "0967cde270ad62aa4824f0bbce283d0a2a6da2825ccf587640753c527d4174da"
SIZE = 6562540479
SOURCE_REVISION = "b5fcbe964b9e4cad4b06782f97d3e51fe74a8702"
REQUIRED_SMALL_FILES = ("config.json", "preprocessor_config.json", "tokenizer.json",
                        "tokenizer_config.json", "vocab.json", "merges.txt")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_small_files(destination):
    for name in REQUIRED_SMALL_FILES:
        path = destination / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing or empty Qwen model asset: {name}")
        if name.endswith(".json"):
            value = json.loads(path.read_text())
            if not isinstance(value, dict) or not value:
                raise ValueError(f"Invalid Qwen model asset: {name}")
    config = json.loads((destination / "config.json").read_text())
    preprocessor = json.loads((destination / "preprocessor_config.json").read_text())
    if config.get("model_type") != "qwen2_audio" or config.get("quantization", {}).get("bits") != 4:
        raise ValueError("Model configuration is not Qwen2-Audio MLX 4-bit")
    if preprocessor.get("sampling_rate") != 16000 or preprocessor.get("feature_size") != 128:
        raise ValueError("Audio preprocessor does not match Qwen2-Audio")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-install", action="store_true", help="Reuse existing .venv-qwen")
    parser.add_argument("--skip-download", action="store_true", help="Install environment only")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("This runtime requires Apple Silicon macOS; main baseline pipeline remains portable.")
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("Install uv first: https://docs.astral.sh/uv/getting-started/installation/")
    env_dir = root / ".venv-qwen"
    python = env_dir / "bin/python"
    if not args.skip_install:
        if not python.is_file():
            subprocess.run([uv, "venv", "--python", "3.11", str(env_dir)], check=True)
        packages = [
            f"mlx-audio @ git+https://github.com/Blaizzy/mlx-audio.git@{SOURCE_REVISION}",
            "mlx==0.32.4", "mlx-metal==0.32.4", "transformers==5.23.0",
            "numpy==2.4.6", "scipy==1.17.1", "soundfile==0.14.0", "huggingface-hub==2.4.0",
        ]
        subprocess.run([uv, "pip", "install", "--python", str(python), *packages], check=True)
    if not python.is_file():
        raise SystemExit("Missing .venv-qwen/bin/python; run without --skip-install first")
    if args.skip_download:
        return
    destination = root / "models/qwen2-audio-mlx-4bit"
    destination.mkdir(parents=True, exist_ok=True)
    weights = destination / "weights.safetensors"
    valid_existing_weights = weights.is_file() and weights.stat().st_size == SIZE and sha256(weights) == SHA256
    if weights.is_file() and not valid_existing_weights:
        raise SystemExit("Existing weight integrity check failed; do not run this model")
    if not valid_existing_weights:
        if shutil.disk_usage(root).free < SIZE + 1024**3:
            raise SystemExit("Need at least 7.6 GB available for the quantized weights; original BF16 model is not downloaded")
    # Always repair/check processor and tokenizer assets. Once the weights are
    # complete and verified, never include them in the download allow-list.
    patterns = ["*.json", "*.txt", "README.md"]
    if not valid_existing_weights:
        patterns.append("*.safetensors")
    env = os.environ.copy()
    env.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
    env.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    code = (
        "from huggingface_hub import snapshot_download; import sys,json; "
        "snapshot_download(sys.argv[1],revision=sys.argv[2],local_dir=sys.argv[3],"
        "allow_patterns=json.loads(sys.argv[4]),max_workers=2)"
    )
    subprocess.run([str(python), "-c", code, REPO, REVISION, str(destination), json.dumps(patterns)], check=True, env=env)
    validate_small_files(destination)
    if not valid_existing_weights and (weights.stat().st_size != SIZE or sha256(weights) != SHA256):
        raise SystemExit("Weight integrity check failed; do not run this model")
    manifest = {
        "model_id": REPO, "revision": REVISION, "base_model": "Qwen/Qwen2-Audio-7B-Instruct",
        "weights": {"file": weights.name, "bytes": SIZE, "sha256": SHA256},
        "quantization": {"bits": 4, "group_size": 64, "scope": "language_model_linear_layers"},
        "mlx_audio_revision": SOURCE_REVISION,
    }
    (destination / "musicprep-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "ready", "model": str(destination), "python": str(python), "sha256": SHA256}))


if __name__ == "__main__":
    main()
