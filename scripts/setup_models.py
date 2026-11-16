#!/usr/bin/env python3
"""Prepare pinned model repositories/assets locally; never upload song audio."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(entry, target):
    if target.is_file() and sha256(target) == entry["sha256"]:
        return "cached"
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(target.suffix + ".part")
    try:
        for attempt in range(3):
            try:
                request = urllib.request.Request(entry["url"], headers={"User-Agent": "musicprep-model-setup/0.2"})
                with urllib.request.urlopen(request, timeout=60) as response, staging.open("wb") as output:
                    shutil.copyfileobj(response, output, 1024 * 1024)
                if sha256(staging) != entry["sha256"]:
                    raise ValueError("SHA-256 mismatch for " + entry["id"])
                staging.replace(target)
                return "downloaded"
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(1)
    finally:
        staging.unlink(missing_ok=True)


def git(args, cwd=None):
    return subprocess.run(["git", "-c", "http.version=HTTP/1.1", *args], cwd=cwd,
                          check=True, capture_output=True, text=True, timeout=300).stdout.strip()


def repository(entry, target):
    if target.exists():
        revision = git(["rev-parse", "HEAD"], target)
        if revision != entry["revision"] or git(["status", "--porcelain", "--untracked-files=no"], target):
            raise ValueError(f"Existing {target} differs from pinned source; keep it and choose another --models-dir")
        return "cached"
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="musicprep-source-", dir=target.parent) as folder:
        staging = Path(folder) / "repo"
        git(["init", str(staging)])
        git(["remote", "add", "origin", entry["url"]], staging)
        git(["fetch", "--depth", "1", "origin", entry["revision"]], staging)
        git(["checkout", "--detach", "FETCH_HEAD"], staging)
        if git(["rev-parse", "HEAD"], staging) != entry["revision"]:
            raise ValueError("Unexpected source revision")
        staging.rename(target)
    return "downloaded"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for group in ("core", "emotion", "tags"):
        parser.add_argument("--" + group, action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    groups = {g for g in ("core", "emotion", "tags") if args.all or getattr(args, g)}
    if not groups:
        groups = {"core"}
    manifest = json.loads((ROOT / "config/model-sources.json").read_text())
    try:
        for kind, handler in (("repositories", repository), ("files", download)):
            for entry in manifest[kind]:
                if entry["group"] in groups:
                    result = handler(entry, args.models_dir / entry["path"])
                    print(f"[{result}] {entry['id']}", flush=True)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Model setup failed: {exc}\n")
    print("Model assets ready. Demucs/MERT/All-In-One additionally use their upstream caches on first inference.")


if __name__ == "__main__":
    main()
