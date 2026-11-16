#!/usr/bin/env python3
"""Install an isolated SOFA environment and pinned official Mandarin assets."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
REVISION = "9549c6a86d16019c817eefe4bb8183405da524cc"
URL = "https://github.com/qiuqiao/SOFA/releases/download/v1.0.1/pretrained_mandarin_singing.zip"
ARCHIVE_SHA = "63a8c6191fe6bdf29ac40af2955ef96816b8391596d94a4340e0f58dcf47f26e"
MODEL_SHA = "612cb69ef3503befc43286f0d90893b4b3ad959df07ed08799dba2bea69d2b51"
REQUIREMENTS = ["torch==2.5.1", "torchaudio==2.5.1", "numpy==1.24.4", "scipy==1.15.3",
                "librosa==0.9.2", "lightning==2.4.0", "pytorch-lightning==2.4.0", "pypinyin==0.55.0",
                "setuptools<81", "soundfile", "click", "einops==0.6.1", "h5py", "matplotlib~=3.7.3",
                "PyYAML~=6.0.1", "tensorboard", "tensorboardX", "tqdm~=4.66.1", "textgrid", "chardet",
                "numba<0.62", "pandas~=2.0.3"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args, cwd=None):
    subprocess.run([str(arg) for arg in args], cwd=cwd, check=True)


def prepare_repo(repo):
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "init", repo])
        run(["git", "-C", repo, "remote", "add", "origin", "https://github.com/qiuqiao/SOFA.git"])
        run(["git", "-c", "http.version=HTTP/1.1", "-C", repo, "fetch", "--depth", "1", "origin", REVISION])
        run(["git", "-C", repo, "checkout", "--detach", "FETCH_HEAD"])
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True)
    if actual != REVISION or dirty:
        raise ValueError("Existing SOFA source differs from pinned revision; refusing to overwrite it")


def prepare_model(directory):
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "v1.0.0_mandarin_singing.ckpt"
    if checkpoint.is_file() and sha256(checkpoint) == MODEL_SHA:
        return checkpoint
    archive = directory / "pretrained_mandarin_singing.zip"
    if not archive.is_file() or sha256(archive) != ARCHIVE_SHA:
        staging = archive.with_suffix(".zip.part")
        request = urllib.request.Request(URL, headers={"User-Agent": "musicprep-sofa-setup/0.2"})
        with urllib.request.urlopen(request, timeout=120) as response, staging.open("wb") as stream:
            shutil.copyfileobj(response, stream, 1024 * 1024)
        if sha256(staging) != ARCHIVE_SHA:
            staging.unlink(missing_ok=True)
            raise ValueError("SOFA archive SHA-256 mismatch")
        staging.replace(archive)
    # Extract only named files; never trust archive paths as filesystem paths.
    with zipfile.ZipFile(archive) as package:
        for filename in (checkpoint.name, "hparams.yaml", "data_providers.md", "readme.md"):
            target = directory / filename
            with package.open("pretrained_mandarin_singing/" + filename) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
    if sha256(checkpoint) != MODEL_SHA:
        raise ValueError("SOFA checkpoint SHA-256 mismatch")
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-install", action="store_true", help="only verify/download source and weights")
    parser.add_argument("--verify-only", action="store_true", help="verify existing assets without installing or downloading")
    args = parser.parse_args()
    repo = ROOT / "models/vendor/SOFA"
    checkpoint = ROOT / "models/sofa/v1.0.0_mandarin_singing.ckpt"
    if args.verify_only:
        if not repo.is_dir() or not checkpoint.is_file() or sha256(checkpoint) != MODEL_SHA:
            raise ValueError("SOFA model assets are missing or invalid")
        prepare_repo(repo)
    else:
        prepare_repo(repo)
        checkpoint = prepare_model(checkpoint.parent)
    if not (args.skip_install or args.verify_only):
        uv = shutil.which("uv")
        if not uv:
            raise RuntimeError("uv must be installed to create the isolated SOFA environment")
        environment = ROOT / ".venv-sofa"
        if not environment.exists():
            run([uv, "venv", "--python", "3.11", environment])
        run([uv, "pip", "install", "--python", environment / "bin/python", *REQUIREMENTS])
    print("SOFA source and Mandarin checkpoint verified:", checkpoint)
    print("Use .venv-sofa/bin/python with backend=sofa; see docs/sofa-implementation.md")


if __name__ == "__main__":
    main()
