"""Deterministic task scheduling, isolated execution, cache and package assembly."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

from . import __version__
from .audio import prepare_audio, sha256_file
from .config import TASKS, load_config

DEPENDENCIES = {
    "separation": (), "rhythm": (), "sections": (),
    "melody": ("separation",), "harmony": ("separation", "rhythm"),
    "descriptions": ("sections", "rhythm", "separation"),
    "lyrics_context": ("sections", "separation"),
}


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def object_hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def implementation_hash():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment_versions():
    result = {"python": sys.version.split()[0], "musicprep": __version__}
    for name in ("numpy", "scipy", "soundfile", "librosa", "torch", "torchaudio", "demucs", "beat-this", "basic-pitch", "torchcrepe", "allin1", "essentia", "transformers", "music2emo"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


MODEL_FILE_SUFFIXES = {
    ".py", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".txt",
    ".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".pb", ".tflite",
    ".h5", ".hdf5", ".npy", ".npz", ".bin", ".pkl", ".pickle",
}
MODEL_DIRECTORY_KEYS = {"repo", "btc_repo", "weights", "checkpoint", "model_dir"}
SKIP_MODEL_DIRS = {".git", "__pycache__", "venv", "env", "node_modules", "audio", "logs", "outputs", ".pytest_cache"}


def model_directory_hash(directory):
    """Hash code/config/weights by content, excluding audio and generated caches."""
    digest, count, errors = hashlib.sha256(), 0, []
    def record_error(exc):
        errors.append({"path": str(exc.filename), "error": type(exc).__name__})
    for root, dirs, files in os.walk(directory, followlinks=False, onerror=record_error):
        dirs[:] = sorted(name for name in dirs if name not in SKIP_MODEL_DIRS
                         and not name.startswith(".venv") and not Path(root, name).is_symlink())
        for name in sorted(files):
            path = Path(root, name)
            if path.suffix.lower() not in MODEL_FILE_SUFFIXES:
                continue
            try:
                checksum = sha256_file(path)
            except OSError as exc:
                record_error(exc)
                continue
            digest.update(path.relative_to(directory).as_posix().encode())
            digest.update(b"\0")
            digest.update(checksum.encode())
            digest.update(b"\0")
            count += 1
    return {"sha256": digest.hexdigest(), "file_count": count, "errors": errors}


def configured_file_hashes(value):
    """Include explicit files and model-repository contents in task cache keys."""
    hashes, seen = {}, {}
    def visit(obj, prefix="", parameter=""):
        if isinstance(obj, dict):
            for key, item in obj.items():
                visit(item, prefix + "." + key, key)
        elif isinstance(obj, list):
            for index, item in enumerate(obj):
                visit(item, prefix + f"[{index}]", parameter)
        elif isinstance(obj, str):
            try:
                path = Path(obj).expanduser()
                if path.is_file():
                    identity = (str(path.resolve()), "file")
                    if identity not in seen:
                        seen[identity] = sha256_file(path)
                    hashes[prefix] = seen[identity]
                elif parameter in MODEL_DIRECTORY_KEYS:
                    identity = (str(path.resolve()), "directory")
                    if identity not in seen:
                        seen[identity] = model_directory_hash(path) if path.is_dir() else {"status": "missing"}
                    hashes[prefix] = seen[identity]
            except OSError as exc:
                hashes[prefix] = {"error": type(exc).__name__, "errno": exc.errno}
    visit(value)
    return hashes


_ENVIRONMENT_PROBE = """
import importlib.metadata, json, sys
packages = sorted((d.metadata.get('Name', '').lower().replace('_', '-'), d.version)
                  for d in importlib.metadata.distributions())
print(json.dumps({'python': sys.version, 'executable': sys.executable, 'packages': packages}, sort_keys=True))
"""


def probe_python_environment(executable):
    """Read package metadata in the actual worker interpreter without model imports."""
    try:
        completed = subprocess.run([executable, "-c", _ENVIRONMENT_PROBE],
                                   capture_output=True, text=True, timeout=20, check=False)
        if completed.returncode:
            return {"status": "failed", "returncode": completed.returncode,
                    "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest()}
        data = json.loads(completed.stdout)
        if not isinstance(data, dict) or not isinstance(data.get("packages"), list):
            raise ValueError("invalid metadata response")
        return {"status": "ok", "python": data["python"], "executable": data["executable"],
                "package_versions_sha256": object_hash(data["packages"]), "package_count": len(data["packages"])}
    except subprocess.TimeoutExpired:
        return {"status": "failed", "error": "metadata_probe_timeout"}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"status": "failed", "error": type(exc).__name__, "errno": getattr(exc, "errno", None)}


def configured_python_environments(config):
    """Capture task worker and nested model interpreters, including missing ones."""
    executables = {"worker": config.get("python", sys.executable)}
    def visit(value, prefix=""):
        if isinstance(value, dict):
            for key, item in value.items():
                name = prefix + "." + key
                if key in ("python", "python_executable"):
                    executables[name] = item
                else:
                    visit(item, name)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, prefix + f"[{index}]")
    visit(config)
    result, seen = {}, {}
    for name, executable in executables.items():
        if not isinstance(executable, str) or not executable:
            result[name] = {"status": "failed", "error": "invalid_python_executable"}
            continue
        if executable not in seen:
            seen[executable] = probe_python_environment(executable)
        result[name] = seen[executable]
    return result


@contextmanager
def output_lock(directory):
    """OS lock: releases automatically on process crash (no stale lock recovery prompt)."""
    import fcntl
    path = Path(directory) / ".run.lock"
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another analysis is using this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_process(command, log_path, timeout, env):
    with Path(log_path).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            return process.wait(timeout=timeout), False
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            if isinstance(exc, KeyboardInterrupt):
                raise
            return process.returncode, True


def cache_artifacts_valid(result):
    """Reject incomplete manifests and changed/missing separation files as cache misses."""
    if not isinstance(result, dict) or not isinstance(result.get("data", {}), dict):
        return False
    artifacts = result.get("artifacts", [])
    if not isinstance(artifacts, list):
        return False
    verified = set()
    try:
        for item in artifacts:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                return False
            path = Path(item["path"])
            if not path.is_absolute() or not path.is_file() or sha256_file(path) != item.get("sha256"):
                return False
            verified.add(str(path.resolve()))
        stems = result.get("data", {}).get("stems", {})
        if not isinstance(stems, dict):
            return False
        for path in stems.values():
            if not isinstance(path, str) or not Path(path).is_absolute() or str(Path(path).resolve()) not in verified:
                return False
    except (OSError, ValueError):
        return False
    return True


def execute_task(task, context, cache_root, execution, fingerprint, force=False):
    python_environments = configured_python_environments(context["config"])
    key = object_hash({"task": task, "source": fingerprint, "config": context["config"],
                       "dependencies": context["dependencies"], "optional_inputs": context.get("input_hashes", {}),
                       "configured_files": configured_file_hashes(context["config"]),
                       "python_environments": python_environments})
    folder = Path(cache_root) / task / key
    folder.mkdir(parents=True, exist_ok=True)
    cached = folder / "cache.json"
    if cached.exists() and not force:
        try:
            result = json.loads(cached.read_text())
            if isinstance(result, dict) and result.get("status") in ("succeeded", "partial", "skipped") and cache_artifacts_valid(result):
                return result, {"cached": True, "cache_key": key, "duration_seconds": 0, "attempts": 0}
        except (ValueError, KeyError, OSError, TypeError):
            pass
    started = time.monotonic()
    result = None
    for attempt in range(execution["retries"] + 1):
        result = None
        attempt_dir = folder / f"attempt-{uuid.uuid4().hex}"
        attempt_dir.mkdir()
        ctx = dict(context, work_dir=str(attempt_dir.resolve()))
        context_file, result_file = attempt_dir / "context.json", attempt_dir / "result.json"
        atomic_json(context_file, ctx)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("OMP_NUM_THREADS", "2")
        env.setdefault("MKL_NUM_THREADS", "2")
        python = context["config"].get("python", sys.executable)
        try:
            code, timeout = run_process([python, "-m", "musicprep.worker", task, str(context_file.resolve()), str(result_file.resolve())],
                                        attempt_dir / "worker.log", execution["timeout_seconds"], env)
        except OSError as exc:
            code, timeout = -1, False
            result = {"status": "failed", "data": {}, "warnings": [f"Could not launch worker: {exc}"], "provenance": {}}
        if timeout:
            result = {"status": "failed", "data": {}, "warnings": [f"Task exceeded {execution['timeout_seconds']} seconds"], "provenance": {}}
        elif result_file.exists():
            try:
                result = json.loads(result_file.read_text())
            except ValueError:
                result = None
        if result is None or (code != 0 and result.get("status") != "failed"):
            result = {"status": "failed", "data": {}, "warnings": [f"Worker exited with code {code}; see task log"], "provenance": {}}
        if result.get("status") != "failed":
            break
        result_file.unlink(missing_ok=True)
    artifacts = []
    if task == "separation":
        for name, path in result.get("data", {}).get("stems", {}).items():
            if isinstance(path, str) and Path(path).is_file():
                normalized = str(Path(path).resolve())
                result["data"]["stems"][name] = normalized
                artifacts.append({"path": normalized, "sha256": sha256_file(Path(path))})
    result["artifacts"] = artifacts
    result.setdefault("provenance", {}).update({"pipeline_version": __version__, "cache_key": key,
                                               "python_environments": python_environments})
    atomic_json(cached, result)
    return result, {"cached": False, "cache_key": key, "duration_seconds": round(time.monotonic() - started, 3),
                    "attempts": attempt + 1, "log": str((attempt_dir / "worker.log").relative_to(cache_root))}


def assemble_package(track_id, manifest, results, records, run_id):
    def section(name, default):
        data = dict(default)
        data.update(results.get(name, {}).get("data", {}))
        data["provenance"] = results.get(name, {}).get("provenance", {})
        return data
    required = ("melody", "harmony", "rhythm", "sections", "descriptions")
    states = [records[name]["status"] for name in required]
    status = "complete" if all(s == "succeeded" for s in states) else "partial"
    if all(s == "failed" for s in states):
        status = "failed"
    warning_list = [f"{task}: {message}" for task, result in results.items() for message in result.get("warnings", [])]
    return {
        "schema_version": "0.1.0", "track_id": track_id,
        "audio": {"source_sha256": manifest["source"]["sha256"], "duration_seconds": manifest["canonical"]["duration_seconds"],
                  "sample_rate": manifest["canonical"]["sample_rate"], "channels": manifest["canonical"]["channels"], "time_origin": "source_start"},
        "run": {"id": run_id, "status": status, "tasks": records},
        "melody": section("melody", {"notes": [], "contour": []}),
        "harmony": section("harmony", {"segments": []}),
        "rhythm": section("rhythm", {"beats": [], "tempo": [], "meter": []}),
        "sections": results.get("sections", {}).get("data", {}).get("segments", []),
        "descriptions": results.get("descriptions", {}).get("data", {}).get("segments", []),
        "lyrics_context": section("lyrics_context", {"lyrics": [], "themes": [], "narrative": [], "imagery": [], "background": [], "sources": []}),
        "warnings": warning_list,
    }


def analyze(source, output, track_id=None, profile="auto", config_path=None, lyrics=None, background=None, force=False, progress=None):
    from .validation import validate_package
    from .report import write_report
    source = Path(source).resolve()
    if not source.is_file():
        raise ValueError(f"Audio file not found: {source}")
    config = load_config(config_path, profile)
    source_hash = sha256_file(source)
    track_id = track_id or (re.sub(r"[^A-Za-z0-9_-]", "-", source.stem).strip("-")[:40] or "track") + "-" + source_hash[:8]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", track_id):
        raise ValueError("Invalid track ID")
    optional = {"lyrics": None, "background": None}
    input_hashes = {}
    for name, value in (("lyrics", lyrics), ("background", background)):
        if value:
            path = Path(value).resolve()
            if not path.is_file():
                raise ValueError(f"Optional input does not exist: {name}")
            optional[name] = str(path)
            input_hashes[name] = sha256_file(path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        audio_dir = output / "audio" / source_hash[:16]
        manifest_file = audio_dir / "manifest.json"
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text())
            if manifest["source"]["sha256"] != source_hash or sha256_file(audio_dir / "canonical.wav") != manifest["canonical"]["sha256"]:
                raise ValueError("Cached canonical audio is stale/corrupt; use a fresh output directory")
        else:
            manifest = prepare_audio(source, audio_dir, track_id)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        versions = environment_versions()
        fingerprint = {"source_sha256": source_hash, "canonical_sha256": manifest["canonical"]["sha256"],
                       "implementation": implementation_hash(), "versions": versions}
        records, results = {}, {}
        for task in TASKS:
            if progress:
                progress(task, "running")
            context = {"audio_path": str(audio_dir / "canonical.wav"), "source_audio_path": str(source),
                       "duration": manifest["canonical"]["duration_seconds"], "sample_rate": manifest["canonical"]["sample_rate"],
                       "channels": manifest["canonical"]["channels"], "config": config["tasks"][task],
                       "dependencies": {name: results.get(name, {}).get("data", {}) for name in DEPENDENCIES[task]},
                       "optional_inputs": optional, "input_hashes": input_hashes}
            result, record = execute_task(task, context, output / "cache", config["execution"], fingerprint, force)
            results[task] = result
            records[task] = dict(record, status=result["status"], provenance=result.get("provenance", {}))
            atomic_json(output / "run-state.json", {"id": run_id, "tasks": records, "versions": versions})
            if progress:
                progress(task, result["status"] + (" (cached)" if record["cached"] else ""))
        package = assemble_package(track_id, manifest, results, records, run_id)
        errors = validate_package(package)
        if errors:
            atomic_json(output / "invalid-package.json", package)
            atomic_json(output / "validation-errors.json", errors)
            raise ValueError("Package validation failed: " + "; ".join(errors[:8]))
        atomic_json(output / "song-package.json", package)
        atomic_json(output / "run-summary.json", {"run_id": run_id, "status": package["run"]["status"], "tasks": records,
                                                  "versions": versions, "config": config, "source_sha256": source_hash})
        write_report(package, output / "report.html", audio_dir / "canonical.wav")
        return package
