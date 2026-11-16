import copy
import json
import math
import os
from pathlib import Path

TASKS = ("separation", "rhythm", "sections", "melody", "harmony", "descriptions", "lyrics_context")
DEFAULT = {
    "schema_version": "0.1.0",
    "execution": {"timeout_seconds": 1800, "retries": 0},
    "tasks": {task: {"backend": "auto", "fallback": True} for task in TASKS},
}

PATH_KEYS = {"repo", "btc_repo", "weights", "checkpoint", "dictionary", "embedding", "graph", "metadata", "basic_pitch_model", "source_metadata"}


def resolve_paths(value, base):
    """Configuration file paths are relative to that file, never a worker's cwd."""
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if isinstance(item, dict):
            result[key] = resolve_paths(item, base)
        elif isinstance(item, str) and (key in PATH_KEYS
                or (key == "model_id" and (item.startswith((".", "/", "~")) or (base / item).is_dir()))
                or (key in {"python", "python_executable"} and ("/" in item or item.startswith("~")))):
            path = Path(item).expanduser()
            target = base / path if not path.is_absolute() else path
            # Resolving a venv's python symlink would bypass that environment.
            result[key] = os.path.abspath(target) if key in {"python", "python_executable"} else str(target.resolve())
        else:
            result[key] = item
    return result


def load_config(path=None, profile="auto"):
    if profile not in ("auto", "baseline"):
        raise ValueError("profile must be auto or baseline")
    config = copy.deepcopy(DEFAULT)
    if profile == "baseline":
        for params in config["tasks"].values():
            params["backend"] = "baseline"
    if path:
        custom = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(custom, dict) or set(custom) - set(DEFAULT):
            raise ValueError("Configuration keys must be schema_version, execution and tasks; the old design-only example is not a runnable config.")
        custom = resolve_paths(custom, Path(path).resolve().parent)
        for key in ("execution", "tasks"):
            if key in custom and not isinstance(custom[key], dict):
                raise ValueError(f"{key} must be an object")
        unknown = set(custom.get("tasks", {})) - set(TASKS)
        if unknown:
            raise ValueError(f"Unknown tasks: {sorted(unknown)}")
        for task, params in custom.get("tasks", {}).items():
            if not isinstance(params, dict):
                raise ValueError(f"Task {task} configuration must be an object")
            config["tasks"][task].update(params)
        config["execution"].update(custom.get("execution", {}))
        if custom.get("schema_version", "0.1.0") != "0.1.0":
            raise ValueError("Unsupported configuration version")
    timeout = config["execution"]["timeout_seconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    retries = config["execution"]["retries"]
    if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= 3:
        raise ValueError("retries must be an integer from 0 to 3")
    for task, params in config["tasks"].items():
        if not isinstance(params.get("backend"), str):
            raise ValueError(f"{task}.backend must be a string")
        if not isinstance(params.get("fallback"), bool):
            raise ValueError(f"{task}.fallback must be boolean")
        if "timeout_seconds" in params:
            value = params["timeout_seconds"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{task}.timeout_seconds must be positive and finite")
    return config
