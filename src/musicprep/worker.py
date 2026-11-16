"""A task process. Keep heavy framework imports isolated from the orchestrator."""
import argparse
import importlib
import json
import os
from pathlib import Path
import traceback

REGISTRY = {
    "separation": ("musicprep.adapters.separation", "run_separation"),
    "rhythm": ("musicprep.adapters.events", "run_rhythm"),
    "sections": ("musicprep.adapters.structure", "run_sections"),
    "melody": ("musicprep.adapters.events", "run_melody"),
    "harmony": ("musicprep.adapters.events", "run_harmony"),
    "descriptions": ("musicprep.adapters.descriptions", "run_descriptions"),
    "lyrics_context": ("musicprep.adapters.lyrics", "run_lyrics_context"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=REGISTRY)
    parser.add_argument("context", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    output = args.result.resolve()
    context = json.loads(args.context.read_text(encoding="utf-8"))
    try:
        module, function = REGISTRY[args.task]
        fn = getattr(importlib.import_module(module), function)
        Path(context["work_dir"]).mkdir(parents=True, exist_ok=True)
        os.chdir(context["work_dir"])
        result = fn(context)
        if not isinstance(result, dict) or result.get("status") not in ("succeeded", "partial", "skipped"):
            raise ValueError("Adapter returned an invalid task result")
        if not isinstance(result.get("data"), dict):
            raise ValueError("Adapter data must be an object")
        if not isinstance(result.get("warnings", []), list):
            raise ValueError("Adapter warnings must be a list")
        json.dumps(result, allow_nan=False)
    except Exception as exc:
        traceback.print_exc()
        result = {"status": "failed", "data": {}, "warnings": [f"{type(exc).__name__}: {exc}"], "provenance": {}}
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
