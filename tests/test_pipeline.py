import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from musicprep.config import load_config, TASKS
from musicprep.pipeline import (atomic_json, assemble_package, configured_file_hashes, object_hash,
                                output_lock, run_process, execute_task, configured_python_environments,
                                probe_python_environment, cache_artifacts_valid)
from musicprep.audio import sha256_file


class PipelineTests(unittest.TestCase):
    def test_config_rejects_old_design_only_shape(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text('{"status":"design_only"}')
            with self.assertRaises(ValueError):
                load_config(path)

    def test_config_rejects_invalid_timeout_and_boolean_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            for execution in ({"timeout_seconds": 0}, {"timeout_seconds": float("inf")}, {"retries": True}):
                path.write_text(json.dumps({"execution": execution}))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_baseline_config_copies_defaults(self):
        baseline = load_config(profile="baseline")
        self.assertTrue(all(p["backend"] == "baseline" for p in baseline["tasks"].values()))
        self.assertEqual(load_config()["tasks"]["melody"]["backend"], "auto")

    def test_config_file_content_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint"
            path.write_text("version1")
            first = configured_file_hashes({"checkpoint": str(path)})
            path.write_text("version2")
            self.assertNotEqual(first, configured_file_hashes({"checkpoint": str(path)}))

    def test_atomic_json_rejects_nan_without_destroying_previous_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "result.json"
            atomic_json(path, {"value": 1})
            with self.assertRaises(ValueError):
                atomic_json(path, {"value": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"value": 1})

    def test_lock_blocks_concurrent_run(self):
        with tempfile.TemporaryDirectory() as folder:
            with output_lock(folder):
                with self.assertRaises(RuntimeError):
                    with output_lock(folder):
                        pass
            with output_lock(folder):
                pass

    def test_timeout_kills_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            started = time.monotonic()
            _, timed_out = run_process([sys.executable, "-c", "import time; time.sleep(30)"], Path(folder) / "log", .1, os.environ.copy())
            self.assertTrue(timed_out)
            self.assertLess(time.monotonic() - started, 8)

    def test_partial_package_does_not_claim_complete(self):
        manifest = {"source": {"sha256": "a" * 64}, "canonical": {"duration_seconds": 10, "sample_rate": 44100, "channels": 2}}
        results = {task: {"status": "succeeded", "data": {}, "warnings": [], "provenance": {"engine": "fixture"}} for task in TASKS}
        records = {task: {"status": "succeeded"} for task in TASKS}
        records["descriptions"]["status"] = "partial"
        package = assemble_package("fixture", manifest, results, records, "run")
        self.assertEqual(package["run"]["status"], "partial")

    def test_cache_reuses_success_but_invalidates_parameter_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            context = {"config": {"backend": "baseline"}, "dependencies": {}, "input_hashes": {}}
            calls = []
            def fake_process(command, *args):
                calls.append(command)
                Path(command[-1]).write_text(json.dumps({"status": "succeeded", "data": {"beats": []}, "warnings": [], "provenance": {}}))
                return 0, False
            with patch("musicprep.pipeline.run_process", fake_process):
                first, record1 = execute_task("rhythm", context, folder, {"retries": 0, "timeout_seconds": 10}, {"source": "fixture"})
                second, record2 = execute_task("rhythm", context, folder, {"retries": 0, "timeout_seconds": 10}, {"source": "fixture"})
                changed = copy.deepcopy(context)
                changed["config"]["tempo_hint"] = 100
                execute_task("rhythm", changed, folder, {"retries": 0, "timeout_seconds": 10}, {"source": "fixture"})
            self.assertEqual(first, second)
            self.assertFalse(record1["cached"])
            self.assertTrue(record2["cached"])
            self.assertEqual(len(calls), 2)

    def test_failed_task_retries_on_next_run(self):
        with tempfile.TemporaryDirectory() as folder:
            context = {"config": {}, "dependencies": {}, "input_hashes": {}}
            calls = []
            def fake_process(command, *args):
                calls.append(command)
                return 1, False
            with patch("musicprep.pipeline.run_process", fake_process):
                for _ in range(2):
                    result, record = execute_task("rhythm", context, folder, {"retries": 0, "timeout_seconds": 10}, {"source": "fixture"})
                    self.assertEqual(result["status"], "failed")
                    self.assertFalse(record["cached"])
            self.assertEqual(len(calls), 2)

    def test_config_paths_resolve_relative_to_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            config = root / "config" / "models.json"
            config.parent.mkdir()
            config.write_text(json.dumps({"tasks": {"descriptions": {
                "python": "../env/bin/python", "model_backends": {
                    "music2emo": {"python": "../emotion/bin/python", "repo": "../models/music2emo"}}},
                "harmony": {"btc_repo": "../models/btc", "checkpoint": "../weights/model.pt"}}}))
            loaded = load_config(config)
            desc = loaded["tasks"]["descriptions"]
            self.assertEqual(desc["python"], str(root / "env/bin/python"))
            self.assertEqual(desc["model_backends"]["music2emo"]["repo"], str(root / "models/music2emo"))
            self.assertEqual(desc["model_backends"]["music2emo"]["python"], str(root / "emotion/bin/python"))
            self.assertEqual(loaded["tasks"]["harmony"]["checkpoint"], str(root / "weights/model.pt"))

    def test_config_preserves_real_python_symlink_for_virtualenv(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            executable = root / "env" / "bin" / "python"
            executable.parent.mkdir(parents=True)
            executable.symlink_to(Path(sys.executable).resolve())
            config = root / "config.json"
            config.write_text(json.dumps({"tasks": {"descriptions": {
                "python": "env/bin/python", "model_backends": {
                    "music2emo": {"python": "env/bin/python"}}},
                "harmony": {"python_executable": "env/bin/python"}}}))
            loaded = load_config(config)["tasks"]
            self.assertEqual(loaded["descriptions"]["python"], str(executable))
            self.assertEqual(loaded["descriptions"]["model_backends"]["music2emo"]["python"], str(executable))
            self.assertEqual(loaded["harmony"]["python_executable"], str(executable))
            self.assertNotEqual(str(executable), str(executable.resolve()))

    def test_repo_code_weights_invalidate_but_audio_git_and_pycache_do_not(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / ".git").mkdir(); (root / "__pycache__").mkdir()
            (root / "model.py").write_text("version=1")
            (root / "weights.pt").write_bytes(b"weight1")
            first = configured_file_hashes({"repo": str(root)})
            (root / "sample.wav").write_bytes(b"audio")
            (root / ".git" / "config").write_text("metadata")
            (root / "__pycache__" / "extra.py").write_text("ignore")
            self.assertEqual(first, configured_file_hashes({"repo": str(root)}))
            (root / "model.py").write_text("version=2")
            code_change = configured_file_hashes({"repo": str(root)})
            self.assertNotEqual(first, code_change)
            (root / "weights.pt").write_bytes(b"weight2")
            self.assertNotEqual(code_change, configured_file_hashes({"repo": str(root)}))

    def test_actual_interpreter_metadata_is_available_without_model_imports(self):
        result = probe_python_environment(sys.executable)
        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["package_count"], 0)
        self.assertEqual(len(result["package_versions_sha256"]), 64)

    def test_nested_interpreters_and_missing_python_are_fingerprinted(self):
        with tempfile.TemporaryDirectory() as folder:
            missing = str(Path(folder) / "missing-python")
            result = configured_python_environments({"python": sys.executable,
                "model_backends": {"emotion": {"python": missing}}})
            self.assertEqual(result["worker"]["status"], "ok")
            self.assertEqual(result[".model_backends.emotion.python"]["status"], "failed")
            self.assertEqual(result[".model_backends.emotion.python"]["error"], "FileNotFoundError")

    def test_missing_python_returns_failed_task_and_can_continue(self):
        with tempfile.TemporaryDirectory() as folder:
            context = {"config": {"python": str(Path(folder) / "missing-python")}, "dependencies": {}}
            result, record = execute_task("rhythm", context, folder,
                {"retries": 1, "timeout_seconds": 10}, {"source": "fixture"})
            self.assertEqual(result["status"], "failed")
            self.assertEqual(record["attempts"], 2)
            self.assertTrue(any("Could not launch worker" in warning for warning in result["warnings"]))
            self.assertEqual(result["provenance"]["python_environments"]["worker"]["status"], "failed")

    def test_environment_change_invalidates_cached_partial(self):
        with tempfile.TemporaryDirectory() as folder:
            context = {"config": {"backend": "auto"}, "dependencies": {}}
            calls = []
            def fake_process(command, *args):
                calls.append(command)
                Path(command[-1]).write_text(json.dumps({"status": "partial", "data": {}, "warnings": [], "provenance": {}}))
                return 0, False
            with patch("musicprep.pipeline.run_process", fake_process), patch("musicprep.pipeline.configured_python_environments") as probe:
                probe.return_value = {"worker": {"status": "ok", "package_versions_sha256": "before-install"}}
                _, first = execute_task("rhythm", context, folder, {"retries": 0, "timeout_seconds": 10}, {})
                _, second = execute_task("rhythm", context, folder, {"retries": 0, "timeout_seconds": 10}, {})
                probe.return_value = {"worker": {"status": "ok", "package_versions_sha256": "after-install"}}
                _, third = execute_task("rhythm", context, folder, {"retries": 0, "timeout_seconds": 10}, {})
            self.assertTrue(second["cached"])
            self.assertFalse(third["cached"])
            self.assertNotEqual(first["cache_key"], third["cache_key"])
            self.assertEqual(len(calls), 2)

    def test_separation_cache_requires_artifact_manifest_and_valid_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder).resolve() / "vocals.wav"; path.write_bytes(b"original")
            result = {"data": {"stems": {"vocals": str(path)}}, "artifacts": []}
            self.assertFalse(cache_artifacts_valid(result))
            result["artifacts"] = [{"path": str(path), "sha256": sha256_file(path)}]
            self.assertTrue(cache_artifacts_valid(result))
            path.write_bytes(b"corrupt")
            self.assertFalse(cache_artifacts_valid(result))
            for malformed in (None, [], {"artifacts": [None]}, {"data": []}):
                self.assertFalse(cache_artifacts_valid(malformed))


if __name__ == "__main__":
    unittest.main()
