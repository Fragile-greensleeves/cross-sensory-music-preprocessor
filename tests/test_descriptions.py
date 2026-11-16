import tempfile
import json
from pathlib import Path
import unittest
import subprocess
from unittest.mock import patch

import numpy as np
import soundfile as sf

from musicprep.adapters.descriptions import run_descriptions, _validate_model_result, _run_model_batch


class DescriptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        sr = 22050
        t = np.arange(sr * 2) / sr
        self.path = self.root / "tone.wav"
        sf.write(self.path, 0.1 * np.sin(2 * np.pi * 440 * t), sr, subtype="FLOAT")
        self.ctx = {"audio_path": str(self.path), "duration": 2.0, "work_dir": str(self.root / "out"),
                    "config": {"backend": "baseline"}, "dependencies": {"sections": {"segments": [
                        {"id": "a", "start": 0, "end": 1}, {"id": "b", "start": 1, "end": 2}]}}}

    def test_baseline_reports_facts_not_invented_emotions(self):
        result = run_descriptions(self.ctx)
        self.assertEqual(result["status"], "partial")
        for section in result["data"]["segments"]:
            self.assertEqual(section["genre"], [])
            self.assertEqual(section["instruments"], [])
            self.assertIsNone(section["emotion"]["valence"])
            self.assertAlmostEqual(section["acoustic_features"]["spectral_centroid_hz"], 440, delta=10)
            self.assertEqual(section["textures"][0]["status"], "candidate")

    def test_silence_does_not_become_dark_or_sad(self):
        sf.write(self.path, np.zeros(44100), 22050)
        segment = run_descriptions(self.ctx)["data"]["segments"][0]
        self.assertTrue(segment["acoustic_features"]["silent"])
        self.assertEqual(segment["textures"], [])
        self.assertEqual(segment["emotion"]["status"], "unknown")

    def test_qwen_long_sections_are_covered_without_truncation(self):
        self.ctx["duration"] = 61.0
        self.ctx["dependencies"]["sections"]["segments"] = [{"id": "long", "start": 0, "end": 61}]
        self.ctx["config"] = {"backend": "qwen2_audio", "model_backends": {"qwen2_audio": {}}}
        sf.write(self.path, np.zeros(22050 * 61), 22050)
        def batch(engine, cfg, entries, work):
            for entry in entries:
                info = sf.info(entry["audio_path"])
                self.assertLessEqual(info.duration, 30)
            return {"results": [{"index": x["index"], "data": {"genre": [], "instruments": [], "textures": []}} for x in entries]}, "log"
        with patch("musicprep.adapters.descriptions._run_model_batch", side_effect=batch):
            records = run_descriptions(self.ctx)["data"]["segments"]
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["analysis_start"], 0)
        self.assertEqual(records[-1]["analysis_end"], 61)
        for first, second in zip(records, records[1:]):
            self.assertEqual(first["analysis_end"], second["analysis_start"])
        self.assertEqual({r["section_id"] for r in records}, {"long"})

    def test_invalid_sections_do_not_create_out_of_bounds_clips(self):
        self.ctx["dependencies"]["sections"]["segments"] = [{"id": "../../escape", "start": -1, "end": 30}]
        result = run_descriptions(self.ctx)
        self.assertEqual(result["data"]["segments"][0]["applies_end"], 2)
        self.assertTrue(any("ignored" in message for message in result["warnings"]))
        self.assertFalse((self.root / "escape").exists())

    def test_section_identifier_is_not_used_as_a_file_path(self):
        self.ctx["dependencies"]["sections"]["segments"][0]["id"] = "../../escape"
        run_descriptions(self.ctx)
        self.assertTrue((self.root / "out/segment_0000.wav").is_file())
        self.assertFalse((self.root / "escape.wav").exists())

    def test_single_model_segment_failure_preserves_other_segments(self):
        self.ctx["config"] = {"backend": "music2emo", "model_backends": {"music2emo": {}}}
        output = {"results": [{"index": 0, "error": "test segment failure"},
                               {"index": 1, "data": {"emotion": {"valence": 5, "arousal": 4,
                                                                 "scale": [1, 9], "moods": ["calm"]}}}]}
        with patch("musicprep.adapters.descriptions._run_model_batch", return_value=(output, "/tmp/test.log")):
            result = run_descriptions(self.ctx)
        a, b = result["data"]["segments"]
        self.assertEqual(a["emotion"]["status"], "unknown")
        self.assertEqual(b["emotion"]["valence"], 5)
        self.assertEqual(result["status"], "partial")
        self.assertIn("music2emo", result["provenance"]["engine"])
        self.assertTrue(any("emotion (1/2 segments)" in warning for warning in result["warnings"]))

    def test_model_invalid_numbers_and_duplicate_indices_are_rejected(self):
        for value in (float("nan"), float("inf"), 12, True):
            with self.assertRaises(ValueError):
                _validate_model_result({"emotion": {"valence": value, "arousal": 4}}, "test")
        self.ctx["config"] = {"backend": "essentia", "model_backends": {"essentia": {}}}
        item = {"index": 0, "data": {"genre": ["rock"]}}
        with patch("musicprep.adapters.descriptions._run_model_batch", return_value=({"results": [item, item]}, "test.log")):
            result = run_descriptions(self.ctx)
        self.assertTrue(any("duplicate" in warning for warning in result["warnings"]))
        self.assertTrue(any("did not return" in warning for warning in result["warnings"]))

    def test_misspelled_model_cannot_be_silently_ignored(self):
        self.ctx["config"] = {"backend": "models", "model_backends": {"qwen_audio_typo": {}}}
        with self.assertRaisesRegex(ValueError, "Unknown description"):
            run_descriptions(self.ctx)

    def test_timeout_reads_completed_checkpoint_instead_of_dropping_it(self):
        completed = {"results": [{"index": 0, "data": {"genre": ["rock"]}}]}
        for kind in ("timeout", "exit"):
            def interrupted(args, **kwargs):
                Path(args[-1]).write_text(json.dumps(completed))
                if kind == "timeout":
                    raise subprocess.TimeoutExpired(args, 1)
                raise subprocess.CalledProcessError(1, args)
            with self.subTest(kind=kind), patch("musicprep.adapters.descriptions.subprocess.run", side_effect=interrupted):
                output, log = _run_model_batch("qwen2_audio", {}, [], self.root)
            self.assertEqual(output["results"], completed["results"])
            self.assertIn("worker_error", output)

    def test_worker_interruption_preserves_results_and_prevents_success(self):
        self.ctx["config"] = {"backend": "qwen2_audio", "model_backends": {"qwen2_audio": {}}}
        def batch(engine, *args):
            return {"results": [{"index": 0, "data": {"genre": ["pop"], "instruments": [], "textures": []}}],
                    "worker_error": "TimeoutExpired"}, "log"
        with patch("musicprep.adapters.descriptions._run_model_batch", side_effect=batch):
            result = run_descriptions(self.ctx)
        model = result["provenance"]["models"]["qwen2_audio"]
        self.assertEqual(model["status"], "partial")
        self.assertEqual(model["counts"]["succeeded"], 1)
        self.assertEqual(model["counts"]["missing"], 1)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["data"]["segments"][0]["model_results"]["qwen2_audio"]["data"]["genre"][0]["label"], "pop")

        completed, _ = batch("qwen2_audio")
        completed["results"].append(dict(completed["results"][0], index=1))
        with patch("musicprep.adapters.descriptions._run_model_batch", return_value=(completed, "log")):
            result = run_descriptions(self.ctx)
        model = result["provenance"]["models"]["qwen2_audio"]
        self.assertEqual(model["counts"]["succeeded"], 2)
        self.assertEqual(model["counts"]["missing"], 0)
        self.assertEqual(model["status"], "partial")

    def test_no_configured_model_falls_back_or_fails_as_configured(self):
        self.ctx["config"] = {"backend": "music2emo"}
        self.assertEqual(run_descriptions(self.ctx)["status"], "partial")
        self.ctx["config"]["fallback"] = False
        with self.assertRaises(ValueError):
            run_descriptions(self.ctx)

    def test_relative_model_paths_survive_isolated_worker_cwd(self):
        cfg = {"repo": "models/vendor/Music2Emotion", "python": ".venv/bin/python"}

        def run(args, **kwargs):
            request = json.loads(Path(args[args.index("--request") + 1]).read_text())
            self.assertEqual(request["config"]["repo"], str(Path(cfg["repo"]).resolve()))
            self.assertEqual(args[0], str(Path(cfg["python"]).absolute()))
            Path(args[args.index("--output") + 1]).write_text('{"results": []}')

        with patch("musicprep.adapters.descriptions.subprocess.run", side_effect=run):
            result, _ = _run_model_batch("music2emo", cfg, [], self.root)
        self.assertEqual(result, {"results": []})
        self.assertEqual(cfg["repo"], "models/vendor/Music2Emotion")

    def _all_models(self, qwen):
        self.ctx["config"] = {"backend": "auto", "model_backends": {name: {} for name in ("essentia", "music2emo", "qwen2_audio")}}
        def batch(engine, *args):
            if engine == "qwen2_audio" and isinstance(qwen, Exception):
                raise qwen
            data = {"essentia": {"genre": [{"label": "rock", "score": .7}], "instruments": ["guitar"]},
                    "music2emo": {"emotion": {"valence": 5, "arousal": 4}}, "qwen2_audio": qwen}[engine]
            return {"results": [{"index": i, "data": data} for i in (0, 1)]}, "test.log"
        with patch("musicprep.adapters.descriptions._run_model_batch", side_effect=batch):
            return run_descriptions(self.ctx)

    def test_qwen_empty_tags_do_not_erase_classifier_evidence(self):
        result = self._all_models({"genre": [], "instruments": [], "textures": ["sustained"]})
        self.assertEqual(result["status"], "succeeded")
        for row in result["data"]["segments"]:
            self.assertEqual(row["genre"], [{"label": "rock", "score": .7, "source": "essentia", "status": "candidate"}])
            self.assertEqual(row["model_results"]["qwen2_audio"]["data"]["genre"], [])
            self.assertTrue(any(tag["source"] == "qwen2_audio" for tag in row["textures"]))
        self.assertEqual(result["provenance"]["models"]["qwen2_audio"]["counts"]["succeeded"], 2)

    def test_qwen_contradiction_keeps_both_sources_distinct(self):
        result = self._all_models({"genre": ["classical"], "instruments": ["piano"], "textures": []})
        self.assertEqual([(x["label"], x["source"]) for x in result["data"]["segments"][0]["genre"]],
                         [("rock", "essentia"), ("classical", "qwen2_audio")])

    def test_configured_qwen_failure_cannot_be_hidden_by_other_models(self):
        result = self._all_models(ValueError("model cannot load"))
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["provenance"]["models"]["qwen2_audio"]["status"], "failed")
        self.assertEqual(result["provenance"]["models"]["qwen2_audio"]["counts"]["failed"], 2)
        self.assertTrue(result["data"]["segments"][0]["genre"])

    def test_qwen_cannot_replace_emotion_with_generated_numbers(self):
        result = self._all_models({"emotion": {"valence": 9, "arousal": 9}, "genre": []})
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["provenance"]["models"]["qwen2_audio"]["counts"]["rejected"], 2)
        self.assertEqual(result["data"]["segments"][0]["emotion"]["valence"], 5)


if __name__ == "__main__":
    unittest.main()
