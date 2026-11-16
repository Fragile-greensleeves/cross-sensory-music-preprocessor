import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
from musicprep.adapters.structure import run_sections


class StructureTests(unittest.TestCase):
    def test_full_coverage_unknown_labels_and_bounded_fallback_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            duration, sr = 65.125, 16000
            path = Path(directory) / "input.wav"
            sf.write(path, np.zeros(round(duration * sr)), sr)
            ctx = {"audio_path": str(path), "duration": duration, "work_dir": directory,
                   "config": {"backend": "baseline", "fallback_window_seconds": 30}}
            result = run_sections(ctx)
            sections = result["data"]["segments"]
            self.assertEqual(sections[0]["start"], 0)
            self.assertEqual(sections[-1]["end"], duration)
            self.assertTrue(all(s["label"] == "unknown" and 0 < s["end"] - s["start"] <= 30 for s in sections))
            self.assertTrue(all(a["end"] == b["start"] for a, b in zip(sections, sections[1:])))
            ctx["config"]["backend"] = "auto"
            with patch("musicprep.adapters.structure._allin1", side_effect=ImportError("not installed")):
                fallback = run_sections(ctx)
                self.assertIn("not installed", fallback["warnings"][0])
                self.assertEqual(fallback["status"], "partial")


if __name__ == "__main__":
    unittest.main()
