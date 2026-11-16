import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import soundfile as sf

from musicprep.pipeline import analyze
from musicprep.validation import validate_package


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class IntegrationTests(unittest.TestCase):
    def test_actual_workers_export_and_resume_without_optional_text(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "合成 测试.wav"
            sr = 16000
            signal = np.zeros(3 * sr)
            signal[sr:2*sr] = .2 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
            sf.write(source, signal, sr)
            first = analyze(source, root / "analysis", "synthetic", profile="baseline")
            self.assertEqual(validate_package(first), [])
            self.assertEqual(first["run"]["status"], "partial")
            self.assertFalse(any(t["status"] == "failed" for t in first["run"]["tasks"].values()))
            self.assertEqual(first["run"]["tasks"]["lyrics_context"]["status"], "skipped")
            self.assertTrue((root / "analysis/report.html").is_file())
            second = analyze(source, root / "analysis", "synthetic", profile="baseline")
            self.assertTrue(all(t["cached"] for t in second["run"]["tasks"].values()))
            self.assertEqual(first["melody"], second["melody"])
            self.assertEqual(json.loads((root / "analysis/song-package.json").read_text())["run"]["id"], second["run"]["id"])
