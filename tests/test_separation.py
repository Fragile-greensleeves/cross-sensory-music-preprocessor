import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
from musicprep.adapters.separation import run_separation


class SeparationTests(unittest.TestCase):
    def test_hpss_keeps_time_and_does_not_claim_instrument_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            sr = 16000
            t = np.arange(sr * 2) / sr
            y = .3 * np.sin(2 * np.pi * 440 * t)
            y[sr // 2:sr // 2 + 100] += np.random.default_rng(1).normal(size=100) * .2
            path = Path(directory) / "input.wav"
            sf.write(path, y, sr, subtype="FLOAT")
            ctx = {"audio_path": str(path), "duration": 2, "work_dir": directory,
                   "config": {"backend": "baseline", "sample_rate": sr}}
            result = run_separation(ctx)
            self.assertEqual(result["status"], "partial")
            self.assertEqual(set(result["data"]["stems"]), {"harmonic", "percussive"})
            components = []
            for output in result["data"]["stems"].values():
                self.assertTrue(Path(output).is_absolute())
                audio, output_sr = sf.read(output)
                self.assertEqual(output_sr, sr)
                self.assertEqual(len(audio), len(y))
                components.append(audio)
            np.testing.assert_allclose(sum(components), y, atol=1e-6)
            ctx["config"]["backend"] = "demucs"
            with patch("musicprep.adapters.separation._demucs", side_effect=ImportError("missing weights")):
                result = run_separation(ctx)
                self.assertIn("missing weights", result["warnings"][0])
                ctx["config"]["fallback"] = False
                with self.assertRaises(ImportError):
                    run_separation(ctx)


if __name__ == "__main__":
    unittest.main()
