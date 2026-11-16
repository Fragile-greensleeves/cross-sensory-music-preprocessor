import array
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
import wave


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_audio.py"
spec = importlib.util.spec_from_file_location("prepare_audio", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg and ffprobe required")
class PrepareAudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "测试 音频.wav"
        self.samples = array.array("h")
        # 0.1s initial silence, then different left/right levels, then 0.1s trailing silence.
        for index in range(8000):
            self.samples.extend((0, 0) if index < 800 or index >= 7200 else (8192, -4096))
        with wave.open(str(self.source), "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(8000)
            handle.writeframes(self.samples.tobytes())

    def tearDown(self):
        self.temp.cleanup()

    def test_preserves_timing_channels_silence_and_amplitude(self):
        out = self.root / "output"
        manifest = module.prepare_audio(self.source, out, "fixture")
        self.assertEqual(manifest["canonical"]["sample_rate"], 8000)
        self.assertEqual(manifest["canonical"]["channels"], 2)
        self.assertAlmostEqual(manifest["canonical"]["duration_seconds"], 1.0)
        decoded = subprocess.run([
            "ffmpeg", "-v", "error", "-i", str(out / "canonical.wav"),
            "-f", "f32le", "-c:a", "pcm_f32le", "-"
        ], check=True, capture_output=True).stdout
        values = struct.unpack("<" + "f" * (len(decoded) // 4), decoded)
        self.assertEqual(len(values), len(self.samples))
        for actual, original in zip(values, self.samples):
            self.assertAlmostEqual(actual, original / 32768.0, places=6)
        saved = json.loads((out / "manifest.json").read_text())
        self.assertEqual(saved["source"]["sha256"], module.sha256_file(self.source))
        self.assertEqual(saved["timeline"]["offset_seconds"], 0)
        self.assertNotIn(str(self.root), json.dumps(saved))

    def test_existing_output_is_not_overwritten(self):
        out = self.root / "output"
        out.mkdir()
        sentinel = out / "keep.txt"
        sentinel.write_text("keep")
        with self.assertRaises(module.PreparationError):
            module.prepare_audio(self.source, out, "fixture")
        self.assertEqual(sentinel.read_text(), "keep")

    def test_corrupt_audio_does_not_leave_successful_output(self):
        bad = self.root / "broken.flac"
        bad.write_bytes(b"not an audio file")
        out = self.root / "output"
        with self.assertRaises(module.PreparationError):
            module.prepare_audio(bad, out, "fixture")
        self.assertFalse(out.exists())

    def test_invalid_id_is_rejected(self):
        with self.assertRaises(module.PreparationError):
            module.prepare_audio(self.source, self.root / "output", "../escape")

    def test_missing_input_is_rejected(self):
        with self.assertRaises(module.PreparationError):
            module.prepare_audio(self.root / "missing.wav", self.root / "output", "fixture")


if __name__ == "__main__":
    unittest.main()
