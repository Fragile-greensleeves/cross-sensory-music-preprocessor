"""Signal-based checks: no invented notes/beats and bounded event timing."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from musicprep.adapters.events import run_melody, run_harmony, run_rhythm, _rhythm_data


class EventTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.sr = 16000

    def context(self, audio, **config):
        path = Path(self.temp.name) / "audio.wav"
        sf.write(path, audio, self.sr, subtype="FLOAT")
        return {"audio_path": str(path), "duration": len(audio) / self.sr,
                "sample_rate": self.sr, "channels": 1, "work_dir": self.temp.name,
                "dependencies": {}, "config": {"backend": "baseline", **config}}

    def test_silence_has_no_notes_or_beats(self):
        ctx = self.context(np.zeros(self.sr))
        melody, rhythm = run_melody(ctx), run_rhythm(ctx)
        self.assertEqual(melody["data"]["notes"], [])
        self.assertTrue(all(not row["valid"] for row in melody["data"]["contour"]))
        self.assertEqual(rhythm["data"]["beats"], [])
        self.assertEqual(rhythm["data"]["tempo"], [])
        self.assertEqual(run_harmony(ctx)["data"]["segments"][0]["label"], "N")

    def test_same_pitch_after_silence_remains_two_notes(self):
        t = np.arange(self.sr * 3) / self.sr
        envelope = ((t >= .2) & (t < 1.1) | (t >= 1.6) & (t < 2.7))
        ctx = self.context(.4 * np.sin(2 * np.pi * 440 * t) * envelope)
        notes = run_melody(ctx)["data"]["notes"]
        self.assertGreaterEqual(len(notes), 2)
        self.assertTrue(any(note["start"] < .4 and note["end"] > .9 for note in notes))
        self.assertTrue(any(note["start"] > 1.4 and note["end"] > 2.5 for note in notes))
        self.assertTrue(all(note["midi"] == 69 for note in notes))
        self.assertTrue(all(0 <= note["start"] < note["end"] <= ctx["duration"] for note in notes))

    def test_pitch_change_and_real_file_end(self):
        t = np.arange(round(self.sr * 2.013)) / self.sr
        audio = .3 * np.sin(2 * np.pi * np.where(t < 1, 440, 523.251) * t)
        ctx = self.context(audio)
        result = run_melody(ctx)["data"]
        self.assertIn(69, [n["midi"] for n in result["notes"]])
        self.assertIn(72, [n["midi"] for n in result["notes"]])
        self.assertTrue(all(n["end"] <= ctx["duration"] for n in result["notes"]))
        self.assertTrue(all(n["time"] < ctx["duration"] for n in result["contour"]))

    def test_instrumental_gap_routing_is_explicit_and_rejects_silence(self):
        t = np.arange(self.sr * 3) / self.sr
        ctx = self.context(np.zeros(len(t)))
        other = Path(self.temp.name) / "other.wav"
        tone = .3 * np.sin(2 * np.pi * 440 * t) * (t < 2)
        sf.write(other, tone, self.sr, subtype="FLOAT")
        ctx["dependencies"] = {"separation": {"stems": {"vocals": ctx["audio_path"], "other": str(other)}}}
        result = run_melody(ctx)
        self.assertEqual(result["status"], "partial")
        self.assertTrue(result["data"]["notes"])
        self.assertTrue(all(n["source"] == "other_instrumental_candidate" for n in result["data"]["notes"]))
        self.assertTrue(all(not row["valid"] for row in result["data"]["contour"] if row["time"] > 2.2))
        self.assertTrue(any("intended melody" in warning for warning in result["warnings"]))

    def test_major_triad_has_correct_pitch_classes(self):
        t = np.arange(self.sr * 2) / self.sr
        y = sum(np.sin(2 * np.pi * f * t) for f in (261.6256, 329.6276, 391.9954)) * .15
        ctx = self.context(y)
        segments = run_harmony(ctx)["data"]["segments"]
        cmajor = sum(s["end"] - s["start"] for s in segments if s["label"] == "C:maj")
        self.assertGreater(cmajor, 1.5)
        self.assertTrue(all(s["pitch_classes"] == [0, 4, 7] for s in segments if s["label"] == "C:maj"))
        self.assertEqual(segments[0]["start"], 0)
        self.assertEqual(segments[-1]["end"], 2)

    def test_regular_clicks_have_half_second_intervals_not_fake_bars(self):
        y = np.zeros(self.sr * 8)
        click = np.random.default_rng(4).normal(size=320) * np.exp(-np.arange(320) / 50)
        for time in np.arange(.5, 7.6, .5):
            start = round(time * self.sr)
            y[start:start + len(click)] += click
        result = run_rhythm(self.context(y))["data"]
        self.assertGreater(len(result["beats"]), 8)
        self.assertAlmostEqual(float(np.median(np.diff([b["time"] for b in result["beats"]]))), .5, delta=.04)
        self.assertTrue(all(b["is_downbeat"] is None and b["bar_index"] is None for b in result["beats"]))
        self.assertEqual(result["meter"], [])

    def test_downbeat_assignment_preserves_pickup_and_notated_meter_unknown(self):
        data = _rhythm_data([0, .5, 1, 1.5, 2, 2.5, 3, 3.5], [1, 3], 4, {})
        self.assertIsNone(data["beats"][0]["bar_index"])
        self.assertEqual(data["beats"][2]["beat_in_bar"], 1)
        self.assertEqual(data["beats"][6]["bar_index"], 1)
        self.assertEqual(data["meter"], [])

    def test_neural_failure_is_visible_and_can_be_fatal(self):
        ctx = self.context(np.zeros(self.sr), backend="auto")
        with patch("musicprep.adapters.events._basic_pitch_melody", side_effect=ImportError("missing model")):
            result = run_melody(ctx)
            self.assertEqual(result["status"], "partial")
            self.assertIn("missing model", result["warnings"][0])
            ctx["config"]["fallback"] = False
            with self.assertRaises(ImportError):
                run_melody(ctx)


if __name__ == "__main__":
    unittest.main()
