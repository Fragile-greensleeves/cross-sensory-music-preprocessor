import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from musicprep.adapters.lyrics import run_lyrics_context


class LyricsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ctx = {"duration": 10.0, "work_dir": str(self.root / "out"), "config": {}, "optional_inputs": {}}

    def text(self, name, content):
        path = self.root / name
        path.write_text(content, encoding="utf-8")
        return str(path)

    def test_missing_text_is_skipped_not_transcribed_or_invented(self):
        result = run_lyrics_context(self.ctx)
        self.assertEqual(result["status"], "skipped")
        self.assertTrue(all(not value for value in result["data"].values()))

    def test_repeated_lrc_lyrics_at_different_times_are_preserved(self):
        self.ctx["optional_inputs"]["lyrics"] = self.text("song.lrc", "[00:01.00][00:04.00]雨\n[00:01.00]雨\n[00:08.00]再见\n")
        result = run_lyrics_context(self.ctx)
        lyrics = result["data"]["lyrics"]
        self.assertEqual([x["start"] for x in lyrics], [1, 4, 8])
        self.assertEqual([x["end"] for x in lyrics], [4, 8, 10])
        rain = next(x for x in result["data"]["imagery"] if x["label"] == "rain")
        self.assertEqual(len(rain["evidence"]), 2)

    def test_bad_timestamps_are_not_clamped_into_valid_singing(self):
        self.ctx["optional_inputs"]["lyrics"] = self.text("song.lrc", "[offset:1000]\n[00:00.00]a\n[00:02.00]b\n[00:20.00]c\n[00:99.00]d")
        result = run_lyrics_context(self.ctx)
        self.assertEqual([x["text"] for x in result["data"]["lyrics"]], ["b"])
        self.assertEqual(result["status"], "partial")

    def test_lrc_offset_positive_advances_negative_delays(self):
        for offset, expected in ((500, 1.5), (-500, 2.5)):
            with self.subTest(offset=offset):
                self.ctx["optional_inputs"]["lyrics"] = self.text("song.lrc", f"[offset:{offset}]\n[00:02]line")
                result = run_lyrics_context(self.ctx)
                self.assertEqual(result["data"]["lyrics"][0]["start"], expected)

    def test_blank_lrc_timestamp_clears_previous_lyric(self):
        self.ctx["optional_inputs"]["lyrics"] = self.text("song.lrc", "[00:01]hello\n[00:03]\n[00:09]next")
        result = run_lyrics_context(self.ctx)
        lyrics = result["data"]["lyrics"]
        self.assertEqual(len(lyrics), 2)
        self.assertEqual((lyrics[0]["start"], lyrics[0]["end"]), (1, 3))
        self.assertEqual((lyrics[1]["start"], lyrics[1]["end"]), (9, 10))

    def test_plain_text_not_given_fictitious_line_times(self):
        self.ctx["optional_inputs"]["lyrics"] = self.text("song.txt", "记得这场雨\n再见")
        result = run_lyrics_context(self.ctx)
        self.assertEqual(result["status"], "partial")
        self.assertTrue(all(item["start"] is None for item in result["data"]["lyrics"]))
        self.assertEqual(result["data"]["narrative"][0]["source"], "extractive")

    def test_background_is_preserved_as_unverified_evidence(self):
        self.ctx["optional_inputs"]["background"] = self.text("background.json", '[{"text":"Supplied claim","url":"https://example.com/source"},{"wrong":1}]')
        result = run_lyrics_context(self.ctx)
        self.assertEqual(len(result["data"]["background"]), 1)
        self.assertEqual(result["data"]["background"][0]["verification_status"], "unverified")
        self.assertEqual(result["data"]["themes"], [])
        self.assertEqual(result["status"], "partial")

    def test_malformed_json_or_missing_file_does_not_crash_other_input(self):
        self.ctx["optional_inputs"] = {"background": self.text("bad.json", "{"),
                                       "lyrics": self.text("valid.lrc", "[00:01]sample")}
        result = run_lyrics_context(self.ctx)
        self.assertEqual(len(result["data"]["lyrics"]), 1)
        self.assertEqual(result["status"], "partial")
        self.ctx["optional_inputs"] = {"lyrics": str(self.root / "missing.txt")}
        self.assertEqual(run_lyrics_context(self.ctx)["status"], "partial")

    def test_sofa_uses_real_output_and_rejects_syllable_mismatch(self):
        repo = self.root / "sofa"
        repo.mkdir()
        (repo / "infer.py").write_text("", encoding="utf-8")
        checkpoint = repo / "weights.ckpt"
        checkpoint.write_bytes(b"test")
        dictionary = repo / "dictionary.txt"
        dictionary.write_text("a a\nb b", encoding="utf-8")
        audio_path = self.root / "audio.wav"
        sf.write(audio_path, np.zeros(22050 * 10), 22050)
        self.ctx["audio_path"] = str(audio_path)
        self.ctx["optional_inputs"]["lyrics"] = self.text("song.txt", "a\nb")
        self.ctx["config"] = {"backend": "sofa", "sofa": {"repo": str(repo), "checkpoint": str(checkpoint),
                              "dictionary": str(dictionary), "text_type": "pinyin"}}

        def run(args, **kwargs):
            folder = Path(args[args.index("--folder") + 1])
            target = folder / "htk/words/lyrics.lab"
            target.parent.mkdir(parents=True)
            target.write_text("10000000 20000000 a\n20000000 30000000 SP\n40000000 50000000 b", encoding="utf-8")

        with patch("musicprep.adapters.lyrics.subprocess.run", side_effect=run):
            result = run_lyrics_context(self.ctx)
        self.assertEqual([item["start"] for item in result["data"]["lyrics"]], [1, 4])
        self.assertEqual(result["status"], "succeeded")
        self.ctx["optional_inputs"]["lyrics"] = self.text("mismatch.txt", "a\nc")
        with patch("musicprep.adapters.lyrics.subprocess.run", side_effect=run):
            result = run_lyrics_context(self.ctx)
        self.assertEqual(result["status"], "partial")
        self.assertTrue(all(item["start"] is None for item in result["data"]["lyrics"]))


if __name__ == "__main__":
    unittest.main()
