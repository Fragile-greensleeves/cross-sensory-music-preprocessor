import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import hashlib

import numpy as np
import soundfile as sf
from musicprep.adapters.qwen import parse_description, audio_token_lengths, _read_audio, patch_official_audio_frontend, make_predictor


class QwenContractTests(unittest.TestCase):
    def test_custom_revision_or_checksum_cannot_claim_pinned_model_identity(self):
        for config in ({"revision": "another-model"}, {"expected_sha256": "0" * 64}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                make_predictor(config)

    def test_json_fence_and_deduplication(self):
        result = parse_description('```json\n{"genre":["Pop","pop"],"instruments":["guitar"],"textures":["warm"]}\n```')
        self.assertEqual(result, {"genre": ["Pop"], "instruments": ["guitar"], "textures": ["warm"]})

    def test_bounded_quote_normalization_keeps_labels_and_rejects_prose(self):
        expected = {"genre": ["pop"], "instruments": [], "textures": []}
        raw = json.dumps(expected)
        self.assertEqual(parse_description(raw.replace('"', r'\"')), expected)
        self.assertEqual(parse_description(json.dumps(raw)), expected)
        with self.assertRaises(ValueError):
            parse_description("Here is my answer: " + raw)
        with self.assertRaises(ValueError):
            parse_description(raw.replace('"', r'\\"'))

    def test_real_model_escaped_json_and_ambiguous_internal_quotes(self):
        raw = r'{\"genre\": [\"rock\", \"indie-rock\"], \"instruments\": [\"acoustic guitar\", \"electric guitar\", \"drums\", \"bass\", \"synthesizer\", \"guitar pedal effects\"], \"textures\": [\"grainy\", \"dusty\", \"crunchy\"]}'
        result = parse_description(raw)
        self.assertEqual(result["genre"], ["rock", "indie-rock"])
        self.assertEqual(len(result["instruments"]), 6)
        quoted_label = json.dumps({"genre": [], "instruments": ['lead "guitar"'], "textures": []})
        with self.assertRaises(ValueError):
            parse_description(quoted_label.replace('"', r'\"'))

    def test_prose_or_emotion_is_not_converted_into_labels(self):
        for text in ('A happy pop song with a guitar.', '{"genre": ["pop"], "instruments":[],"textures":[],"emotion":"happy"}',
                     '{"genre": "pop", "instruments":[],"textures":[]}',
                     '{"genre":[],"instruments":[0.9],"textures":[]}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_description(text)

    def test_ambiguous_multiple_json_objects_fail(self):
        value = json.dumps({"genre": [], "instruments": [], "textures": []})
        with self.assertRaises(ValueError):
            parse_description(value + value)

    def test_actual_audio_lengths_not_fixed_750_tokens(self):
        self.assertEqual(audio_token_lengths(800), (400, 200))
        self.assertEqual(audio_token_lengths(3000), (1500, 750))
        with self.assertRaises(ValueError):
            audio_token_lengths(3001)

    def test_audio_resampled_and_long_clip_rejected_without_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            sf.write(path, np.zeros((22050, 2)), 22050)
            audio, duration = _read_audio(path)
            self.assertEqual(len(audio), 16000)
            self.assertEqual(duration, 1)
            sf.write(path, np.zeros(16000 * 31), 16000)
            with self.assertRaises(ValueError):
                _read_audio(path)

    def test_setup_repairs_missing_tokenizer_without_redownloading_valid_weights(self):
        source = Path(__file__).resolve().parents[1] / "scripts/setup_qwen.py"
        spec = importlib.util.spec_from_file_location("qwen_setup_test", source)
        setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(setup)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / ".venv-qwen/bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
            model = root / "models/qwen2-audio-mlx-4bit"
            model.mkdir(parents=True)
            weights = model / "weights.safetensors"
            weights.write_bytes(b"data")
            calls = []

            def download(args, **kwargs):
                patterns = json.loads(args[-1])
                calls.append(patterns)
                self.assertNotIn("*.safetensors", patterns)
                for name in setup.REQUIRED_SMALL_FILES:
                    if name == "merges.txt":
                        (model / name).write_text("#version: 0.2\na b\n")
                    else:
                        value = {"test": True}
                        if name == "config.json":
                            value = {"model_type": "qwen2_audio", "quantization": {"bits": 4}}
                        if name == "preprocessor_config.json":
                            value = {"sampling_rate": 16000, "feature_size": 128}
                        (model / name).write_text(json.dumps(value))

            with patch.object(setup, "__file__", str(root / "scripts/setup_qwen.py")), \
                 patch.object(setup, "SIZE", 4), \
                 patch.object(setup, "SHA256", hashlib.sha256(b"data").hexdigest()), \
                 patch.object(setup.platform, "system", return_value="Darwin"), \
                 patch.object(setup.platform, "machine", return_value="arm64"), \
                 patch.object(setup.shutil, "which", return_value="/usr/bin/uv"), \
                 patch.object(setup.subprocess, "run", side_effect=download):
                setup.main(["--skip-install"])
            self.assertEqual(len(calls), 1)
            self.assertTrue((model / "tokenizer.json").is_file())
            self.assertEqual(weights.read_bytes(), b"data")


@unittest.skipUnless(importlib.util.find_spec("mlx") is not None, "Run numerical frontend test in .venv-qwen")
class QwenMLXFrontendTests(unittest.TestCase):
    def test_padding_keys_do_not_change_valid_audio_features(self):
        import mlx.core as mx
        import mlx.nn as nn
        from transformers import WhisperFeatureExtractor
        from mlx_audio.stt.models.qwen2_audio.qwen2_audio import Qwen2AudioEncoder
        from mlx_audio.stt.models.qwen2_audio.config import EncoderConfig

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.audio_tower = Qwen2AudioEncoder(EncoderConfig(d_model=8, encoder_layers=1,
                    encoder_attention_heads=2, encoder_ffn_dim=16, num_mel_bins=128))
                self.multi_modal_projector = nn.Linear(8, 8)
                self._processor = SimpleNamespace(feature_extractor=WhisperFeatureExtractor(feature_size=128))
                # Packed quantized weights must never determine feature dtype.
                self.language_model = SimpleNamespace(model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=mx.zeros((1,), dtype=mx.uint32))))

        mx.random.seed(5)
        model = TinyModel()
        patch_official_audio_frontend(model)
        features, tokens = model._extract_features(np.zeros(16000, dtype=np.float32))
        self.assertEqual(tokens, 25)
        original = model.get_audio_features(features)[:, :tokens]
        changed = features.at[:, :, 200:].add(50)
        perturbed = model.get_audio_features(changed)[:, :tokens]
        mx.eval(original, perturbed)
        self.assertTrue(mx.issubdtype(original.dtype, mx.floating))
        np.testing.assert_allclose(np.asarray(original), np.asarray(perturbed), atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
