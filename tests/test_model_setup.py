import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("model_setup", Path(__file__).resolve().parents[1] / "scripts/setup_models.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class ModelSetupTests(unittest.TestCase):
    def test_download_requires_expected_hash_before_replacing_existing_asset(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "weights"
            target.write_bytes(b"old local asset")
            entry = {"id": "fixture", "url": "https://example.com/weights", "sha256": "0" * 64}
            with patch.object(setup.urllib.request, "urlopen", side_effect=lambda *a, **k: io.BytesIO(b"wrong bytes")), patch.object(setup.time, "sleep"):
                with self.assertRaises(ValueError):
                    setup.download(entry, target)
            self.assertEqual(target.read_bytes(), b"old local asset")
            self.assertFalse(target.with_suffix(".part").exists())

    def test_matching_asset_reuses_cache_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "weights"
            target.write_bytes(b"verified")
            entry = {"sha256": hashlib.sha256(b"verified").hexdigest()}
            with patch.object(setup.urllib.request, "urlopen") as network:
                self.assertEqual(setup.download(entry, target), "cached")
                network.assert_not_called()

    def test_modified_source_repository_is_not_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(setup, "git", side_effect=["expected", " M model.py"]):
                with self.assertRaises(ValueError):
                    setup.repository({"revision": "expected"}, Path(folder))
