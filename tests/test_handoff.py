import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from musicprep.validation import validate_package
from test_validation import package_fixture

spec = importlib.util.spec_from_file_location('handoff_export', Path(__file__).parents[1] / 'scripts/export_handoff.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class HandoffTest(unittest.TestCase):
    def test_public_export_is_valid_preserves_music_and_removes_private_payloads(self):
        p = package_fixture()
        secret = 'PRIVATE-LYRIC-NOT-FOR-PUBLICATION'
        p['lyrics_context']['lyrics'] = [{'text': secret, 'start': 1., 'end': 2., 'words': []}]
        p['run']['tasks']['melody']['log_path'] = '/Users/private/model.log'
        p['descriptions'][0]['raw_output'] = secret
        before = copy.deepcopy(p)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = module.export_handoff(p, tmp)
            data = (Path(tmp) / 'song-package.json').read_bytes()
            q = json.loads(data)
            report = (Path(tmp) / 'report.html').read_text()
        self.assertEqual(p, before)
        self.assertEqual(validate_package(q), [])
        self.assertEqual(q['melody']['notes'], p['melody']['notes'])
        self.assertEqual(q['melody']['contour'], p['melody']['contour'])
        self.assertEqual(q['rhythm']['beats'], p['rhythm']['beats'])
        self.assertEqual(q['run']['status'], 'partial')
        self.assertEqual(manifest['data_sha256'], hashlib.sha256(data).hexdigest())
        self.assertEqual(manifest['counts']['private_lyrics_lines'], 1)
        for content in (data.decode(), report):
            self.assertNotIn(secret, content)
            self.assertNotIn('/Users/private/', content)

    def test_invalid_input_does_not_produce_deliverable(self):
        p = package_fixture(); p['melody']['notes'][0]['end'] = 100
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            module.export_handoff(p, tmp)
            self.assertFalse((Path(tmp) / 'song-package.json').exists())
