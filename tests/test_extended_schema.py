"""Structural compatibility for aligned lyrics and model execution outcomes."""
import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from musicprep.report import _public_package
from tests.test_validation import package_fixture


class ExtendedSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema_path = Path(__file__).parents[1] / 'schemas/song-package.schema.json'
        cls.schema = json.loads(schema_path.read_text())
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema)

    @staticmethod
    def aligned_line():
        return {'text': 'Re', 'start': 1.0, 'end': 2.0, 'words': [
            {'text': 'Re', 'token': 'latin_re', 'start': 1.0, 'end': 2.0,
             'phonemes': ['r', 'ei'], 'pronunciation_source': 'explicit_alias'}]}

    def test_empty_untimed_lrc_and_aligned_lyrics_remain_valid(self):
        variants = [[], [{'text': 'Untimed'}],
                    [{'text': 'Untimed', 'start': None, 'end': None}],
                    [{'text': 'Legacy LRC', 'start': 1.0, 'end': 2.0}],
                    [self.aligned_line()]]
        for lyrics in variants:
            with self.subTest(lyrics=lyrics):
                package = package_fixture()
                package['lyrics_context']['lyrics'] = lyrics
                self.validator.validate(package)

    def test_malformed_word_fields_are_rejected(self):
        variants = []
        for key in ('text', 'token', 'start', 'end'):
            word = self.aligned_line()['words'][0]
            del word[key]
            variants.append(word)
        for key, value in [('text', ' '), ('text', 1), ('token', ''),
                           ('token', '\t'), ('start', -1), ('start', '1'),
                           ('start', True), ('end', None), ('end', -1),
                           ('phonemes', []), ('phonemes', 'r ei'),
                           ('phonemes', ['r', ' ']), ('phonemes', [1])]:
            word = self.aligned_line()['words'][0]
            word[key] = value
            variants.append(word)
        variants.extend([None, 'word', []])
        for word in variants:
            with self.subTest(word=word):
                package = package_fixture()
                line = self.aligned_line()
                line['words'] = [word]
                package['lyrics_context']['lyrics'] = [line]
                self.assertFalse(self.validator.is_valid(package))

    def test_lyrics_and_words_must_be_arrays_of_objects(self):
        for lyrics in ({}, ['text'], [None], [{'text': ' '}],
                       [{'text': 'x', 'words': {}}],
                       [{'text': 'x', 'words': None}]):
            with self.subTest(lyrics=lyrics):
                package = package_fixture()
                package['lyrics_context']['lyrics'] = lyrics
                self.assertFalse(self.validator.is_valid(package))

    def test_phonemes_optional_for_external_aligner(self):
        package = package_fixture()
        line = self.aligned_line()
        del line['words'][0]['phonemes']
        package['lyrics_context']['lyrics'] = [line]
        self.validator.validate(package)

    def test_model_result_outcomes_and_public_status_only(self):
        package = package_fixture()
        package['descriptions'][0]['model_results'] = {
            'caption': {'status': 'succeeded', 'data': {'caption': 'Candidate'},
                        'inference_status': 'generated'},
            'emotion': {'status': 'failed', 'error': 'Unavailable'},
            'tags': {'status': 'missing'},
            'rejected': {'status': 'rejected', 'error': 'Malformed output'},
        }
        package['lyrics_context']['lyrics'] = [self.aligned_line()]
        self.validator.validate(package)
        public = _public_package(package)
        self.assertEqual(public['lyrics_context']['lyrics'], [])
        self.assertEqual(public['descriptions'][0]['model_results']['caption'],
                         {'status': 'succeeded', 'inference_status': 'generated'})
        # The public review snapshot intentionally omits run.id. Validate its
        # redacted payload fields in a complete package, not that separate format.
        redacted_payload = copy.deepcopy(package)
        redacted_payload['lyrics_context'] = public['lyrics_context']
        redacted_payload['descriptions'] = public['descriptions']
        self.validator.validate(redacted_payload)

    def test_malformed_model_result_rejected(self):
        for result in ({}, [], None, {'status': 'invented'}, {'status': True},
                       {'status': 'succeeded', 'data': []},
                       {'status': 'failed', 'error': 1},
                       {'status': 'failed', 'error': ' '}):
            with self.subTest(result=result):
                package = package_fixture()
                package['descriptions'][0]['model_results'] = {'model': result}
                self.assertFalse(self.validator.is_valid(package))


if __name__ == '__main__':
    unittest.main()
