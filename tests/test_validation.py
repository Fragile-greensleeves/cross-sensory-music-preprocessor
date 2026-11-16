"""Contract tests cover corrupt estimates, not implementation internals."""
import copy
import json
import math
import unittest
from pathlib import Path

from musicprep.validation import validate_package


def package_fixture():
    return {
        'schema_version': '0.1.0', 'track_id': 'test-track',
        'audio': {'source_sha256': 'a' * 64, 'duration_seconds': 10.0,
                  'sample_rate': 44100, 'channels': 2, 'time_origin': 'source_start'},
        'run': {'id': 'run-1', 'status': 'partial',
                'tasks': {'melody': {'status': 'partial', 'provenance': {'engine': 'test-only'}}}},
        'melody': {'notes': [{'start': 1.0, 'end': 2.0, 'midi': 60, 'source': 'vocals', 'status': 'estimated'}],
                   'contour': [{'time': 1.0, 'f0_hz': 261.63, 'midi_float': 60.0, 'valid': True},
                               {'time': 2.0, 'f0_hz': None, 'midi_float': None, 'valid': False}]},
        'harmony': {'segments': [{'start': 0.0, 'end': 10.0, 'label': 'C', 'pitch_classes': [0, 4, 7], 'status': 'estimated'}]},
        'rhythm': {'beats': [{'time': 1.0, 'is_downbeat': None, 'bar_index': None, 'beat_in_bar': None}],
                   'tempo': [{'start': 1.0, 'end': 9.0, 'bpm': 120.0, 'beat_unit': 'estimated_beat'}],
                   'meter': [{'start': 0.0, 'end': 10.0, 'numerator': None, 'denominator': None, 'status': 'unknown'}]},
        'sections': [{'id': 's1', 'start': 0.0, 'end': 10.0, 'label': 'unknown', 'source': 'fallback', 'status': 'unknown'}],
        'descriptions': [{'section_id': 's1', 'analysis_start': 0.0, 'analysis_end': 10.0,
                          'applies_start': 0.0, 'applies_end': 10.0, 'genre': [], 'instruments': [],
                          'textures': [], 'emotion': {'valence': None, 'arousal': None, 'scale': None}}],
        'lyrics_context': {key: [] for key in ('lyrics', 'themes', 'narrative', 'imagery', 'background', 'sources')},
        'warnings': ['This fixture is synthetic; not a music measurement.']
    }


class ValidationTest(unittest.TestCase):
    @staticmethod
    def package_with_words():
        package = package_fixture()
        package['lyrics_context']['lyrics'] = [{'text': 'a b', 'start': 1.0, 'end': 3.0,
            'words': [{'text': 'a', 'token': 'a', 'start': 1.0, 'end': 2.0, 'phonemes': ['a']},
                      {'text': 'b', 'token': 'b', 'start': 2.0, 'end': 3.0, 'phonemes': ['b']}]}]
        return package

    def test_valid_partial_unknowns_and_extensions(self):
        package = package_fixture()
        package['extension'] = {'nullable': None}
        before = copy.deepcopy(package)
        self.assertEqual(validate_package(package), [])
        self.assertEqual(package, before)

    def test_schema_accepts_fixture(self):
        import jsonschema
        schema = json.loads((Path(__file__).parents[1] / 'schemas/song-package.schema.json').read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(package_fixture(), schema)

    def test_invalid_time_order_overlap_and_bounds(self):
        cases = [
            ('end before start', lambda p: p['melody']['notes'][0].update(end=0.5)),
            ('negative', lambda p: p['rhythm']['beats'][0].update(time=-1)),
            ('past duration', lambda p: p['sections'][0].update(end=11)),
            ('duplicate time', lambda p: p['rhythm']['beats'].append(copy.deepcopy(p['rhythm']['beats'][0]))),
            ('overlap', lambda p: p['harmony']['segments'].append(copy.deepcopy(p['harmony']['segments'][0]))),
        ]
        for name, mutate in cases:
            with self.subTest(name=name):
                package = package_fixture(); mutate(package)
                self.assertTrue(validate_package(package))

    def test_nonfinite_anywhere(self):
        for value in (math.nan, math.inf, -math.inf):
            package = package_fixture(); package['extension'] = {'quality': value}
            self.assertTrue(any('finite' in e for e in validate_package(package)))

    def test_unknown_is_not_no_chord(self):
        package = package_fixture()
        chord = package['harmony']['segments'][0]
        chord.update(status='unknown', label=None, pitch_classes=[])
        self.assertEqual(validate_package(package), [])
        chord['label'] = 'N'
        self.assertTrue(any('no_chord' in e for e in validate_package(package)))
        chord['status'] = 'no_chord'
        self.assertEqual(validate_package(package), [])
        chord['pitch_classes'] = [0]
        self.assertTrue(validate_package(package))

    def test_invalid_pitch_and_duplicate_pitch_classes(self):
        package = package_fixture(); package['melody']['contour'][1]['f0_hz'] = 0
        self.assertTrue(validate_package(package))
        package = package_fixture(); package['harmony']['segments'][0]['pitch_classes'] = [0, 0, 7]
        self.assertTrue(validate_package(package))

    def test_emotion_requires_scale_and_context_covers_application(self):
        package = package_fixture(); desc = package['descriptions'][0]
        desc['emotion']['valence'] = 0.2
        self.assertTrue(any('scale' in e for e in validate_package(package)))
        desc['emotion']['scale'] = [0, 1]
        self.assertEqual(validate_package(package), [])
        desc['analysis_start'] = 1
        self.assertTrue(validate_package(package))

    def test_complete_cannot_hide_partial_tasks(self):
        package = package_fixture(); package['run']['status'] = 'complete'
        self.assertTrue(validate_package(package))

    def test_complete_optional_lyrics_task_is_type_safe(self):
        package = package_fixture()
        package['run']['status'] = 'complete'
        package['run']['tasks'] = {name: {'status': 'succeeded'} for name in
                                  ('melody', 'harmony', 'rhythm', 'sections', 'descriptions')}
        self.assertEqual(validate_package(package), [])
        for task in (None, False, [], 'succeeded', 1, {'status': 'partial'}):
            with self.subTest(task=task):
                package['run']['tasks']['lyrics_context'] = task
                errors = validate_package(package)
                self.assertTrue(any('supplied lyrics/context' in error for error in errors))
        for status in ('skipped', 'succeeded'):
            package['run']['tasks']['lyrics_context'] = {'status': status}
            self.assertEqual(validate_package(package), [])

    def test_word_intervals_are_optional_valid_and_not_mutated(self):
        package = self.package_with_words()
        before = copy.deepcopy(package)
        self.assertEqual(validate_package(package), [])
        self.assertEqual(package, before)
        line = package['lyrics_context']['lyrics'][0]
        del line['words']
        line.update(start=None, end=None)
        self.assertEqual(validate_package(package), [])

    def test_word_time_bounds_order_overlap_and_parent_containment(self):
        cases = [
            ('negative', lambda words: words[0].update(start=-1)),
            ('past duration', lambda words: words[1].update(end=11)),
            ('zero duration', lambda words: words[0].update(end=1)),
            ('reverse interval', lambda words: words[0].update(end=0.5)),
            ('reverse order', lambda words: words.reverse()),
            ('overlap', lambda words: words[1].update(start=1.5)),
            ('outside parent', lambda words: words[0].update(start=0.5)),
            ('nonfinite', lambda words: words[0].update(start=float('nan'))),
            ('boolean time', lambda words: words[0].update(start=True)),
            ('missing end', lambda words: words[0].pop('end')),
        ]
        for name, mutate in cases:
            with self.subTest(name=name):
                package = self.package_with_words()
                mutate(package['lyrics_context']['lyrics'][0]['words'])
                self.assertTrue(any('words' in error for error in validate_package(package)))
        package = self.package_with_words()
        package['lyrics_context']['lyrics'][0].update(start=None, end=None)
        self.assertTrue(any('timed parent' in error for error in validate_package(package)))

    def test_word_array_text_tokens_and_phonemes_reject_malformed_values(self):
        for value in (None, {}, 'not an array', 1, [None], ['word']):
            with self.subTest(words=value):
                package = self.package_with_words()
                package['lyrics_context']['lyrics'][0]['words'] = value
                self.assertTrue(any('words' in error for error in validate_package(package)))
        for key in ('text', 'token'):
            for value in (None, '', ' ', [], 3):
                with self.subTest(key=key, value=value):
                    package = self.package_with_words()
                    package['lyrics_context']['lyrics'][0]['words'][0][key] = value
                    self.assertTrue(any('.' + key in error for error in validate_package(package)))
        for value in (None, 'a', [], [None], [''], [1]):
            with self.subTest(phonemes=value):
                package = self.package_with_words()
                package['lyrics_context']['lyrics'][0]['words'][0]['phonemes'] = value
                self.assertTrue(any('phonemes' in error for error in validate_package(package)))

    def test_malformed_values_return_errors_not_type_errors(self):
        for value in ([], {}, False, 'bad', None):
            for path in [('run', 'status'), ('melody', 'notes', 0, 'status'),
                         ('harmony', 'segments', 0, 'label'), ('descriptions', 0, 'section_id')]:
                with self.subTest(value=value, path=path):
                    package = package_fixture(); target = package
                    for key in path[:-1]: target = target[key]
                    target[path[-1]] = value
                    result = validate_package(package)
                    self.assertIsInstance(result, list)


if __name__ == '__main__':
    unittest.main()
