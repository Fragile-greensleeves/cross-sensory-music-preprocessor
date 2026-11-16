import copy
import unittest

from musicprep.reference import package_digest, supplement_gaps
from musicprep.validation import validate_package
from test_validation import package_fixture


class ReferenceTest(unittest.TestCase):
    def setUp(self):
        self.package = package_fixture()
        self.reference = {"audio_sha256": self.package["audio"]["source_sha256"],
                          "base_package_sha256": package_digest(self.package),
                          "sources": [{"name": "synthetic score"}],
                          "method": "test fixture", "notes": []}

    def proposal(self, start=2.1, end=2.5, midi=62):
        return {"start": start, "end": end, "midi": midi, "evidence": {"kind": "synthetic"}}

    def test_gap_addition_preserves_all_original_data(self):
        original = copy.deepcopy(self.package)
        self.reference["notes"] = [self.proposal()]
        result = supplement_gaps(self.package, self.reference)
        self.assertEqual(self.package, original)
        self.assertEqual(validate_package(result), [])
        for key in original:
            if key != "melody":
                self.assertEqual(result[key], original[key])
        self.assertEqual(result["melody"]["contour"], original["melody"]["contour"])
        self.assertEqual(result["melody"]["notes"][0], original["melody"]["notes"][0])
        self.assertEqual(result["melody"]["notes"][1]["source"], "score_assisted_gap")
        self.assertEqual(result["run"]["status"], "partial")

    def test_conflicting_or_duplicate_events_never_overwrite(self):
        self.reference["notes"] = [self.proposal(1.5, 2.3, 80), self.proposal(), self.proposal()]
        result = supplement_gaps(self.package, self.reference)
        audit = result["melody"]["provenance"]["score_supplement"]
        self.assertEqual(audit["added_count"], 1)
        self.assertEqual(audit["rejected_count"], 2)

    def test_wrong_audio_or_stale_baseline_rejected(self):
        for key in ("audio_sha256", "base_package_sha256"):
            ref = copy.deepcopy(self.reference); ref[key] = "wrong"
            with self.assertRaises(ValueError):
                supplement_gaps(self.package, ref)

    def test_invalid_notes_rejected(self):
        for note in (self.proposal(-1, 0), self.proposal(9, 11), self.proposal(midi=float("nan")),
                     self.proposal(midi=True), self.proposal(midi=128)):
            self.reference["notes"] = [note]
            with self.assertRaises(ValueError):
                supplement_gaps(self.package, self.reference)

    def test_measured_pitch_conflict_is_preserved(self):
        self.package["melody"]["contour"].append({"time": 2.3, "valid": True, "midi_float": 65, "f0_hz": 349.23})
        self.reference["base_package_sha256"] = package_digest(self.package)
        self.reference["notes"] = [self.proposal()]
        result = supplement_gaps(self.package, self.reference)
        self.assertEqual(result["melody"]["notes"], self.package["melody"]["notes"])
        self.assertEqual(result["melody"]["provenance"]["score_supplement"]["audit"][0]["reason"],
                         "conflicting_measured_contour_preserved")

    def test_second_supplement_retains_previous_evidence(self):
        self.reference["notes"] = [self.proposal()]
        first = supplement_gaps(self.package, self.reference)
        ref = copy.deepcopy(self.reference)
        ref["base_package_sha256"] = package_digest(first)
        ref["notes"] = [self.proposal(3.1, 3.5, 65)]
        second = supplement_gaps(first, ref)
        self.assertEqual(len(second["melody"]["notes"]), 3)
        self.assertEqual(second["melody"]["provenance"]["score_supplement"]["previous_supplement"],
                         first["melody"]["provenance"]["score_supplement"])


class MelodyReviewTest(unittest.TestCase):
    def setUp(self):
        self.p = package_fixture()
        old = self.p['melody']['notes'][0]
        self.review = {'audio_sha256': self.p['audio']['source_sha256'],
                       'base_package_sha256': package_digest(self.p),
                       'sources': [{'name': 'synthetic evidence'}],
                       'decisions': [{'resolution': 'onset correction'}],
                       'windows': [[.9, 2.0]],
                       'notes_replace': [{'before': copy.deepcopy(old),
                                          'after': dict(old, start=.9)}]}

    def test_review_is_exact_audited_and_preserves_other_data(self):
        from musicprep.reference import apply_melody_review
        original = copy.deepcopy(self.p)
        result = apply_melody_review(self.p, self.review)
        self.assertEqual(self.p, original)
        self.assertEqual(result['melody']['notes'][0]['start'], .9)
        self.assertEqual(result['melody']['contour'], original['melody']['contour'])
        self.assertEqual(result['melody']['provenance']['melody_review'], self.review)
        for key in original:
            if key != 'melody':
                self.assertEqual(result[key], original[key])

    def test_stale_out_of_scope_or_invalid_changes_rejected(self):
        from musicprep.reference import apply_melody_review
        cases = []
        r = copy.deepcopy(self.review); r['base_package_sha256'] = 'stale'; cases.append(r)
        r = copy.deepcopy(self.review); r['notes_replace'][0]['before']['midi'] = 61; cases.append(r)
        r = copy.deepcopy(self.review); r['notes_replace'][0]['after']['start'] = .8; cases.append(r)
        r = copy.deepcopy(self.review); r['notes_replace'][0]['after']['midi'] = 200; cases.append(r)
        r = copy.deepcopy(self.review); r['notes_add'] = [dict(self.p['melody']['notes'][0])]; cases.append(r)
        for r in cases:
            with self.subTest(review=r), self.assertRaises(ValueError):
                apply_melody_review(self.p, r)

    def test_invalid_contour_cannot_be_filled_without_valid_frequency(self):
        from musicprep.reference import apply_melody_review
        before = self.p['melody']['contour'][1]
        self.review['contour_replace'] = [{'before': before, 'after': dict(before, valid=True)}]
        with self.assertRaises(ValueError):
            apply_melody_review(self.p, self.review)
