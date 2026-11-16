"""Opt-in, gap-only score supplements; never alter an automatic baseline."""
import copy
import hashlib
import json
import math


def package_digest(package):
    data = json.dumps(package, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def supplement_gaps(package, reference):
    """Apply explicitly transcribed/timed reference notes, not automatic score OCR.

    The manifest is bound to one audio and one baseline. Existing note events
    always win, even if their pitch differs. Measured contours are never edited.
    Evidence in the manifest must be reviewed before this opt-in operation.
    """
    if reference.get("audio_sha256") != package["audio"]["source_sha256"]:
        raise ValueError("Reference belongs to a different audio file")
    if reference.get("base_package_sha256") != package_digest(package):
        raise ValueError("Reference does not match this baseline package")
    if not reference.get("sources") or not reference.get("method"):
        raise ValueError("Reference sources and transcription/alignment method are required")
    proposals = reference.get("notes")
    if not isinstance(proposals, list):
        raise ValueError("Reference notes must be an array")
    for note in proposals:
        for key in ("start", "end", "midi"):
            value = note.get(key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"Invalid reference note {key}")
        if not 0 <= note["start"] < note["end"] <= package["audio"]["duration_seconds"]:
            raise ValueError("Reference note is outside the audio timeline")
        if not 0 <= note["midi"] <= 127 or not note.get("evidence"):
            raise ValueError("Reference pitch and supporting evidence are required")

    result = copy.deepcopy(package)
    notes = result["melody"]["notes"]
    audit = []
    for proposal in sorted(proposals, key=lambda n: n["start"]):
        a, b, pitch = proposal["start"], proposal["end"], proposal["midi"]
        reason = None
        if any(n["start"] < b and n["end"] > a for n in notes):
            reason = "existing_note_preserved"
        elif any(f.get("valid") and a <= f["time"] < b
                 and abs(f["midi_float"] - pitch) > 0.5
                 for f in result["melody"]["contour"]):
            reason = "conflicting_measured_contour_preserved"
        audit.append({"start": a, "end": b, "midi": pitch,
                      "decision": "rejected" if reason else "added",
                      "reason": reason, "evidence": copy.deepcopy(proposal["evidence"])})
        if not reason:
            notes.append({"start": a, "end": b, "midi": pitch,
                          "source": "score_assisted_gap", "status": "estimated",
                          "quality": {"calibrated": False},
                          "reference_evidence": copy.deepcopy(proposal["evidence"])})
    notes.sort(key=lambda n: n["start"])
    provenance = result["melody"].setdefault("provenance", {})
    previous = provenance.get("score_supplement")
    provenance["score_supplement"] = {
        "method": reference["method"], "policy": "gap_only_original_wins",
        "base_package_sha256": reference["base_package_sha256"],
        "sources": copy.deepcopy(reference["sources"]),
        "added_count": sum(row["decision"] == "added" for row in audit),
        "rejected_count": sum(row["decision"] == "rejected" for row in audit),
        "limitations": reference.get("limitations", "Score-assisted estimates, not measured continuous pitch."),
        "audit": audit,
    }
    if previous:
        provenance["score_supplement"]["previous_supplement"] = previous
    return result


def apply_melody_review(package, review):
    """Apply evidence-reviewed, exact before/after edits as a separate transaction.

    Unlike supplement_gaps this explicitly permits correcting existing melody
    observations. The caller must provide evidence, scoped windows and the exact
    original records; all other data is preserved and the full edit is audited.
    """
    from .validation import validate_package

    if (review.get("audio_sha256") != package["audio"]["source_sha256"]
            or review.get("base_package_sha256") != package_digest(package)):
        raise ValueError("Review does not match this audio and baseline")
    if not review.get("sources") or not review.get("decisions"):
        raise ValueError("Review sources and evidence-backed decisions are required")
    duration = package["audio"]["duration_seconds"]
    windows = review.get("windows", [])
    if not windows or any(not 0 <= a < b <= duration for a, b in windows):
        raise ValueError("Review windows must be within the audio timeline")

    def inside(row, kind):
        a = row["time"] if kind == "contour" else row["start"]
        b = a if kind == "contour" else row["end"]
        return any(left <= a and b <= right for left, right in windows)

    result = copy.deepcopy(package)
    for kind in ("notes", "contour"):
        rows = result["melody"][kind]
        for edit in review.get(kind + "_replace", []):
            before, after = edit["before"], edit["after"]
            if rows.count(before) != 1 or not inside(before, kind) or not inside(after, kind):
                raise ValueError("Review record is stale, ambiguous or outside its window")
            if kind == "contour" and before["time"] != after["time"]:
                raise ValueError("A contour review cannot move timestamps")
            rows[rows.index(before)] = copy.deepcopy(after)
    for note in review.get("notes_add", []):
        if not inside(note, "notes") or any(
                n["start"] < note["end"] and n["end"] > note["start"]
                for n in result["melody"]["notes"]):
            raise ValueError("Added review note is out of scope or overlaps an existing note")
        result["melody"]["notes"].append(copy.deepcopy(note))
    result["melody"]["notes"].sort(key=lambda n: n["start"])
    # Replacements must not introduce an overlap absent in the baseline.
    for edit in review.get("notes_replace", []):
        after = edit["after"]
        for note in result["melody"]["notes"]:
            if note == after:
                continue
            if note["start"] < after["end"] and note["end"] > after["start"]:
                before = edit["before"]
                if not (note["start"] < before["end"] and note["end"] > before["start"]):
                    raise ValueError("Review introduces a note overlap")
    provenance = result["melody"].setdefault("provenance", {})
    previous = provenance.get("melody_review")
    provenance["melody_review"] = copy.deepcopy(review)
    if previous:
        provenance["melody_review"]["previous_review"] = previous
    errors = validate_package(result)
    if errors:
        raise ValueError("Invalid reviewed package: " + "; ".join(errors))
    return result
