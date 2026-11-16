"""Validate the song-package contract without requiring a third-party runtime.

The JSON Schema describes structure; this module also checks temporal and music
semantics that JSON Schema cannot express. Unknown estimates stay nullable.
"""
import math
import re

VERSION = "0.1.0"
TASK_STATES = ("pending", "running", "succeeded", "partial", "failed", "skipped")
CONTENT_STATES = ("estimated", "unknown", "absent", "no_chord")


def validate_package(package):
    """Return human-readable validation errors; never mutate the package."""
    errors = []

    def error(path, message):
        if len(errors) < 500:
            errors.append(f"{path}: {message}")

    def number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    def integer(value):
        return isinstance(value, int) and not isinstance(value, bool)

    def finite_tree(value, path="$"):
        if isinstance(value, float) and not math.isfinite(value):
            error(path, "must be finite; NaN/Infinity are not JSON data")
        elif isinstance(value, dict):
            for key, child in value.items():
                finite_tree(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                finite_tree(child, f"{path}[{index}]")

    def obj(value, path):
        if not isinstance(value, dict):
            error(path, "must be an object")
            return {}
        return value

    def arr(value, path):
        if not isinstance(value, list):
            error(path, "must be an array")
            return []
        return value

    def string(value, path, nullable=False):
        if nullable and value is None:
            return
        if not isinstance(value, str) or not value.strip():
            error(path, "must be a non-empty string" + (" or null" if nullable else ""))

    def fields(record, path, names):
        for name in names:
            if name not in record:
                error(path + "." + name, "required field is missing")

    def status(record, path, harmony=False):
        allowed = CONTENT_STATES if harmony else CONTENT_STATES[:-1]
        if record.get("status") not in allowed:
            error(path + ".status", "unsupported content status")

    if not isinstance(package, dict):
        return ["$: must be an object"]
    finite_tree(package)
    required = ("schema_version", "track_id", "audio", "run", "melody", "harmony", "rhythm", "sections", "descriptions", "lyrics_context", "warnings")
    for name in required:
        if name not in package:
            error(name, "required field is missing")
    if package.get("schema_version") != VERSION:
        error("schema_version", "unsupported version; expected 0.1.0")
    string(package.get("track_id"), "track_id")
    audio = obj(package.get("audio"), "audio")
    duration = audio.get("duration_seconds")
    if not number(duration) or duration <= 0:
        error("audio.duration_seconds", "must be positive and finite")
        duration = 0
    for name in ("sample_rate", "channels"):
        if not integer(audio.get(name)) or audio[name] <= 0:
            error(f"audio.{name}", "must be a positive integer")
    if audio.get("time_origin") != "source_start":
        error("audio.time_origin", "must be source_start")
    if not isinstance(audio.get("source_sha256"), str) or not re.fullmatch(r"[0-9a-fA-F]{64}", audio["source_sha256"]):
        error("audio.source_sha256", "must be a SHA-256 hex string")

    def time_value(value, path):
        if not number(value) or value < 0 or value > duration + 1e-6:
            error(path, "must be a finite original-audio time within duration")
            return False
        return True

    def interval(record, path, start="start", end="end"):
        first, last = record.get(start), record.get(end)
        valid_first = time_value(first, f"{path}.{start}")
        valid_last = time_value(last, f"{path}.{end}")
        if valid_first and valid_last and last <= first:
            error(path, "interval end must be after start")

    def sorted_records(records, path, key, strict=False):
        previous = None
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                error(f"{path}[{index}]", "must be an object")
                continue
            value = record.get(key)
            if number(value):
                if previous is not None and (value < previous or (strict and value == previous)):
                    error(f"{path}[{index}].{key}", "timestamps must be ordered" + (" and unique" if strict else ""))
                previous = value

    run = obj(package.get("run"), "run")
    string(run.get("id"), "run.id")
    if run.get("status") not in ("complete", "partial", "failed"):
        error("run.status", "must be complete, partial or failed")
    tasks = obj(run.get("tasks"), "run.tasks")
    for name, task in tasks.items():
        task = obj(task, f"run.tasks.{name}")
        if task.get("status") not in TASK_STATES:
            error(f"run.tasks.{name}.status", "unsupported task status")
    if run.get("status") == "complete":
        for name in ("melody", "harmony", "rhythm", "sections", "descriptions"):
            if not isinstance(tasks.get(name), dict) or tasks[name].get("status") != "succeeded":
                error("run.status", f"complete requires successful {name}")

    melody = obj(package.get("melody"), "melody")
    notes = arr(melody.get("notes"), "melody.notes")
    sorted_records(notes, "melody.notes", "start")
    for index, note in enumerate(notes):
        path = f"melody.notes[{index}]"
        if not isinstance(note, dict):
            continue
        interval(note, path)
        fields(note, path, ("midi",))
        status(note, path)
        string(note.get("source"), path + ".source")
        pitch = note.get("midi")
        if pitch is None and note.get("status") in ("unknown", "absent"):
            pass
        elif not number(pitch) or not 0 <= pitch <= 127:
            error(path + ".midi", "must be a MIDI pitch in 0–127, or null for unknown/absent")
    contour = arr(melody.get("contour"), "melody.contour")
    sorted_records(contour, "melody.contour", "time", strict=True)
    for index, frame in enumerate(contour):
        path = f"melody.contour[{index}]"
        if not isinstance(frame, dict):
            continue
        fields(frame, path, ("time", "f0_hz", "midi_float", "valid"))
        time_value(frame.get("time"), path + ".time")
        if not isinstance(frame.get("valid"), bool):
            error(path + ".valid", "must be boolean")
        if frame.get("valid") is True:
            if not number(frame.get("f0_hz")) or frame["f0_hz"] <= 0:
                error(path + ".f0_hz", "valid pitch requires a positive frequency")
            if not number(frame.get("midi_float")):
                error(path + ".midi_float", "valid pitch requires a finite semitone value")
        elif frame.get("valid") is False:
            if frame.get("f0_hz") is not None or frame.get("midi_float") is not None:
                error(path, "invalid frames must use null pitch, not fabricated zero/held pitch")

    harmony = obj(package.get("harmony"), "harmony")
    chords = arr(harmony.get("segments"), "harmony.segments")
    sorted_records(chords, "harmony.segments", "start")
    prior_end = None
    for index, chord in enumerate(chords):
        path = f"harmony.segments[{index}]"
        if not isinstance(chord, dict):
            continue
        interval(chord, path)
        status(chord, path, harmony=True)
        fields(chord, path, ("label", "pitch_classes"))
        pcs = arr(chord.get("pitch_classes"), path + ".pitch_classes")
        if any(not integer(pc) or not 0 <= pc <= 11 for pc in pcs):
            error(path + ".pitch_classes", "must contain integers in 0–11")
        elif len(set(pcs)) != len(pcs):
            error(path + ".pitch_classes", "must contain unique pitch classes")
        if chord.get("status") in ("no_chord", "unknown", "absent") and pcs:
            error(path, "unknown/no_chord/absent must not invent chord pitch classes")
        label = chord.get("label")
        if chord.get("status") == "estimated":
            string(label, path + ".label")
            if not pcs:
                error(path + ".pitch_classes", "estimated chord requires pitch classes")
        elif label is not None and not isinstance(label, str):
            error(path + ".label", "must be a string or null")
        if chord.get("status") == "unknown" and label in ("N", "no_chord"):
            error(path, "unknown must not be encoded as no_chord")
        if number(chord.get("start")) and prior_end is not None and chord["start"] < prior_end - 1e-6:
            error(path, "chord intervals must not overlap")
        if number(chord.get("end")):
            prior_end = chord["end"]

    rhythm = obj(package.get("rhythm"), "rhythm")
    beats = arr(rhythm.get("beats"), "rhythm.beats")
    sorted_records(beats, "rhythm.beats", "time", strict=True)
    for index, beat in enumerate(beats):
        path = f"rhythm.beats[{index}]"
        if not isinstance(beat, dict):
            continue
        fields(beat, path, ("time", "is_downbeat", "bar_index", "beat_in_bar"))
        time_value(beat.get("time"), path + ".time")
        if beat.get("is_downbeat") is not None and not isinstance(beat["is_downbeat"], bool):
            error(path + ".is_downbeat", "must be boolean or null")
        for name in ("bar_index", "beat_in_bar"):
            value = beat.get(name)
            if value is not None and (not integer(value) or value < (1 if name == "beat_in_bar" else 0)):
                error(path + "." + name, "must be a valid integer or null")
    tempi = arr(rhythm.get("tempo"), "rhythm.tempo")
    sorted_records(tempi, "rhythm.tempo", "start")
    for index, tempo in enumerate(tempi):
        path = f"rhythm.tempo[{index}]"
        if not isinstance(tempo, dict):
            continue
        interval(tempo, path)
        if not number(tempo.get("bpm")) or tempo["bpm"] <= 0:
            error(path + ".bpm", "must be positive and finite")
        string(tempo.get("beat_unit"), path + ".beat_unit")
    for index, meter in enumerate(arr(rhythm.get("meter"), "rhythm.meter")):
        path = f"rhythm.meter[{index}]"
        if not isinstance(meter, dict):
            error(path, "must be an object")
            continue
        interval(meter, path)
        fields(meter, path, ("numerator", "denominator"))
        if "status" in meter:
            status(meter, path)
        numerator, denominator = meter.get("numerator"), meter.get("denominator")
        if numerator is None and denominator is None:
            continue
        if not integer(numerator) or numerator <= 0:
            error(path + ".numerator", "must be a positive integer, or both meter values null")
        if not integer(denominator) or denominator <= 0 or denominator & (denominator - 1):
            error(path + ".denominator", "must be a positive power of two, or both meter values null")

    sections = arr(package.get("sections"), "sections")
    sorted_records(sections, "sections", "start")
    section_ids = set()
    prior_end = None
    for index, section in enumerate(sections):
        path = f"sections[{index}]"
        if not isinstance(section, dict):
            continue
        interval(section, path)
        identifier = section.get("id")
        string(identifier, path + ".id")
        if isinstance(identifier, str):
            if identifier in section_ids:
                error(path + ".id", "must be unique")
            section_ids.add(identifier)
        string(section.get("label"), path + ".label")
        string(section.get("source"), path + ".source")
        status(section, path)
        if number(section.get("start")) and prior_end is not None and section["start"] < prior_end - 1e-6:
            error(path, "section intervals must not overlap")
        if number(section.get("end")):
            prior_end = section["end"]
    descriptions = arr(package.get("descriptions"), "descriptions")
    sorted_records(descriptions, "descriptions", "applies_start")
    for index, description in enumerate(descriptions):
        path = f"descriptions[{index}]"
        if not isinstance(description, dict):
            continue
        interval(description, path, "analysis_start", "analysis_end")
        interval(description, path, "applies_start", "applies_end")
        fields(description, path, ("section_id",))
        identifier = description.get("section_id")
        if identifier is not None and (not isinstance(identifier, str) or identifier not in section_ids):
            error(path + ".section_id", "must reference an existing section or be null")
        if all(number(description.get(key)) for key in ("analysis_start", "analysis_end", "applies_start", "applies_end")):
            if description["analysis_start"] > description["applies_start"] or description["analysis_end"] < description["applies_end"]:
                error(path, "analysis context must contain the applicable interval")
        for name in ("genre", "instruments", "textures"):
            if name in description and description[name] is not None:
                arr(description[name], path + "." + name)
        emotion = description.get("emotion")
        if emotion is not None:
            emotion = obj(emotion, path + ".emotion")
            for axis in ("valence", "arousal"):
                value = emotion.get(axis)
                if value is not None and not number(value):
                    error(path + ".emotion." + axis, "must be finite or null")
            # Different engines expose different scales; do not silently assume 0–1.
            if any(emotion.get(axis) is not None for axis in ("valence", "arousal")) and not emotion.get("scale"):
                error(path + ".emotion.scale", "numeric emotion estimates must declare their scale")

    lyrics = obj(package.get("lyrics_context"), "lyrics_context")
    for key in ("lyrics", "themes", "narrative", "imagery", "background", "sources"):
        values = arr(lyrics.get(key), "lyrics_context." + key)
        for index, value in enumerate(values):
            if isinstance(value, dict) and ("start" in value or "end" in value):
                if value.get("start") is None and value.get("end") is None:
                    continue  # Unaligned whole-song semantics are allowed.
                interval(value, f"lyrics_context.{key}[{index}]")
    for index, warning in enumerate(arr(package.get("warnings"), "warnings")):
        if not isinstance(warning, str):
            error(f"warnings[{index}]", "must be a string")
    return errors
