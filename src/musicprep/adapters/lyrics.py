"""Optional supplied text, timed lyrics, and SOFA forced alignment.

No text input means no inferred lyrics or song biography. Extractive annotations
refer to actual source lines; they are not a claim about an author's intent.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile


_TIME = re.compile(r"\[(\d{1,3}):([0-5]?\d(?:\.\d{1,3})?)\]")
_META = re.compile(r"^\[(?:ar|ti|al|by|re|ve|length|offset):.*\]$", re.IGNORECASE)
_RULES = {
    "themes": {"recollection": ["回忆", "记得", "从前", "曾经", "remember", "memory"],
               "parting": ["离开", "离别", "分开", "再见", "goodbye", "farewell"],
               "affection": ["喜欢", "爱情", "爱你", "love"]},
    "imagery": {"rain": ["雨", "rain"], "wind": ["风", "wind"], "sky": ["天空", "晴天", "sky"],
                "sunlight": ["阳光", "太阳", "sunlight"], "night": ["夜", "night"],
                "sea": ["海", "ocean", "sea"], "road": ["路", "road"]},
}


def _read_text(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"text input does not exist: {path}")
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("text input exceeds 2 MiB limit")
    return path.read_text(encoding="utf-8-sig"), str(path)


def _parse_lyrics(text, duration):
    warnings, entries = [], []
    boundaries = set()
    offsets = re.findall(r"\[offset:([+-]?\d+)\]", text, flags=re.IGNORECASE)
    offset = int(offsets[-1]) / 1000 if offsets else 0
    timed = bool(_TIME.search(text))
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or _META.match(line):
            continue
        stamps = list(_TIME.finditer(line))
        if timed:
            if not stamps:
                warnings.append(f"lyrics line {line_number} ignored: timed file line lacks a valid timestamp")
                continue
            content = _TIME.sub("", line).strip()
            for stamp in stamps:
                # Common LRC convention: positive offset displays lyrics sooner.
                start = int(stamp.group(1)) * 60 + float(stamp.group(2)) - offset
                if not 0 <= start < duration:
                    warnings.append(f"lyrics line {line_number} timestamp outside audio; ignored")
                    continue
                # Empty timed lines clear the displayed lyric and delimit the
                # preceding line, even though they are not themselves lyrics.
                boundaries.add(start)
                if not content:
                    continue
                entries.append({"text": content, "start": start, "end": None,
                                "source_line": line_number, "source": "supplied_lrc",
                                "timing_status": "supplied", "end_is_boundary_estimate": True})
        else:
            if line.startswith("[") and re.match(r"\[\d", line):
                warnings.append(f"lyrics line {line_number} ignored: malformed timestamp")
                continue
            entries.append({"text": line, "start": None, "end": None, "source_line": line_number,
                            "source": "supplied_text", "timing_status": "unavailable"})
    if timed:
        entries.sort(key=lambda item: (item["start"], item["source_line"]))
        deduped, seen = [], set()
        for item in entries:
            # Same chorus at different times is essential information, not a duplicate.
            key = (item["start"], item["text"])
            if key not in seen:
                deduped.append(item)
                seen.add(key)
        entries = deduped
        starts = sorted(boundaries)
        next_start = dict(zip(starts, starts[1:] + [duration]))
        for item in entries:
            item["end"] = next_start[item["start"]]
    for index, item in enumerate(entries):
        item["id"] = f"lyric_{index:04d}"
    return entries, warnings


def _extract_annotations(lyrics):
    output = {"themes": [], "imagery": [], "narrative": []}
    for field, labels in _RULES.items():
        for label, terms in labels.items():
            evidence = []
            for item in lyrics:
                lower = item["text"].lower()
                matches = [term for term in terms if (re.search(r"\b" + re.escape(term) + r"\b", lower)
                           if term.isascii() else term in lower)]
                if matches:
                    evidence.append({"lyric_id": item["id"], "matched_terms": matches,
                                     "start": item["start"], "end": item["end"]})
            if evidence:
                output[field].append({"label": label, "status": "candidate", "source": "lexical_rule",
                                      "evidence": evidence})
    # An ordered extract is transparent; an invented story summary would not be.
    for item in lyrics:
        output["narrative"].append({"text": item["text"], "lyric_id": item["id"],
                                    "start": item["start"], "end": item["end"],
                                    "source": "extractive", "status": "source_excerpt"})
    return output


def _parse_background(text, path):
    warnings = []
    if Path(path).suffix.lower() == ".json":
        data = json.loads(text)
        rows = data.get("background", data.get("items")) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise ValueError("background JSON must be an array or an object with background/items array")
    else:
        rows = [{"text": line.strip()} for line in text.splitlines() if line.strip()]
    output = []
    for index, row in enumerate(rows):
        if isinstance(row, str):
            row = {"text": row}
        if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip():
            warnings.append(f"background item {index} ignored: nonempty text required")
            continue
        record = {"text": row["text"].strip(), "source": "supplied_background",
                  "source_file": path, "source_item": index, "verification_status": "unverified"}
        url = row.get("url", row.get("source_url"))
        if url is not None:
            if isinstance(url, str) and re.match(r"^https?://[^\s]+$", url):
                record["source_url"] = url
            else:
                warnings.append(f"background item {index}: invalid source URL omitted")
        output.append(record)
    return output, warnings


def _sofa_align(ctx, lyrics, cfg):
    """Use SOFA's documented infer.py + HTK word export, then regroup by line."""
    from .descriptions import _read_clip
    import soundfile as sf

    repo = Path(cfg["repo"]).expanduser().resolve()
    checkpoint = Path(cfg["checkpoint"]).expanduser().resolve()
    dictionary = Path(cfg.get("dictionary", repo / "dictionary/opencpop-extension.txt")).expanduser().resolve()
    if not (repo / "infer.py").is_file() or not checkpoint.is_file() or not dictionary.is_file():
        raise FileNotFoundError("SOFA repo, checkpoint or dictionary missing")
    tokens_by_line = []
    for item in lyrics:
        text = item["text"]
        if cfg.get("text_type", "mandarin") == "mandarin":
            from pypinyin import lazy_pinyin, Style
            chinese = "".join(re.findall(r"[\u3400-\u9fff]", text))
            if re.search(r"[A-Za-z0-9]", text):
                raise ValueError("Mandarin SOFA input contains non-Chinese words; provide a matching G2P/dictionary configuration")
            tokens = lazy_pinyin(chinese, style=Style.NORMAL, errors="ignore")
        elif cfg["text_type"] == "pinyin":
            tokens = text.split()
        else:
            raise ValueError("SOFA text_type must be mandarin or pinyin")
        if not tokens:
            raise ValueError("SOFA cannot align a lyric line with no supported syllables")
        tokens_by_line.append(tokens)
    work = Path(ctx["work_dir"]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="sofa-", dir=work))
    separation = ctx.get("dependencies", {}).get("separation", {})
    vocal = separation.get("stems", {}).get("vocals")
    if isinstance(vocal, dict):
        vocal = vocal.get("path")
    audio, sr = _read_clip(vocal or ctx["audio_path"], 0, float(ctx["duration"]), 44100)
    sf.write(directory / "lyrics.wav", audio, sr, subtype="PCM_16")
    (directory / "lyrics.lab").write_text(" ".join(token for line in tokens_by_line for token in line), encoding="utf-8")
    log_path = directory / "worker.log"
    args = [str(cfg.get("python", sys.executable)), str(repo / "infer.py"), "--ckpt", str(checkpoint),
            "--folder", str(directory), "--dictionary", str(dictionary), "--out_formats", "htk", "--save_confidence"]
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(args, cwd=repo, stdout=log, stderr=log, check=True,
                       timeout=float(cfg.get("timeout_seconds", 1800)))
    words_file = directory / "htk/words/lyrics.lab"
    words = []
    for raw in words_file.read_text(encoding="utf-8").splitlines():
        parts = raw.split(maxsplit=2)
        if len(parts) != 3:
            raise ValueError("invalid SOFA HTK word output")
        start, end = int(parts[0]) / 1e7, int(parts[1]) / 1e7
        if not 0 <= start < end <= float(ctx["duration"]) + 0.02:
            raise ValueError("SOFA alignment outside audio timeline")
        if parts[2] not in {"SP", "AP", "<SP>", "<AP>", "sil", "pau"}:
            words.append({"start": start, "end": min(end, float(ctx["duration"])), "text": parts[2]})
    expected = [token for line in tokens_by_line for token in line]
    if [word["text"] for word in words] != expected:
        raise ValueError("SOFA output words do not match supplied syllables; refusing fabricated line alignment")
    if any(a["end"] > b["start"] + 0.02 for a, b in zip(words, words[1:])):
        raise ValueError("SOFA word alignment is not monotonic")
    cursor = 0
    result = []
    for item, tokens in zip(lyrics, tokens_by_line):
        span = words[cursor:cursor + len(tokens)]
        result.append(dict(item, start=span[0]["start"], end=span[-1]["end"], words=span,
                           source="sofa_supplied_lyrics", timing_status="estimated", end_is_boundary_estimate=False))
        cursor += len(tokens)
    return result, {"engine": "SOFA", "checkpoint": str(checkpoint), "log_path": str(log_path),
                    "g2p": cfg.get("text_type", "mandarin"), "input_audio": "vocals" if vocal else "mix"}


def run_lyrics_context(ctx):
    duration = float(ctx["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite")
    cfg = ctx.get("config", {})
    backend = cfg.get("backend", "auto")
    if backend not in {"auto", "baseline", "extractive", "sofa"}:
        raise ValueError(f"unsupported lyrics backend: {backend}")
    data = {"lyrics": [], "themes": [], "narrative": [], "imagery": [], "background": [], "sources": []}
    warnings = []
    provenance = {"engine": "supplied_text_extractive", "model_id": None, "method_version": "1.0"}
    inputs = ctx.get("optional_inputs", {})
    if not inputs.get("lyrics") and not inputs.get("background"):
        return {"status": "skipped", "data": data, "warnings": ["No supplied lyrics or song background; semantic content is not fabricated."],
                "provenance": provenance}
    for kind in ("lyrics", "background"):
        if not inputs.get(kind):
            continue
        try:
            text, path = _read_text(inputs[kind])
            data["sources"].append({"kind": kind, "path": path, "verification_status": "user_supplied"})
            if kind == "lyrics":
                data["lyrics"], messages = _parse_lyrics(text, duration)
            else:
                data["background"], messages = _parse_background(text, path)
            warnings.extend(messages)
            if not data[kind]:
                warnings.append(f"{kind} input contains no usable entries")
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            if not cfg.get("fallback", True):
                raise
            warnings.append(f"{kind} input rejected: {exc}")
    sofa = cfg.get("sofa", cfg.get("model_backends", {}).get("sofa"))
    if data["lyrics"] and (backend == "sofa" or (backend == "auto" and sofa and any(x["start"] is None for x in data["lyrics"]))):
        try:
            if not sofa:
                raise ValueError("SOFA configuration is missing")
            data["lyrics"], provenance["alignment"] = _sofa_align(ctx, data["lyrics"], sofa)
        except (OSError, ValueError, ImportError, subprocess.SubprocessError) as exc:
            if not cfg.get("fallback", True):
                raise
            warnings.append(f"SOFA alignment unavailable/failed: {type(exc).__name__}: {exc}")
    data.update(_extract_annotations(data["lyrics"]))
    if any(item["start"] is None for item in data["lyrics"]):
        warnings.append("Plain lyrics have no timing; no timing was guessed. Configure SOFA or provide LRC for alignment.")
    if data["themes"] or data["imagery"]:
        warnings.append("Theme/imagery labels are lexical candidates with evidence, not inferred author intent.")
    # 'Succeeded' means supplied content was processed; background facts remain explicitly unverified.
    incomplete = any(item["start"] is None for item in data["lyrics"]) or any(
        token in message for message in warnings for token in ("rejected", "ignored", "failed", "no usable", "missing"))
    status = "partial" if incomplete or not (data["lyrics"] or data["background"]) else "succeeded"
    return {"status": status, "data": data, "warnings": warnings, "provenance": provenance}
