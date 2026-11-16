"""Optional supplied text, timed lyrics, and SOFA forced alignment.

No text input means no inferred lyrics or song biography. Extractive annotations
refer to actual source lines; they are not a claim about an author's intent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import unicodedata
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


_SOLFEGE_PHONES = {"do": ["d", "o"], "re": ["r", "ei"], "mi": ["m", "i"],
                   "fa": ["f", "a"], "so": ["s", "o"], "sol": ["s", "o"],
                   "la": ["l", "a"], "si": ["s", "i"], "ti": ["t", "i"]}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sofa_tokens(lyrics, dictionary, cfg):
    """Preserve every lexical unit; unknown words are errors, never dropped."""
    lexicon = {}
    for row in Path(dictionary).read_text(encoding="utf-8-sig").splitlines():
        if row.strip():
            token, phones = row.split("\t", 1)
            lexicon[token.strip()] = phones.split()
    supported = {phone for phones in lexicon.values() for phone in phones}
    overrides = {key.lower(): value for key, value in cfg.get("token_pronunciations", {}).items()}
    if cfg.get("solfege", False):
        overrides = dict(_SOLFEGE_PHONES, **overrides)
    all_units, approximations = [], []
    for item in lyrics:
        text = item["text"]
        units = []
        if cfg.get("text_type", "mandarin") == "pinyin":
            units = [{"text": token, "token": token, "pronunciation_source": "supplied_pinyin"}
                     for token in text.split()]
        else:
            if cfg.get("text_type", "mandarin") != "mandarin":
                raise ValueError("SOFA text_type must be mandarin or pinyin")
            for match in re.finditer(r"[\u3400-\u9fff]+|[A-Za-z]+|.", text, re.DOTALL):
                value = match.group()
                if re.fullmatch(r"[\u3400-\u9fff]+", value):
                    from pypinyin import lazy_pinyin, Style
                    syllables = lazy_pinyin(value, style=Style.NORMAL, v_to_u=False, errors="default")
                    if len(syllables) != len(value):
                        raise ValueError("G2P output does not preserve input characters")
                    units.extend({"text": char, "token": token, "pronunciation_source": "pypinyin"}
                                 for char, token in zip(value, syllables))
                elif re.fullmatch(r"[A-Za-z]+", value):
                    phones = overrides.get(value.lower())
                    if not isinstance(phones, list) or not phones or any(p not in supported for p in phones):
                        raise ValueError(f"Unsupported word {value!r}; explicit model-supported token_pronunciations required")
                    token = "latin_" + value.lower()
                    lexicon[token] = phones
                    units.append({"text": value, "token": token, "pronunciation_source": "configured_approximation"})
                    approximations.append({"original": value, "token": token, "phonemes": phones,
                                           "status": "approximation", "source": "solfege_profile_or_custom_lexicon"})
                elif value.isspace() or unicodedata.category(value)[0] in {"P", "Z"}:
                    continue
                else:
                    raise ValueError(f"Unsupported lyric character {value!r}; refusing to discard it")
        if not units:
            raise ValueError("SOFA cannot align a lyric line with no lexical units")
        for unit in units:
            if unit["token"] not in lexicon:
                raise ValueError(f"Syllable {unit['token']!r} is missing from the SOFA dictionary")
            unit["phonemes"] = lexicon[unit["token"]]
        all_units.append({"original_text": text, "units": units})
    unique = {entry["token"]: entry for entry in approximations}
    return all_units, lexicon, list(unique.values())


def _read_sofa_words(words_file, duration):
    words = []
    for raw in Path(words_file).read_text(encoding="utf-8").splitlines():
        parts = raw.split(maxsplit=2)
        if len(parts) != 3:
            raise ValueError("invalid SOFA HTK word output")
        start, end = int(parts[0]) / 1e7, int(parts[1]) / 1e7
        nonlexical = parts[2] in {"SP", "AP", "BND", "<SP>", "<AP>", "sil", "pau"}
        if not 0 <= start <= end <= duration + 0.02 or (not nonlexical and start == end):
            raise ValueError("SOFA alignment outside audio timeline")
        if not nonlexical:
            words.append({"start": start, "end": min(end, duration), "token": parts[2]})
    if any(a["end"] > b["start"] + 0.02 for a, b in zip(words, words[1:])):
        raise ValueError("SOFA word alignment is not monotonic")
    return words


def _sofa_align(ctx, lyrics, cfg):
    """Run the official SOFA modules in their isolated environment on full audio."""
    from .descriptions import _read_clip
    import soundfile as sf

    cfg = dict(cfg)
    repo = Path(cfg["repo"]).expanduser().resolve()
    checkpoint = Path(cfg["checkpoint"]).expanduser().resolve()
    dictionary = Path(cfg.get("dictionary", repo / "dictionary/opencpop-extension.txt")).expanduser().resolve()
    if not (repo / "infer.py").is_file() or not checkpoint.is_file() or not dictionary.is_file():
        raise FileNotFoundError("SOFA repo, checkpoint or dictionary missing")
    cfg.update(repo=str(repo), checkpoint=str(checkpoint), dictionary=str(dictionary))
    work = Path(ctx["work_dir"]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="sofa-", dir=work))
    separation = ctx.get("dependencies", {}).get("separation", {})
    vocal = separation.get("stems", {}).get("vocals")
    if isinstance(vocal, dict):
        vocal = vocal.get("path")
    audio, sr = _read_clip(vocal or ctx["audio_path"], 0, float(ctx["duration"]), 44100)
    sf.write(directory / "lyrics.wav", audio, sr, subtype="PCM_16")
    request = directory / "request.json"
    request.write_text(json.dumps({"config": cfg, "lyrics": lyrics, "duration": ctx["duration"]}, ensure_ascii=False), encoding="utf-8")
    log_path = directory / "worker.log"
    python = cfg.get("python", sys.executable)
    if "/" in python or "\\" in python:
        python = str(Path(python).expanduser().absolute())  # Preserve virtualenv symlink.
    args = [str(python), str(Path(__file__).resolve()), "--sofa-worker", "--request", str(request),
            "--folder", str(directory)]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(cfg.get("threads", 2))
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(args, cwd=directory, env=env, stdout=log, stderr=log, check=True,
                       timeout=float(cfg.get("timeout_seconds", 1800)))
    tokenized = json.loads((directory / "tokenized.json").read_text(encoding="utf-8"))
    lines = tokenized["lines"]
    if [line["original_text"] for line in lines] != [item["text"] for item in lyrics]:
        raise ValueError("SOFA tokenized text does not match supplied lyrics")
    words = _read_sofa_words(directory / "htk/words/lyrics.lab", float(ctx["duration"]))
    expected = [unit for line in lines for unit in line["units"]]
    if [word["token"] for word in words] != [unit["token"] for unit in expected]:
        raise ValueError("SOFA output words do not match supplied syllables; refusing fabricated line alignment")
    for word, unit in zip(words, expected):
        word.update(unit)
    cursor = 0
    result = []
    for item, line in zip(lyrics, lines):
        span = words[cursor:cursor + len(line["units"])]
        if not span:
            raise ValueError("SOFA returned an empty lyric alignment")
        result.append(dict(item, start=span[0]["start"], end=span[-1]["end"], words=span,
                           source="sofa_reference_lyrics", timing_status="estimated", end_is_boundary_estimate=False))
        cursor += len(line["units"])
    model_meta = json.loads((directory / "model_meta.json").read_text(encoding="utf-8"))
    return result, dict(model_meta, engine="SOFA", checkpoint=str(checkpoint),
                       checkpoint_sha256=_sha256(checkpoint), dictionary_sha256=_sha256(dictionary),
                       effective_dictionary_sha256=_sha256(directory / "effective_dictionary.txt") if (directory / "effective_dictionary.txt").is_file() else None,
                       log_path=str(log_path), g2p=cfg.get("text_type", "mandarin"),
                       pronunciation_approximations=tokenized["approximations"],
                       analysis_start=0.0, analysis_end=float(ctx["duration"]),
                       alignment_mode="whole_audio_force", reference_timing_used=False,
                       input_audio="vocals" if vocal else "mix")


def _sofa_worker(request_path, directory):
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    cfg = request["config"]
    folder = Path(directory).resolve()
    lines, lexicon, approximations = _sofa_tokens(request["lyrics"], cfg["dictionary"], cfg)
    (folder / "tokenized.json").write_text(json.dumps({"lines": lines, "approximations": approximations}, ensure_ascii=False), encoding="utf-8")
    dictionary = folder / "effective_dictionary.txt"
    dictionary.write_text("\n".join(token + "\t" + " ".join(phones) for token, phones in lexicon.items()), encoding="utf-8")
    # Explicit phrase boundaries are passed as SP tokens supported by current SOFA.
    text = " SP ".join(" ".join(unit["token"] for unit in line["units"]) for line in lines)
    (folder / "lyrics.lab").write_text(text, encoding="utf-8")
    sys.path.insert(0, cfg["repo"])
    import torch
    import lightning as pl
    from modules.g2p import DictionaryG2P
    from modules.task.forced_alignment import LitForcedAlignmentTask
    from modules.AP_detector import LoudnessSpectralcentroidAPDetector
    from modules.utils.post_processing import post_processing
    from modules.utils.export_tool import Exporter
    torch.set_num_threads(int(cfg.get("threads", 2)))
    torch.set_grad_enabled(False)
    g2p = DictionaryG2P(dictionary=str(dictionary))
    g2p.set_in_format("lab")
    dataset = g2p.get_dataset([folder / "lyrics.wav"])
    start = time.monotonic()
    model = LitForcedAlignmentTask.load_from_checkpoint(cfg["checkpoint"], map_location="cpu")
    required_phones = {phone for line in lines for unit in line["units"] for phone in unit["phonemes"]}
    if required_phones - set(model.vocab):
        raise ValueError("Configured pronunciations contain phonemes unsupported by the checkpoint")
    model.set_inference_mode("force")
    trainer = pl.Trainer(logger=False, accelerator="cpu", devices=1, enable_progress_bar=False,
                         enable_checkpointing=False)
    predictions = trainer.predict(model, dataloaders=dataset, return_predictions=True)
    predictions = LoudnessSpectralcentroidAPDetector().process(predictions)
    predictions, errors = post_processing(predictions)
    if errors or len(predictions) != 1:
        raise ValueError(f"SOFA post-processing failed: {errors}")
    Exporter(predictions, errors).export(["htk", "textgrid", "confidence"])
    confidence = float(predictions[0][2])
    if not math.isfinite(confidence):
        raise ValueError("SOFA returned non-finite confidence")
    try:
        revision = subprocess.check_output(["git", "-C", cfg["repo"], "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    (folder / "model_meta.json").write_text(json.dumps({"source_revision": revision,
        "model_id": cfg.get("model_id", "qiuqiao/SOFA:" + Path(cfg["checkpoint"]).stem), "device": "cpu",
        "threads": int(cfg.get("threads", 2)), "versions": {"torch": torch.__version__, "lightning": pl.__version__},
        "elapsed_seconds": time.monotonic() - start, "alignment_confidence": confidence,
        "confidence_interpretation": "upstream alignment score, not calibrated correctness probability",
        "word_count": sum(len(line["units"]) for line in lines)}, ensure_ascii=False), encoding="utf-8")


def _source_record(path, kind, metadata_path=None):
    record = {"kind": kind, "path": path, "text_sha256": _sha256(path),
              "source_type": "supplied_text", "verification_status": "content_unverified"}
    if metadata_path:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        if metadata.get("text_sha256") != record["text_sha256"]:
            raise ValueError("source_metadata text_sha256 does not match supplied lyrics")
        url = metadata.get("source_url")
        if not isinstance(url, str) or not re.match(r"^https?://[^\s]+$", url):
            raise ValueError("source_metadata requires an HTTP(S) source_url")
        record.update(source_url=url, source_type=metadata.get("source_type", "external_reference_text"),
                      verification_status="text_hash_verified_content_unverified",
                      metadata_path=str(Path(metadata_path).resolve()))
    return record


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
            source = _source_record(path, kind, cfg.get("source_metadata") if kind == "lyrics" else None)
            data["sources"].append(source)
            if kind == "lyrics":
                provenance["text_source"] = source
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
            provenance["engine"] = "SOFA+supplied_text_extractive"
            provenance["model_id"] = provenance["alignment"].get("model_id")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sofa-worker", action="store_true", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--folder", required=True)
    arguments = parser.parse_args()
    _sofa_worker(arguments.request, arguments.folder)
