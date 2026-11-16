"""Qwen2-Audio's real quantized MLX weights, with audited audio-only descriptions.

This module has no heavy imports at import time. Run the predictor in the
dedicated MLX environment; the main pipeline's Torch environment is unchanged.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import re
from pathlib import Path
import time
from types import MethodType

MODEL_ID = "mlx-community/Qwen2-Audio-7B-Instruct-4bit"
MODEL_REVISION = "c65570002626f41b4dc08b7b54f42f99f3e82e7f"
MODEL_SHA256 = "0967cde270ad62aa4824f0bbce283d0a2a6da2825ccf587640753c527d4174da"
MLX_AUDIO_REVISION = "b5fcbe964b9e4cad4b06782f97d3e51fe74a8702"
PROMPT = (
    'Describe only the audible music. Output a JSON object using this schema: '
    '{"genre": ["genre name"], "instruments": ["instrument or voice"], '
    '"textures": ["timbre or texture"]}. '
    'Use at most 2 genres, 6 instruments or voices, and 4 textures. '
    'Use short English labels and empty arrays when uncertain. '
    'Use ordinary double quotes without backslashes. Output no other text. '
    'Do not identify the song or artist. Do not include lyrics, background, '
    'emotions, mood, valence, arousal, or confidence scores.'
)


def normalize_model_json(text):
    """Undo one explicit wrapper/quote escape, never infer labels from prose."""
    if not isinstance(text, str):
        raise ValueError("Qwen output must be text")
    candidate = text.strip()
    normalization = []
    if candidate.startswith("```json\n") and candidate.endswith("```"):
        candidate = candidate[8:-3].strip()
        normalization.append("json_fence")
    elif candidate.startswith("```\n") and candidate.endswith("```"):
        candidate = candidate[4:-3].strip()
        normalization.append("json_fence")
    if candidate.startswith('"') and candidate.endswith('"'):
        decoded = json.loads(candidate)
        if not isinstance(decoded, str):
            raise ValueError("Expected a JSON string wrapper")
        candidate = decoded.strip()
        normalization.append("json_string_wrapper")
    elif candidate.startswith("{") and candidate.endswith("}"):
        # The selected model sometimes emits {\\"genre\\": ...}. Only
        # remove one quote-escape layer when every quote has that exact form.
        if r'\"' in candidate and not re.search(r'(?<!\\)"', candidate):
            if re.search(r'\\{2,}"', candidate):
                raise ValueError("Ambiguous nested quote escapes in model JSON")
            candidate = candidate.replace(r'\"', '"')
            normalization.append("unescape_json_quotes_once")
    return candidate, normalization


def parse_description(text):
    """Strict complete JSON plus bounded format normalization; no free prose."""
    candidate, _ = normalize_model_json(text)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError("Qwen output is not one complete JSON object") from exc
    if not isinstance(parsed, dict) or set(parsed) != {"genre", "instruments", "textures"}:
        raise ValueError("Qwen must return exactly one genre/instruments/textures JSON object")
    clean = {}
    limits = {"genre": 2, "instruments": 6, "textures": 4}
    for key, limit in limits.items():
        values = parsed[key]
        if not isinstance(values, list) or len(values) > limit:
            raise ValueError(f"Qwen {key} must be an array with at most {limit} labels")
        labels = []
        for value in values:
            if not isinstance(value, str) or not value.strip() or len(value.strip()) > 80:
                raise ValueError(f"Qwen {key} contains an invalid label")
            value = value.strip()
            if value.casefold() not in {label.casefold() for label in labels}:
                labels.append(value)
        clean[key] = labels
    return clean


def audio_token_lengths(feature_frames):
    """Qwen2-Audio's strided convolution and avg-pooling length equations."""
    if feature_frames <= 0 or feature_frames > 3000:
        raise ValueError("Audio feature length must be in [1,3000]")
    encoded = (int(feature_frames) - 1) // 2 + 1
    pooled = (encoded - 2) // 2 + 1
    if pooled < 1:
        raise ValueError("Audio is too short for Qwen2-Audio")
    return encoded, pooled


def _read_audio(path):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    with sf.SoundFile(path) as stream:
        if stream.frames < 1 or stream.frames > 30 * stream.samplerate:
            raise ValueError("Qwen2-Audio accepts nonempty clips of at most 30 seconds; split first")
        sample_rate = stream.samplerate
        audio = stream.read(dtype="float32", always_2d=True).mean(axis=1)
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite samples")
    duration = len(audio) / sample_rate
    if sample_rate != 16000:
        divisor = math.gcd(sample_rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, sample_rate // divisor).astype(np.float32)
    return audio, duration


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patch_official_audio_frontend(model):
    """Correct the selected MLX port's frontend to HF Qwen2-Audio semantics.

    Reuses all original encoder/projector weights. The reference Whisper
    extractor supplies Slaney mel features; effective lengths determine audio
    tokens and padding-key masks. Projected features stay floating point (a
    quantized embedding's packed uint32 weight is not a feature dtype).
    """
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    extractor = model._processor.feature_extractor

    def extract(this, audio):
        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(waveform) > 480000 or not len(waveform):
            raise ValueError("Expected 1–480000 mono 16 kHz audio samples")
        values = extractor(waveform, sampling_rate=16000, padding="max_length",
                           return_attention_mask=True, return_tensors="np")
        frames = int(values["attention_mask"].sum())
        this._musicprep_valid_encoder_frames, tokens = audio_token_lengths(frames)
        features = mx.array(values["input_features"], dtype=this.audio_tower.conv1.weight.dtype)
        return features, tokens

    def encode(this, features):
        encoder = this.audio_tower
        x = nn.gelu(encoder.conv1(features.transpose(0, 2, 1)))
        x = nn.gelu(encoder.conv2(x))
        x = x + encoder.embed_positions[:x.shape[1]]
        valid = this._musicprep_valid_encoder_frames
        mask = (mx.arange(x.shape[1]) < valid)[None, None, None, :]
        for layer in encoder.layers:
            residual = x
            normalized = layer.self_attn_layer_norm(x)
            attention = layer.self_attn
            batch, length, _ = normalized.shape
            shape = (batch, length, attention.num_heads, attention.head_dim)
            q = attention.q_proj(normalized).reshape(shape).transpose(0, 2, 1, 3)
            k = attention.k_proj(normalized).reshape(shape).transpose(0, 2, 1, 3)
            v = attention.v_proj(normalized).reshape(shape).transpose(0, 2, 1, 3)
            attended = mx.fast.scaled_dot_product_attention(q, k, v, scale=attention.scale, mask=mask)
            attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, attention.embed_dim)
            x = residual + attention.out_proj(attended)
            x = x + layer.fc2(nn.gelu(layer.fc1(layer.final_layer_norm(x))))
            # Materialize per encoder layer to bound lazy intermediate memory.
            mx.eval(x)
        batch, length, channels = x.shape
        x = x[:, :length // 2 * 2].reshape(batch, length // 2, 2, channels).mean(axis=2)
        x = encoder.layer_norm(x)
        return this.multi_modal_projector(x)

    model._extract_features = MethodType(extract, model)
    model.get_audio_features = MethodType(encode, model)


def make_predictor(cfg, root=None):
    """Return callable(path) -> three JSON label arrays, with provenance/audit."""
    if cfg.get("revision", MODEL_REVISION) != MODEL_REVISION:
        raise ValueError("This Qwen adapter only supports its pinned model revision")
    if cfg.get("expected_sha256", MODEL_SHA256) != MODEL_SHA256:
        raise ValueError("This Qwen adapter only supports its pinned weight checksum")
    import mlx.core as mx
    import numpy as np
    from mlx_audio.stt.utils import load_model

    distribution = importlib.metadata.distribution("mlx-audio")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    source_revision = direct_url.get("vcs_info", {}).get("commit_id")
    if source_revision != MLX_AUDIO_REVISION:
        raise RuntimeError("Use scripts/setup_qwen.py to install the pinned MLX Audio source required by this frontend")
    model_path = Path(cfg.get("model_id", cfg.get("model_path", ""))).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError("Qwen MLX requires a local model directory prepared by model setup")
    model_path = model_path.resolve()
    model_config = json.loads((model_path / "config.json").read_text())
    quantization = model_config.get("quantization", {})
    if model_config.get("model_type") != "qwen2_audio" or quantization.get("bits") != 4:
        raise ValueError("Expected real Qwen2-Audio 4-bit MLX model configuration")
    weights = model_path / "weights.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Missing Qwen2-Audio weights: {weights}")
    weight_hash = _sha256(weights)
    if weight_hash != MODEL_SHA256:
        raise ValueError("Qwen2-Audio weight checksum does not match the pinned model")
    # Keep Metal allocator cache bounded on a shared 16 GB workstation.
    mx.set_cache_limit(int(cfg.get("cache_limit_mb", 256)) * 1024 * 1024)
    mx.random.seed(int(cfg.get("seed", 0)))
    model = load_model(str(model_path))
    patch_official_audio_frontend(model)
    audit_root = Path(root or Path.cwd()).resolve() / "qwen-audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    provenance = {
        "engine": "qwen2_audio_mlx", "model_id": MODEL_ID,
        "device": str(mx.default_device()),
        "base_model": "Qwen/Qwen2-Audio-7B-Instruct",
        "model_revision": MODEL_REVISION,
        "model_sha256": weight_hash, "quantization": quantization,
        "audio_encoder_quantized": False, "projector_quantized": False,
        "mlx_audio_revision": source_revision,
        "versions": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-audio", "transformers")},
        "audio_frontend": "WhisperFeatureExtractor+effective_length+padding_key_mask+float_projector",
        "temperature": 0.0, "max_new_tokens": int(cfg.get("max_new_tokens", 256)),
        "prompt": PROMPT, "labels_are": "unverified_candidates", "audit_directory": str(audit_root),
    }

    def predict(path):
        path = Path(path).resolve()
        audio, duration = _read_audio(path)
        key = _sha256(path)
        audit = {"audio_sha256": key, "audio_path": str(path), "duration": duration,
                 "model_revision": provenance["model_revision"], "attempts": []}
        started = time.monotonic()
        try:
            if float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) < float(cfg.get("silence_rms", 1e-5)):
                result = {"genre": [], "instruments": [], "textures": []}
                audit["status"] = "silent_audio"
            else:
                attempts = max(1, min(2, int(cfg.get("json_attempts", 2))))
                last_error = None
                for attempt in range(attempts):
                    prompt = PROMPT if attempt == 0 else PROMPT + ' JSON schema: {"genre":[],"instruments":[],"textures":[]}. No additional keys.'
                    generated = model.generate(audio, prompt=prompt, temperature=0.0,
                                               max_tokens=provenance["max_new_tokens"],
                                               prefill_step_size=int(cfg.get("prefill_step_size", 256)),
                                               verbose=False)
                    raw = generated.text
                    entry = {"raw_output": raw, "prompt": prompt,
                             "generation_tokens": getattr(generated, "generation_tokens", None)}
                    audit["attempts"].append(entry)
                    try:
                        _, entry["normalization"] = normalize_model_json(raw)
                        result = parse_description(raw)
                        audit["status"] = "generated"
                        break
                    except ValueError as exc:
                        last_error = exc
                        entry["parse_error"] = str(exc)
                else:
                    raise ValueError(f"Qwen produced invalid structured output: {last_error}")
            audit["result"] = result
            return result
        except Exception as exc:
            audit.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            audit["elapsed_seconds"] = time.monotonic() - started
            audit["mlx_peak_memory_bytes"] = mx.get_peak_memory()
            (audit_root / f"{key}.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
            predict.audit = audit
            mx.clear_cache()

    predict.provenance = provenance
    predict.audit = None
    return predict
