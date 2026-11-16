"""BTC inference using upstream architecture/weights with sample-based timestamps.

Runs in its own process: upstream imports generic ``utils`` and uses NumPy's
removed float alias. No vendor source is modified and no MIDI side output needed.
"""
import argparse
import json
from pathlib import Path
import sys


def infer(repo, audio, output):
    import librosa
    import numpy as np
    import torch
    import yaml

    repo = Path(repo).resolve()
    sys.path.insert(0, str(repo))
    # Compatibility confined to this short-lived upstream-model process.
    if "float" not in np.__dict__:
        np.float = float
    from btc_model import BTC_model
    from utils.mir_eval_modules import idx2voca_chord

    config = yaml.safe_load((repo / "run_config.yaml").read_text())
    config["model"]["num_chords"] = 170
    torch.set_num_threads(2)
    model = BTC_model(config["model"]).cpu().eval()
    checkpoint = torch.load(repo / "test/btc_model_large_voca.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    sr, hop = int(config["mp3"]["song_hz"]), int(config["feature"]["hop_length"])
    y, _ = librosa.load(audio, sr=sr, mono=True)
    duration = len(y) / sr
    block_samples = round(config["mp3"]["inst_len"] * sr)
    features, times = [], []
    # Keep the upstream ten-second CQT context, but map every frame back to
    # its true chunk offset instead of accumulating the rounded 10/108 rate.
    for begin in range(0, len(y), block_samples):
        clip = y[begin:begin + block_samples]
        if len(clip) < 4096:
            clip = np.pad(clip, (0, 4096 - len(clip)))
        cqt = librosa.cqt(clip, sr=sr, n_bins=config["feature"]["n_bins"],
                          bins_per_octave=config["feature"]["bins_per_octave"], hop_length=hop)
        t = begin / sr + np.arange(cqt.shape[1]) * hop / sr
        valid = t < min(duration, (begin + block_samples) / sr)
        features.append(np.log(np.abs(cqt[:, valid]) + 1e-6).T)
        times.extend(t[valid].tolist())
    matrix = np.concatenate(features)
    matrix = (matrix - checkpoint["mean"]) / checkpoint["std"]
    steps = int(config["model"]["timestep"])
    padded = np.pad(matrix, ((0, (-len(matrix)) % steps), (0, 0)))
    predictions = []
    with torch.inference_mode():
        for begin in range(0, len(padded), steps):
            data = torch.as_tensor(padded[begin:begin + steps], dtype=torch.float32)[None]
            hidden, _ = model.self_attn_layers(data)
            labels, _ = model.output_layer(hidden)
            predictions.extend(labels.flatten().tolist())
    predictions = predictions[:len(times)]
    names = idx2voca_chord()
    segments = []
    first = 0
    for end in range(1, len(predictions) + 1):
        if end < len(predictions) and predictions[end] == predictions[first]:
            continue
        segments.append({"start": times[first], "end": times[end] if end < len(times) else duration,
                         "label": names[predictions[first]]})
        first = end
    Path(output).write_text(json.dumps(segments, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    parser.add_argument("audio")
    parser.add_argument("output")
    args = parser.parse_args()
    infer(args.repo, args.audio, args.output)
