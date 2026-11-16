"""Create an original synthetic audio fixture; not a copyrighted song."""
from pathlib import Path
import argparse
import numpy as np
import soundfile as sf


def create(path):
    sr = 22050
    y = np.zeros(8 * sr, dtype=np.float32)
    for start, end, hz in ((1, 2, 261.63), (2.2, 3.2, 329.63), (3.4, 4.4, 392), (4.6, 6, 523.25)):
        n = round((end - start) * sr)
        envelope = np.minimum(1, np.arange(n) / (sr * 0.03)) * np.minimum(1, (n - 1 - np.arange(n)) / (sr * 0.03))
        y[round(start * sr):round(start * sr) + n] += .2 * envelope * np.sin(2 * np.pi * hz * np.arange(n) / sr)
    for time in np.arange(1, 7, .5):
        n = round(sr * .02)
        y[round(time * sr):round(time * sr) + n] += .15 * np.exp(-np.arange(n) / (sr * .003))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y, sr)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("work/demo.wav"))
    args = parser.parse_args()
    print(create(args.output))
