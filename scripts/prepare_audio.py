#!/usr/bin/env python3
"""Compatibility entry point; implementation lives in musicprep.audio."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from musicprep.audio import PreparationError, prepare_audio, sha256_file, main

if __name__ == "__main__":
    main()
