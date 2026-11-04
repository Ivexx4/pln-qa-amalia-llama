"""Generate model answers from the configured test dataset."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import main

if __name__ == "__main__":
    raise SystemExit(main(["responses", *sys.argv[1:]]))
