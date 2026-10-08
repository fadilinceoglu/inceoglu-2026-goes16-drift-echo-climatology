"""Select multi-channel drift-echo sequences from CLEAN candidates."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drift_echo_climatology.classification import main

if __name__ == "__main__":
    raise SystemExit(main(root=ROOT))
