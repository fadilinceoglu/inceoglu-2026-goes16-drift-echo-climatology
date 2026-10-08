"""Run the paper workflow from public observations to figures."""

import argparse
from datetime import timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drift_echo_climatology.acquisition import _date_argument
from drift_echo_climatology.config import STUDY_START, STUDY_END


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=_date_argument, default=STUDY_START)
    parser.add_argument("--end", type=_date_argument, default=STUDY_END)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("--start must be on or before --end")
    if args.start < STUDY_START or args.end > STUDY_END:
        parser.error("Choose dates within the paper's study interval")

    from drift_echo_climatology import acquisition, preprocessing, detection, classification, catalog, figures

    interval = ["--start", args.start.isoformat(), "--end", args.end.isoformat()]
    stages = [
        ("Download GOES observations", acquisition.main,
         ["--start", args.start.isoformat(), "--end", (args.end + timedelta(days=1)).isoformat(),
          "--data-dir", str(args.data_dir), "--products", "mpsh", "mag", "ephe"]),
        ("Download OMNI", acquisition.main,
         ["--start", STUDY_START.isoformat(), "--end", STUDY_END.isoformat(),
          "--data-dir", str(args.data_dir), "--products", "omni"]),
        ("Preprocess", preprocessing.main, interval + ["--data-dir", str(args.data_dir)]),
        ("CLEAN", detection.main, interval),
        ("Select events", classification.main, interval),
        ("Prepare plot data", catalog.main, interval + ["--omni-dir", str(args.data_dir / "omni")]),
        ("Make figures", figures.main, []),
    ]
    for name, run, arguments in stages:
        print(name, flush=True)
        status = run(arguments, root=ROOT)
        if status:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
