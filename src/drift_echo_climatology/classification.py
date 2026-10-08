"""Select and classify multi-channel sequences from daily CLEAN candidates."""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
import pandas as pd

from .acquisition import _date_argument, _sha256
from .config import (STUDY_START, STUDY_END, CLEAN_SIMULATIONS,
                     CLEAN_MAX_ITERATIONS, CLEAN_SIGNIFICANCE)
from .detection import COLUMNS, _typed, load_candidates
from .selection import CATEGORIES, SELECTED_COLUMNS, select_candidates


CANDIDATE_NAME = re.compile(r"g16_d(\d{8})\.csv")


def load_detection(path, allow_partial=False):
    """Require a checked candidate/provenance pair before selection."""
    path = Path(path)
    record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if record.get("schema_version") != 1 or record.get("candidate_sha256") != _sha256(path):
        raise ValueError("Candidate schema or hash is invalid; regenerate CLEAN outputs")
    settings = record.get("parameters", {})
    complete = (record.get("complete_day") is True
                and settings.get("simulations") == CLEAN_SIMULATIONS
                and settings.get("max_iterations") == CLEAN_MAX_ITERATIONS
                and settings.get("significance") == CLEAN_SIGNIFICANCE
                and set(settings.get("species", [])) == {"fedu", "fpdu"}
                and all(settings.get(name) is None for name in ("telescopes", "channels", "starts")))
    if not complete and not allow_partial:
        raise ValueError("Partial CLEAN input; use --allow-partial and a separate --output-dir for checks")
    for field in ("max_echo_cand_per", "expected_drift_per", "est_drift_per_a0_adj", "est_drift_per_a90_adj"):
        if record.get("units", {}).get(field) != "minutes":
            raise ValueError("Candidate period units must be minutes: " + field)
    frame = load_candidates(path)
    match = CANDIDATE_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError("Unsupported candidate filename")
    day = pd.Timestamp(match[1])
    if (frame["date"].isna().any() or not frame["date"].eq(day).all()
            or frame["date1_utc_echo"].isna().any() or frame["date2_utc_echo"].isna().any()
            or not frame["species"].isin(("fedu", "fpdu")).all()
            or not frame["satellite"].eq("goes16").all()):
        raise ValueError("Candidate identities or UTC dates do not match their daily file")
    return frame, record, complete


def load_selected(path):
    """Read selected sequences, including a typed empty daily table."""
    frame = pd.read_csv(path, float_precision="round_trip", dtype={"category": str, "event_id": str})
    if list(frame.columns) != SELECTED_COLUMNS:
        raise ValueError("Selected columns do not match the study table schema")
    result = _typed(frame[COLUMNS].copy())
    if not frame["category"].isin(CATEGORIES).all() or frame["event_id"].isna().any():
        raise ValueError("Selected category or window identity is invalid")
    lengths = pd.to_numeric(frame["sequence_length"], errors="raise")
    if not np.isfinite(lengths).all() or np.any(lengths < 4) or np.any(lengths % 1 != 0):
        raise ValueError("Selected sequences must contain at least four components")
    result["category"] = frame["category"].astype(str)
    result["event_id"] = frame["event_id"].astype(str)
    result["sequence_length"] = lengths.astype("int64")
    for _, group in result.groupby("event_id", sort=False):
        if (group["category"].nunique() != 1 or group["sequence_length"].nunique() != 1
                or int(group["sequence_length"].iloc[0]) != len(group)):
            raise ValueError("Selected window rows or sequence length disagree")
    return result


def save_selected(path, frame, metadata):
    """Stage and validate both outputs; the JSON hash marks completion."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = []
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name + ".",
                                         suffix=".part", mode="w", encoding="utf-8", newline="", delete=False) as stream:
            temporary = Path(stream.name)
            staged.append(temporary)
            frame.to_csv(stream, index=False, date_format="%Y-%m-%dT%H:%M:%S")
            stream.flush()
            os.fsync(stream.fileno())
        load_selected(temporary)
        record = dict(metadata, schema_version=1, selected_sha256=_sha256(temporary))
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name + ".",
                                         suffix=".part", mode="w", encoding="utf-8", delete=False) as stream:
            manifest = Path(stream.name)
            staged.append(manifest)
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.replace(manifest, path.with_suffix(".json"))
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)


def _identity(path, allow_partial):
    digest = hashlib.sha256()
    for name in ("config.py", "detection.py", "selection.py", "classification.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return {"candidate_filename": path.name, "candidate_sha256": _sha256(path),
            "candidate_metadata_sha256": _sha256(path.with_suffix(".json")),
            "implementation_sha256": digest.hexdigest(), "allow_partial": allow_partial,
            "dependencies": {name: version(name) for name in ("numpy", "pandas")}}


def main(argv=None, root=None):
    root = Path.cwd() if root is None else Path(root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=_date_argument, default=STUDY_START)
    parser.add_argument("--end", type=_date_argument, default=STUDY_END)
    parser.add_argument("--candidate-dir", type=Path, default=root / "data" / "detected" / "GOES16")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-partial", action="store_true", help="select bounded test inputs in a separate directory")
    parser.add_argument("--force", action="store_true", help="recompute and replace existing selected days")
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("--start must be on or before --end")
    if args.allow_partial and args.output_dir is None:
        parser.error("--allow-partial requires an explicit --output-dir")
    output_dir = args.output_dir or root / "data" / "selected" / "GOES16"
    if output_dir.resolve() == args.candidate_dir.resolve():
        parser.error("Selection output must be separate from CLEAN candidates")
    paths = []
    for path in sorted(args.candidate_dir.glob("*.csv")):
        match = CANDIDATE_NAME.fullmatch(path.name)
        if match and args.start <= pd.Timestamp(match[1]).date() <= args.end:
            paths.append(path)
    if not paths:
        print("No candidate days in the requested interval; run CLEAN first.", file=sys.stderr)
        return 1
    failures = 0
    for path in paths:
        destination = output_dir / path.name
        try:
            candidates, source, complete = load_detection(path, args.allow_partial)
            identity = _identity(path, args.allow_partial)
            manifest = destination.with_suffix(".json")
            if (destination.exists() or manifest.exists()) and not args.force:
                previous = json.loads(manifest.read_text(encoding="utf-8"))
                if (any(previous.get(key) != value for key, value in identity.items())
                        or previous.get("selected_sha256") != _sha256(destination)):
                    raise ValueError("Selected outputs or their input/settings changed; use --force to recompute")
                load_selected(destination)
                print("Keep validated selected day: " + path.stem)
                continue
            selected, stats = select_candidates(candidates)
            metadata = dict(identity, complete_day=complete, statistics=stats,
                            units=source["units"], amplitude_units=source.get("amplitude_units", {}),
                            detection_provenance=source,
                            event_definition="one selected sequence per species/day/telescope/UTC window; overlapping windows remain separate")
            save_selected(destination, selected, metadata)
            windows = selected["event_id"].nunique()
            print(f"Selected {path.stem}: {windows} windows, {len(selected)} components", flush=True)
        except Exception as exc:
            failures += 1
            print(f"FAILED {path.name}: {exc}", file=sys.stderr)
    print(f"Selected or reused {len(paths) - failures}/{len(paths)} days; failures {failures}.", file=sys.stderr)
    return 1 if failures else 0
