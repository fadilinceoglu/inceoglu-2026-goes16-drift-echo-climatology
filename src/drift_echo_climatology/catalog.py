"""Prepare the valid selected sequences and IMF clock angles used by the plots."""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

from .acquisition import PATTERNS, _date_argument, _sha256, month_starts
from .classification import CANDIDATE_NAME, load_selected
from .config import STUDY_START, STUDY_END
from .detection import COLUMNS, _typed
from .io import clean_omni_imf, read_omni_imf
from .selection import SELECTED_COLUMNS


CATALOG_COLUMNS = SELECTED_COLUMNS + ["clock_angle_deg_median", "imf_samples"]
IMF_COLUMNS = ["BX_GSE", "BY_GSM", "BZ_GSM"]
MEDIAN_COLUMNS = [name + "_median" for name in IMF_COLUMNS]
STUDY_FIRST = pd.Timestamp(STUDY_START)
STUDY_LAST = pd.Timestamp(STUDY_END)
PERIOD_COLUMNS = ("max_echo_cand_per", "expected_drift_per", "est_drift_per_a0_adj", "est_drift_per_a90_adj")


def _omni_times(frame):
    times = pd.to_datetime(frame["time"], errors="raise", utc=True).dt.tz_localize(None).to_numpy()
    ticks = times.astype("datetime64[ns]").astype("int64")
    if (pd.isna(times).any() or np.any(np.diff(ticks) <= 0)
            or np.any(ticks % (60 * 10 ** 9) != 0)):
        raise ValueError("OMNI timestamps must be unique, increasing UTC minutes")
    return times


def _omni_arrays(frame):
    times = _omni_times(frame)
    values = [pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=float) for name in IMF_COLUMNS]
    if any(np.isinf(array).any() for array in values):
        raise ValueError("OMNI components must contain finite values or masked NaNs")
    return times, values


def window_imf_medians(frame, omni):
    """Compute inclusive-window component medians and their IMF clock angle.

    Each unique UTC start/end pair is evaluated once. All three components
    need a finite sample independently, as in the paper preparation script.
    Sample counts include timestamps with masked components.
    """
    times, arrays = _omni_arrays(omni)
    starts = pd.to_datetime(frame["date1_utc_echo"], errors="raise", utc=True).dt.tz_localize(None)
    ends = pd.to_datetime(frame["date2_utc_echo"], errors="raise", utc=True).dt.tz_localize(None)
    if starts.isna().any() or ends.isna().any() or (starts > ends).any():
        raise ValueError("Detection windows need ordered, finite UTC endpoints")
    cache, rows = {}, []
    for start, end in zip(starts, ends):
        key = (start, end)
        if key not in cache:
            result = [np.nan, np.nan, np.nan, np.nan, 0]
            left = np.searchsorted(times, start.to_datetime64(), side="left")
            right = np.searchsorted(times, end.to_datetime64(), side="right")
            if right > left:
                slices = [array[left:right] for array in arrays]
                if all(np.isfinite(part).any() for part in slices):
                    bx, by, bz = [np.nanmedian(part) for part in slices]
                    clock = np.mod(np.degrees(np.arctan2(by, bz)), 360.0)
                    result = [bx, by, bz, clock, int(right - left)]
            cache[key] = result
        rows.append(cache[key])
    result = pd.DataFrame(rows, index=frame.index,
                          columns=MEDIAN_COLUMNS + ["clock_angle_deg_median", "imf_samples"])
    for name in MEDIAN_COLUMNS + ["clock_angle_deg_median"]:
        result[name] = result[name].astype(float)
    result["imf_samples"] = result["imf_samples"].astype("int64")
    return result


def _typed_catalog(frame):
    if list(frame) != CATALOG_COLUMNS:
        raise ValueError("Catalog columns do not match the plot table schema")
    result = _typed(frame[COLUMNS].copy())
    if (not frame["category"].eq("valid").all() or frame["event_id"].isna().any()
            or not result["satellite"].eq("goes16").all()
            or not result["species"].isin(("fedu", "fpdu")).all()
            or any(result[name].isna().any() for name in ("date", "date1_utc_echo", "date2_utc_echo"))
            or (result["date1_utc_echo"] > result["date2_utc_echo"]).any()):
        raise ValueError("Catalog must contain valid GOES-16 windows with UTC identities")
    result["category"] = frame["category"].astype(str)
    result["event_id"] = frame["event_id"].astype(str)
    for name, minimum in (("sequence_length", 4), ("imf_samples", 0)):
        values = pd.to_numeric(frame[name], errors="raise")
        if not np.isfinite(values).all() or (values < minimum).any() or (values % 1 != 0).any():
            raise ValueError("Invalid catalog count: " + name)
        result[name] = values.astype("int64")
    clock = pd.to_numeric(frame["clock_angle_deg_median"], errors="raise").astype(float)
    if (np.isinf(clock).any() or ((clock < 0) | (clock >= 360)).any()
            or (clock.notna() & result["imf_samples"].eq(0)).any()):
        raise ValueError("Catalog clock angles must be in [0, 360) degrees or NaN")
    result["clock_angle_deg_median"] = clock
    for _, group in result.groupby("event_id", sort=False):
        if (group["category"].nunique() != 1 or group["sequence_length"].nunique() != 1
                or int(group["sequence_length"].iloc[0]) != len(group)):
            raise ValueError("Catalog event rows or sequence lengths disagree")
    for _, group in result.groupby(["date1_utc_echo", "date2_utc_echo"], sort=False):
        if (group["clock_angle_deg_median"].nunique(dropna=False) != 1
                or group["imf_samples"].nunique(dropna=False) != 1):
            raise ValueError("Shared UTC windows have inconsistent IMF enrichment")
    return result.loc[:, CATALOG_COLUMNS]


def load_catalog(path):
    """Read a checked plot catalog and attach its provenance to frame.attrs."""
    path = Path(path)
    record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if record.get("schema_version") != 1 or record.get("catalog_sha256") != _sha256(path):
        raise ValueError("Plot catalog schema or hash is invalid; regenerate plot data")
    if (any(record.get("units", {}).get(name) != "minutes" for name in PERIOD_COLUMNS)
            or record.get("units", {}).get("clock_angle_deg_median") != "degrees"
            or record.get("units", {}).get("imf_samples") != "count"
            or not isinstance(record.get("complete_detection_days"), bool)):
        raise ValueError("Plot catalog units or completeness metadata is invalid")
    frame = pd.read_csv(path, float_precision="round_trip", dtype={"category": str, "event_id": str})
    result = _typed_catalog(frame)
    result.attrs["provenance"] = record
    return result


def _selected_pair(path, allow_partial):
    record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if record.get("schema_version") != 1 or record.get("selected_sha256") != _sha256(path):
        raise ValueError("Selected table schema or hash is invalid: " + path.name)
    complete = record.get("complete_day") is True
    if not complete and not allow_partial:
        raise ValueError("Partial selected input requires --allow-partial and an explicit --output")
    if any(record.get("units", {}).get(name) != "minutes" for name in PERIOD_COLUMNS):
        raise ValueError("Selected period units must be minutes: " + path.name)
    frame = load_selected(path)
    day = pd.Timestamp(CANDIDATE_NAME.fullmatch(path.name)[1])
    if frame["date"].isna().any() or not frame["date"].eq(day).all():
        raise ValueError("Selected anchor dates disagree with daily filename: " + path.name)
    source = {"filename": path.name, "sha256": _sha256(path),
              "metadata_sha256": _sha256(path.with_suffix(".json")), "complete_day": complete}
    return frame, record, source


def _needed_months(frame):
    """The original global IQR filter needs the full study even for subset plots."""
    return list(month_starts(STUDY_START, STUDY_END)) if len(frame) else []


def _read_months(directory, months):
    available = {}
    for path in directory.glob("*.cdf"):
        match = PATTERNS["omni"].fullmatch(path.name)
        if not match:
            continue
        try:
            month = _date_argument(match[1][:4] + "-" + match[1][4:6] + "-" + match[1][6:])
        except argparse.ArgumentTypeError:
            continue
        if month.day != 1:
            continue
        key = (int(match[2]), path.name)
        if month not in available or key > available[month][0]:
            available[month] = (key, path)
    missing = [month.isoformat() for month in months if month not in available]
    if missing:
        raise FileNotFoundError("Missing required OMNI months: " + ", ".join(missing))
    frames, sources = [], []
    for month in months:
        (source_version, _), path = available[month]
        frame = read_omni_imf(path)
        times = _omni_times(frame)
        next_month = pd.Timestamp(month) + pd.offsets.MonthBegin(1)
        if not len(frame) or np.any(times < np.datetime64(month)) or np.any(times >= next_month.to_datetime64()):
            raise ValueError("OMNI timestamps disagree with source month: " + path.name)
        frames.append(frame)
        sources.append({"filename": path.name, "sha256": _sha256(path), "version": source_version})
    combined = (pd.concat(frames, ignore_index=True) if frames
                else pd.DataFrame({"time": pd.Series(dtype="datetime64[ns]"),
                                   **{name: pd.Series(dtype=float) for name in IMF_COLUMNS}}))
    _omni_times(combined)
    combined = combined.loc[(combined["time"] >= STUDY_FIRST) & (combined["time"] <= STUDY_LAST)].reset_index(drop=True)
    return clean_omni_imf(combined), sources


def _save_catalog(path, frame, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = []
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name + ".", suffix=".part",
                                         mode="w", encoding="utf-8", newline="", delete=False) as stream:
            temporary = Path(stream.name)
            staged.append(temporary)
            frame.to_csv(stream, index=False, date_format="%Y-%m-%dT%H:%M:%S")
            stream.flush()
            os.fsync(stream.fileno())
        _typed_catalog(pd.read_csv(temporary, float_precision="round_trip", dtype={"category": str, "event_id": str}))
        record = dict(metadata, schema_version=1, catalog_sha256=_sha256(temporary))
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name + ".", suffix=".part",
                                         mode="w", encoding="utf-8", delete=False) as stream:
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


def main(argv=None, root=None):
    root = Path.cwd() if root is None else Path(root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=_date_argument, default=STUDY_START)
    parser.add_argument("--end", type=_date_argument, default=STUDY_END)
    parser.add_argument("--selected-dir", type=Path, default=root / "data" / "selected" / "GOES16")
    parser.add_argument("--omni-dir", type=Path, default=root / "data" / "raw" / "omni")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-partial", action="store_true", help="allow bounded selected inputs in an explicit output")
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("--start must be on or before --end")
    if args.allow_partial and args.output is None:
        parser.error("--allow-partial requires an explicit --output")
    output = args.output or root / "data" / "catalog.csv"
    if output.suffix != ".csv":
        parser.error("--output must be a CSV filename")
    paths = []
    for path in sorted(args.selected_dir.glob("*.csv")):
        match = CANDIDATE_NAME.fullmatch(path.name)
        if match and args.start <= pd.Timestamp(match[1]).date() <= args.end:
            paths.append(path)
    if not paths:
        print("No selected days in the requested interval; run selection first.", file=sys.stderr)
        return 1
    if any(output.resolve() == path.resolve() for path in paths):
        parser.error("Catalog output must be separate from selected daily inputs")
    try:
        frames, selected_sources, units, amplitude_units = [], [], None, {}
        for path in paths:
            frame, record, source = _selected_pair(path, args.allow_partial)
            frames.append(frame.loc[frame["category"] == "valid"])
            selected_sources.append(source)
            if units is not None and units != record["units"]:
                raise ValueError("Selected sources have inconsistent units")
            units = record["units"]
            for species, unit in record.get("amplitude_units", {}).items():
                if species in amplitude_units and amplitude_units[species] != unit:
                    raise ValueError("Selected sources have inconsistent amplitude units")
                amplitude_units[species] = unit
        selected = pd.concat(frames, ignore_index=True)
        omni, omni_sources = _read_months(args.omni_dir, _needed_months(selected))
        enrichment = window_imf_medians(selected, omni)
        selected["clock_angle_deg_median"] = enrichment["clock_angle_deg_median"]
        selected["imf_samples"] = enrichment["imf_samples"]
        catalog = _typed_catalog(selected.loc[:, CATALOG_COLUMNS])
        digest = hashlib.sha256()
        for name in ("catalog.py", "io.py"):
            digest.update(Path(__file__).with_name(name).read_bytes())
        metadata = {"complete_detection_days": all(source["complete_day"] for source in selected_sources),
                    "anchor_start": args.start.isoformat(), "anchor_end": args.end.isoformat(),
                    "rows": len(catalog), "events": int(catalog["event_id"].nunique()),
                    "selected_sources": selected_sources, "omni_sources": omni_sources,
                    "units": dict(units, clock_angle_deg_median="degrees", imf_samples="count"),
                    "amplitude_units": amplitude_units, "implementation_sha256": digest.hexdigest(),
                    "dependencies": {name: version(name) for name in ("numpy", "pandas", "cdflib")},
                    "omni_interval_utc": [STUDY_FIRST.isoformat(), STUDY_LAST.isoformat()],
                    "imf_masking": "original finite/common-fill rules; BX_GSE and BY_GSM physical bounds; BZ_GSM 10-IQR filter over the full study interval",
                    "clock_angle": "mod(degrees(atan2(median BY_GSM, median BZ_GSM)), 360); all three IMF components require finite samples",
                    "event_definition": "selected valid UTC windows; overlapping windows remain separate"}
        _save_catalog(output, catalog, metadata)
        print(f"Prepared {len(catalog)} components in {catalog['event_id'].nunique()} windows: {output}", flush=True)
        return 0
    except Exception as exc:
        print(f"FAILED plot catalog: {exc}", file=sys.stderr)
        return 1
