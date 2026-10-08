"""Run CLEAN on prepared days and save flat, unclassified candidate tables."""

import argparse
import hashlib
from importlib.metadata import version
import json
from numbers import Integral
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
import pandas as pd

from .acquisition import _date_argument, _sha256
from .clean import clean_window
from .config import (STUDY_START, STUDY_END, LSHELL, REST_ENERGY_KEV,
                     DRIFT_CONSTANT_SECONDS, CLEAN_SIMULATIONS, CLEAN_MAX_ITERATIONS,
                     CLEAN_SIGNIFICANCE, CLEAN_FFT_LENGTH, RANDOM_SEED, WINDOW_MINUTES)
from .preprocessing import load_prepared


COLUMNS = ["species", "date", "satellite", "telescope_no", "energy_channel_index",
           "energy_channel", "window_start_minute", "window_seed", "t1_mlt_echo",
           "t2_mlt_echo", "date1_utc_echo", "date2_utc_echo", "amp", "phase",
           "max_echo_cand_frq", "max_echo_cand_per", "expected_drift_per",
           "energy_keV_from_period", "median_pitch_angle_deg", "est_drift_per_a0",
           "est_drift_per_a90", "est_drift_per_a0_adj", "est_drift_per_a90_adj", "dec_year"]
DATE_COLUMNS = ("date", "date1_utc_echo", "date2_utc_echo")
INTEGER_COLUMNS = ("telescope_no", "energy_channel_index", "window_start_minute", "window_seed")
TEXT_COLUMNS = ("species", "satellite")
PREPARED_NAME = re.compile(r"g16_d(\d{8})\.npz")


def _typed(frame):
    if list(frame.columns) != COLUMNS:
        raise ValueError("Candidate columns do not match the study table schema")
    for name in COLUMNS:
        if name in DATE_COLUMNS:
            frame[name] = pd.to_datetime(frame[name], errors="raise")
        elif name in TEXT_COLUMNS:
            frame[name] = frame[name].astype(str)
        elif name in INTEGER_COLUMNS:
            values = pd.to_numeric(frame[name], errors="raise")
            if not np.isfinite(values).all() or np.any(values % 1 != 0):
                raise ValueError("Candidate integer field is invalid: " + name)
            frame[name] = values.astype("int64")
        else:
            frame[name] = pd.to_numeric(frame[name], errors="raise").astype(float)
    return frame


def window_starts(sample_count, species):
    """Preserve the working code's window endpoint and quarter-window step."""
    duration = WINDOW_MINUTES[species]
    step = duration // 4
    last_start = min(1440 - step, sample_count - 1 - duration)
    return np.arange(0, last_start + 1, step, dtype=int)


def window_seed(master_seed, day, species, telescope_no, energy_channel_index, start_minute):
    """Stable local MT19937 seed, independent of selected tasks and run order."""
    key = json.dumps([int(master_seed), str(day), species, int(telescope_no),
                      int(energy_channel_index), int(start_minute)], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big")


def _indices(requested, size, name):
    result = list(range(1, size + 1)) if requested is None else sorted(set(requested))
    if not result or any(isinstance(value, bool) or not isinstance(value, Integral)
                         or not 1 <= value <= size for value in result):
        raise ValueError(f"{name} must be one-based integers from 1 to {size}")
    return result


def energy_from_period(period_minutes, pitch_degrees, species):
    """Preserve the study's analytic inverse Walt expression, returning keV."""
    if not np.isfinite(pitch_degrees) or not np.isfinite(period_minutes) or period_minutes <= 0:
        return np.nan
    angle = pitch_degrees * np.pi / 180.0
    ratio = DRIFT_CONSTANT_SECONDS[species] * (1 - 0.3333 * np.sin(angle) ** 0.62)
    ratio = ratio / (period_minutes * 60.0) / LSHELL
    normalized = 0.5 * (-(2.0 - ratio) + np.sqrt((2.0 - ratio) ** 2 + 4.0 * ratio))
    return normalized * REST_ENERGY_KEV[species]


def detect_day(prepared, *, random_seed=RANDOM_SEED, simulations=CLEAN_SIMULATIONS,
               max_iterations=CLEAN_MAX_ITERATIONS, significance=CLEAN_SIGNIFICANCE,
               species=("fedu", "fpdu"), telescopes=None, channels=None, starts=None,
               progress=None):
    """Detect all components before multi-channel classification.

    Telescope/channel selectors are one-based; starts are minutes since the
    anchor UTC midnight. Filtering and task order cannot change a window's RNG.
    The existing bidirectional interpolation rule is retained inside each window.
    """
    for name, value in (("random_seed", random_seed), ("simulations", simulations),
                        ("max_iterations", max_iterations)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < (0 if name == "random_seed" else 1):
            raise ValueError(name + " must be a nonnegative seed or positive count")
    if not np.isfinite(significance) or not 0 <= significance <= 1:
        raise ValueError("significance must be between zero and one")
    selected_species = sorted(set(species))
    if not selected_species or any(name not in WINDOW_MINUTES for name in selected_species):
        raise ValueError("species must contain fedu and/or fpdu")
    day = str(prepared["date"])
    midnight = pd.Timestamp(day)
    sample_count = prepared["j2000"].size
    hours = np.arange(sample_count) / 60.0
    stats = {"windows_total": 0, "windows_processed": 0, "windows_skipped_missing_flux": 0,
             "candidates": 0}
    rows = []
    for particle in selected_species:
        duration = WINDOW_MINUTES[particle]
        selected_starts = list(window_starts(sample_count, particle))
        if starts is not None:
            requested = sorted(set(starts))
            if not requested or any(isinstance(value, bool) or not isinstance(value, Integral)
                                    or value not in selected_starts for value in requested):
                raise ValueError("Requested start minute is not an available " + particle + " window")
            selected_starts = requested
        n_telescopes, n_channels = prepared[particle + "_energies"].shape
        for telescope in _indices(telescopes, n_telescopes, "telescopes"):
            for channel in _indices(channels, n_channels, "channels"):
                t, c = telescope - 1, channel - 1
                for start in selected_starts:
                    stats["windows_total"] += 1
                    stop = start + duration
                    values = pd.Series(prepared[particle + "_hp"][start:stop, t, c]).interpolate(
                        method="linear", limit=3, limit_direction="both").to_numpy()
                    if not np.isfinite(values).all():
                        stats["windows_skipped_missing_flux"] += 1
                        continue
                    seed = window_seed(random_seed, day, particle, telescope, channel, start)
                    peaks = clean_window(values, hours[start:stop] * 3600,
                                         rng=np.random.RandomState(seed), simulations=simulations,
                                         max_iterations=max_iterations, significance=significance,
                                         fft_length=CLEAN_FFT_LENGTH)
                    stats["windows_processed"] += 1
                    pitch_values = prepared["pitch_" + particle][start:stop, t]
                    pitch_radians = np.deg2rad(pd.Series(pitch_values).median())
                    pitch = np.rad2deg(pitch_radians)
                    drift0 = prepared[particle + "_drift_seconds_0"][t, c] / 60
                    drift90 = prepared[particle + "_drift_seconds_90"][t, c] / 60
                    expected = (prepared[particle + "_drift_seconds_0"][t, c]
                                * (1 - 0.3333 * np.sin(pitch_radians) ** 0.62)) / 60
                    year_start = pd.Timestamp(midnight.year, 1, 1)
                    next_year = pd.Timestamp(midnight.year + 1, 1, 1)
                    metadata = {"species": particle, "date": midnight, "satellite": "goes16",
                                "telescope_no": telescope, "energy_channel_index": channel,
                                "energy_channel": prepared[particle + "_energies"][t, c],
                                "window_start_minute": start, "window_seed": seed,
                                "t1_mlt_echo": prepared["mlt"][start], "t2_mlt_echo": prepared["mlt"][stop - 1],
                                "date1_utc_echo": midnight + pd.Timedelta(minutes=int(start)),
                                "date2_utc_echo": midnight + pd.Timedelta(minutes=int(stop - 1)),
                                "median_pitch_angle_deg": pitch, "expected_drift_per": expected,
                                "est_drift_per_a0": drift0, "est_drift_per_a90": drift90,
                                "est_drift_per_a0_adj": 1.25 * drift0, "est_drift_per_a90_adj": drift90 / 3,
                                "dec_year": np.round(midnight.year + (midnight - year_start) / (next_year - year_start), 3)}
                    for peak in peaks.itertuples(index=False):
                        rows.append(dict(metadata, amp=peak.amp, phase=peak.phase,
                                         max_echo_cand_frq=peak.freq, max_echo_cand_per=peak.period,
                                         energy_keV_from_period=energy_from_period(peak.period, pitch, particle)))
                    if progress is not None:
                        progress(stats)
    stats["candidates"] = len(rows)
    return _typed(pd.DataFrame(rows, columns=COLUMNS)), stats


def load_candidates(path):
    """Read flat candidates with explicit units, dates and empty-table schema."""
    return _typed(pd.read_csv(path, float_precision="round_trip"))


def save_candidates(path, frame, metadata):
    """Write a validated CSV followed by its completion/hash metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = []
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name + ".",
                                         suffix=".part", mode="w", encoding="utf-8", newline="", delete=False) as output:
            temporary = Path(output.name)
            staged.append(temporary)
            frame.to_csv(output, index=False, date_format="%Y-%m-%dT%H:%M:%S")
            output.flush()
            os.fsync(output.fileno())
        load_candidates(temporary)
        record = dict(metadata, schema_version=1, candidate_sha256=_sha256(temporary),
                      units={"max_echo_cand_per": "minutes", "expected_drift_per": "minutes",
                             "est_drift_per_a0": "minutes", "est_drift_per_a90": "minutes",
                             "est_drift_per_a0_adj": "minutes", "est_drift_per_a90_adj": "minutes",
                             "energy_channel": "keV", "energy_keV_from_period": "keV",
                             "phase": "radians", "max_echo_cand_frq": "Hz",
                             "median_pitch_angle_deg": "degrees", "t1_mlt_echo": "hours",
                             "t2_mlt_echo": "hours", "date1_utc_echo": "UTC", "date2_utc_echo": "UTC"})
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name + ".",
                                         suffix=".part", mode="w", encoding="utf-8", delete=False) as output:
            manifest = Path(output.name)
            staged.append(manifest)
            json.dump(record, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        os.replace(manifest, path.with_suffix(".json"))
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)


def _identity(path, parameters):
    digest = hashlib.sha256()
    for name in ("config.py", "clean.py", "detection.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return {"prepared_filename": path.name, "prepared_sha256": _sha256(path),
            "parameters": parameters, "implementation_sha256": digest.hexdigest(),
            "dependencies": {name: version(name) for name in ("numpy", "scipy", "pandas")},
            "method": {"fft_length": CLEAN_FFT_LENGTH, "window_minutes": WINDOW_MINUTES,
                       "advance_minutes": {name: duration // 4 for name, duration in WINDOW_MINUTES.items()},
                       "hann_coherent_gain": 0.5, "peak_threshold_noise_std": 4},
            "seed_policy": "first 32 bits of SHA256 JSON [master,date,species,telescope,channel,start_minute]; MT19937"}


def main(argv=None, root=None):
    root = Path.cwd() if root is None else Path(root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=_date_argument, default=STUDY_START)
    parser.add_argument("--end", type=_date_argument, default=STUDY_END)
    parser.add_argument("--prepared-dir", type=Path, default=root / "data" / "processed" / "GOES16")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--simulations", type=int, default=CLEAN_SIMULATIONS)
    parser.add_argument("--max-iterations", type=int, default=CLEAN_MAX_ITERATIONS)
    parser.add_argument("--species", nargs="+", choices=tuple(WINDOW_MINUTES), default=list(WINDOW_MINUTES))
    parser.add_argument("--telescopes", nargs="+", type=int, choices=range(1, 6))
    parser.add_argument("--channels", nargs="+", type=int, choices=range(1, 12))
    parser.add_argument("--window-start", nargs="+", type=int, choices=range(24), help="UTC start hours for a bounded run")
    parser.add_argument("--force", action="store_true", help="recompute and replace existing candidate days")
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("--start must be on or before --end")
    if args.random_seed < 0 or args.simulations < 1 or args.max_iterations < 1:
        parser.error("seed must be nonnegative; simulation and iteration counts must be positive")
    restricted = (set(args.species) != set(WINDOW_MINUTES) or args.telescopes is not None
                  or args.channels is not None or args.window_start is not None
                  or args.simulations != CLEAN_SIMULATIONS or args.max_iterations != CLEAN_MAX_ITERATIONS)
    if restricted and args.output_dir is None:
        parser.error("Restricted runs require --output-dir so test candidates stay separate from the full study")
    output_dir = args.output_dir or root / "data" / "detected" / "GOES16"
    parameters = {"random_seed": args.random_seed, "simulations": args.simulations,
                  "max_iterations": args.max_iterations, "significance": CLEAN_SIGNIFICANCE,
                  "species": sorted(set(args.species)), "telescopes": args.telescopes,
                  "channels": args.channels,
                  "starts": None if args.window_start is None else [hour * 60 for hour in args.window_start]}
    paths = []
    for path in sorted(args.prepared_dir.glob("*.npz")):
        match = PREPARED_NAME.fullmatch(path.name)
        if match and args.start <= pd.Timestamp(match[1]).date() <= args.end:
            paths.append(path)
    if not paths:
        print("No prepared days in the requested interval; run preprocessing first.", file=sys.stderr)
        return 1
    failures = 0
    for path in paths:
        destination = output_dir / (path.stem + ".csv")
        manifest = destination.with_suffix(".json")
        try:
            identity = _identity(path, parameters)
            if (destination.exists() or manifest.exists()) and not args.force:
                record = json.loads(manifest.read_text(encoding="utf-8"))
                if (any(record.get(key) != value for key, value in identity.items())
                        or record.get("candidate_sha256") != _sha256(destination)):
                    raise ValueError("Candidates or their input/settings changed; use --force to recompute")
                load_candidates(destination)
                print(f"Keep validated candidate day: {path.stem}")
                continue
            prepared = load_prepared(path)
            if str(prepared["date"]).replace("-", "") != PREPARED_NAME.fullmatch(path.name)[1]:
                raise ValueError("Prepared filename and anchor date disagree")
            def progress(stats):
                if stats["windows_processed"] % 100 == 0:
                    print(f"{str(prepared['date'])}: CLEAN processed {stats['windows_processed']} windows", flush=True)
            candidates, stats = detect_day(prepared, progress=progress, **parameters)
            metadata = dict(identity, complete_day=not restricted, statistics=stats,
                            amplitude_units={name: str(prepared.get(name + "_units", ""))
                                             for name in parameters["species"]})
            save_candidates(destination, candidates, metadata)
            print(f"Detected {str(prepared['date'])}: {len(candidates)} candidates from {stats['windows_processed']} windows", flush=True)
        except Exception as exc:
            failures += 1
            print(f"FAILED {path.name}: {exc}", file=sys.stderr)
    print(f"Detected or reused {len(paths) - failures}/{len(paths)} days; failures {failures}.", file=sys.stderr)
    return 1 if failures else 0
