"""Prepare daily flux, pitch-angle and MLT arrays for CLEAN detection."""

import argparse
from datetime import datetime, timedelta
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy.signal import butter, filtfilt

from .acquisition import PATTERNS, _date_argument, _sha256
from .config import (STUDY_START, STUDY_END, PADDING_MINUTES, FILTER_ORDER,
                     FILTER_MULTIPLIER, LSHELL, REST_ENERGY_KEV, DRIFT_CONSTANT_SECONDS)
from .geometry import read_geometry
from .io import ASSET_DIR, read_mpsh, load_pitch_climatology, uses_pitch_climatology


SCHEMA_VERSION = 1
SPECIES = ("fedu", "fpdu")


def index_product(data_dir, product):
    """Select one highest supported numeric version per UTC day."""
    latest = {}
    directory = Path(data_dir) / "GOES16" / product
    for path in sorted(directory.glob("*.nc")):
        match = PATTERNS[product].fullmatch(path.name)
        if match is None:
            continue
        day = datetime.strptime(match[1], "%Y%m%d").date()
        product_version = tuple(int(part) for part in match.groups()[1:])
        if product == "mpsh" and product_version[0] != 2:
            continue
        previous = latest.get(day)
        if previous is None or (product_version, path.name) > (previous[0], previous[1].name):
            latest[day] = (product_version, path)
    return {day: record[1] for day, record in latest.items()}


def high_pass(values, cutoff_minutes):
    """Preserve the study's fifth-order, zero-filled Butterworth filtering."""
    missing = np.isnan(values)
    nyquist = 0.5 * (1 / 60)
    normalized_cutoff = (1 / (cutoff_minutes * 60)) / nyquist
    b, a = butter(FILTER_ORDER, normalized_cutoff, btype="high")
    result = filtfilt(b, a, np.where(missing, 0, values))
    result[missing] = np.nan
    return result


def _align(timestamps, values, grid):
    indices = np.searchsorted(grid, timestamps)
    valid = indices < grid.size
    if not np.all(valid) or not np.array_equal(grid[indices], timestamps):
        raise ValueError("Source timestamps do not fit the requested UTC minute grid")
    result = np.full((grid.size,) + values.shape[1:], np.nan, dtype=values.dtype)
    result[indices] = values
    return result


def _geometry(mag_path, ephe_path):
    if mag_path is not None:
        try:
            return read_geometry(mag_path, "mag"), "mag"
        except Exception as exc:
            print(f"MAG unavailable: {Path(mag_path).name}: {exc}", file=sys.stderr)
    if ephe_path is not None:
        return read_geometry(ephe_path, "ephe"), "ephe"
    raise ValueError("No usable MAG or ephemeris input; acquire those products for this day")


def prepare_day(mpsh_path, next_mpsh_path=None, mag_path=None, next_mag_path=None,
                ephe_path=None, next_ephe_path=None):
    """Prepare Day N and up to eight UTC hours of the next calendar day.

    Day N's effective energies define the channel calculations, as in the study.
    Commissioning-period pitch substitution is selected by the anchor day.
    Window interpolation remains part of the later CLEAN detection stage.
    """
    source = read_mpsh(mpsh_path)
    day = datetime.strptime(source["source"]["date"], "%Y-%m-%d").date()
    following = None
    if next_mpsh_path is not None:
        try:
            following = read_mpsh(next_mpsh_path)
        except Exception as exc:
            print(f"Next-day particle data unavailable; proceeding without padding: {exc}", file=sys.stderr)
    if following is not None and following["source"]["date"] != (day + timedelta(days=1)).isoformat():
        raise ValueError("Padding must come from the next calendar day")
    midnight = (datetime.combine(day, datetime.min.time()) - datetime(2000, 1, 1, 12)).total_seconds()
    grid = midnight + np.arange(1440 + (PADDING_MINUTES if following is not None else 0)) * 60.0
    geometry, method = _geometry(mag_path, ephe_path)
    if not np.array_equal(geometry["j2000"], midnight + np.arange(1440) * 60):
        raise ValueError("Primary geometry must match the particle-data UTC day")
    result = {"j2000": grid, "schema_version": np.array(SCHEMA_VERSION),
              "date": np.array(day.isoformat()), "padding_minutes": np.array(grid.size - 1440),
              "mlt": _align(geometry["j2000"], geometry["mlt"], grid)}
    for species in SPECIES:
        result["pitch_" + species] = _align(geometry["j2000"], geometry["pitch_" + species], grid)
    if following is not None:
        try:
            next_geometry, next_method = _geometry(next_mag_path, next_ephe_path)
            if not np.array_equal(next_geometry["j2000"], midnight + (1440 + np.arange(1440)) * 60):
                raise ValueError("Padding geometry must match the next particle-data UTC day")
        except Exception as exc:
            print(f"Padding has no usable geometry: {day + timedelta(days=1)}: {exc}", file=sys.stderr)
            next_method = "missing"
        else:
            selected = next_geometry["j2000"] < midnight + (1440 + PADDING_MINUTES) * 60
            positions = np.searchsorted(grid, next_geometry["j2000"][selected])
            result["mlt"][positions] = next_geometry["mlt"][selected]
            for species in SPECIES:
                result["pitch_" + species][positions] = next_geometry["pitch_" + species][selected]
        method += "/" + next_method
    if uses_pitch_climatology(day):
        climate = load_pitch_climatology()
        for species in SPECIES:
            result["pitch_" + species] = climate[species][np.arange(grid.size) % 1440]
        result["pitch_source"] = np.array("commissioning_climatology")
    else:
        result["pitch_source"] = np.array(method)
    result["geometry_source"] = np.array(method)

    for species in SPECIES:
        values = _align(source["j2000"], source[species], grid)
        if following is not None:
            if following[species].shape[1:] != values.shape[1:]:
                raise ValueError("Channel layout changes between padded days for " + species)
            selected = following["j2000"] < midnight + grid.size * 60
            positions = np.searchsorted(grid, following["j2000"][selected])
            values[positions] = following[species][selected]
        energies = source[species + "_energies"]
        normalized = energies.astype(float) / REST_ENERGY_KEV[species]
        drift0 = DRIFT_CONSTANT_SECONDS[species] / LSHELL * ((normalized + 1) / (normalized * (normalized + 2)))
        cutoff = np.ceil(FILTER_MULTIPLIER * (drift0 / 60) / 10) * 10
        filtered = np.empty(values.shape, dtype=float)
        for telescope, channel in np.ndindex(energies.shape):
            filtered[:, telescope, channel] = high_pass(values[:, telescope, channel], cutoff[telescope, channel])
        result[species] = values
        result[species + "_hp"] = filtered
        result[species + "_energies"] = energies
        result[species + "_cutoff_minutes"] = cutoff
        result[species + "_drift_seconds_0"] = drift0
        result[species + "_drift_seconds_90"] = drift0 * (1 - 0.3333)
        result[species + "_units"] = np.array(source[species + "_units"])
    return result


def load_prepared(path):
    """Read a validated prepared day without pickle or object arrays."""
    with np.load(path, allow_pickle=False) as archive:
        result = {name: np.array(archive[name], copy=True) for name in archive.files}
    if int(result["schema_version"]) != SCHEMA_VERSION:
        raise ValueError("Unsupported prepared-data schema")
    grid = result["j2000"]
    if grid.ndim != 1 or grid.size not in (1440, 1440 + PADDING_MINUTES) or np.any(np.diff(grid) != 60):
        raise ValueError("Prepared day must use a complete UTC minute grid")
    if result["mlt"].shape != grid.shape:
        raise ValueError("Prepared MLT grid does not match timestamps")
    day = datetime.strptime(str(result["date"]), "%Y-%m-%d")
    midnight = (day - datetime(2000, 1, 1, 12)).total_seconds()
    if grid[0] != midnight or int(result["padding_minutes"]) != grid.size - 1440:
        raise ValueError("Prepared date/padding metadata does not match timestamps")
    for species in SPECIES:
        flux = result[species]
        if (flux.ndim != 3 or flux.shape[0] != grid.size or flux.shape[1] != 5
                or result[species + "_hp"].shape != flux.shape
                or result[species + "_energies"].shape != flux.shape[1:]
                or result["pitch_" + species].shape != (grid.size, 5)):
            raise ValueError("Inconsistent prepared arrays for " + species)
        for suffix in ("_cutoff_minutes", "_drift_seconds_0", "_drift_seconds_90"):
            values = result[species + suffix]
            if values.shape != flux.shape[1:] or not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError("Invalid prepared channel parameters: " + species + suffix)
    return result


def save_prepared(path, data):
    """Atomically save numerical arrays and text metadata in a compressed NPZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name + ".",
                                         suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            np.savez_compressed(output, **data)
            output.flush()
            os.fsync(output.fileno())
        load_prepared(temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _provenance(inputs):
    sources = []
    for name, path in sorted(inputs.items()):
        if path is not None:
            sources.append({"role": name, "filename": Path(path).name, "sha256": _sha256(Path(path))})
    anchor_day = datetime.strptime(PATTERNS["mpsh"].fullmatch(Path(inputs["mpsh_path"]).name)[1], "%Y%m%d").date()
    if uses_pitch_climatology(anchor_day):
        asset = ASSET_DIR / "goes16_pitch_climatology.npz"
        sources.append({"role": "pitch_climatology", "filename": asset.name, "sha256": _sha256(asset)})
    code = hashlib.sha256()
    for name in ("config.py", "io.py", "geometry.py", "preprocessing.py"):
        code.update(Path(__file__).with_name(name).read_bytes())
    return json.dumps({"sources": sources, "implementation_sha256": code.hexdigest(),
                       "dependencies": {name: version(name) for name in
                                        ("numpy", "scipy", "pandas", "netCDF4", "spacepy")}}, sort_keys=True)


def main(argv=None, root=None):
    root = Path.cwd() if root is None else Path(root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=_date_argument, default=STUDY_START)
    parser.add_argument("--end", type=_date_argument, default=STUDY_END)
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "processed" / "GOES16")
    parser.add_argument("--force", action="store_true", help="recompute and replace existing prepared days")
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("--start must be on or before --end")
    products = {name: index_product(args.data_dir, name) for name in ("mpsh", "mag", "ephe")}
    days = sorted(day for day in products["mpsh"] if args.start <= day <= args.end)
    if not days:
        print("No supported MPS-HI files in the requested interval; run the downloader first.", file=sys.stderr)
        return 1
    os.environ.setdefault("SPACEPY", str(root / ".cache" / "spacepy"))
    failures = 0
    for day in days:
        following = day + timedelta(days=1)
        inputs = {"mpsh_path": products["mpsh"][day], "next_mpsh_path": products["mpsh"].get(following),
                  "mag_path": products["mag"].get(day), "ephe_path": products["ephe"].get(day)}
        inputs["next_mag_path"] = products["mag"].get(following) if inputs["next_mpsh_path"] else None
        inputs["next_ephe_path"] = products["ephe"].get(following) if inputs["next_mpsh_path"] else None
        destination = args.output_dir / f"g16_d{day:%Y%m%d}.npz"
        try:
            provenance = _provenance(inputs)
            if destination.exists() and not args.force:
                saved = load_prepared(destination)
                if str(saved.get("provenance_json", "")) != provenance:
                    raise ValueError("Prepared inputs or code changed; use --force to recompute")
                print(f"Keep validated prepared day: {day}")
                continue
            prepared = prepare_day(**inputs)
            prepared["provenance_json"] = np.array(provenance)
            save_prepared(destination, prepared)
            print(f"Prepared {day}: {prepared['j2000'].size} minutes, geometry {str(prepared['geometry_source'])}")
        except Exception as exc:
            failures += 1
            print(f"FAILED {day}: {exc}", file=sys.stderr)
    print(f"Prepared or reused {len(days) - failures}/{len(days)} days; failures {failures}.", file=sys.stderr)
    return 1 if failures else 0
