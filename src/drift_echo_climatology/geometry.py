"""GOES-16 local time and telescope pitch angles on a daily UTC grid."""

from datetime import datetime
from pathlib import Path
import re
import warnings

from netCDF4 import Dataset
import numpy as np
import pandas as pd

from .io import TIME_UNITS, j2000_to_datetime


LOOK_BETA = {"fedu": np.array([-35., 35., -70., 0., 70.]),
             "fpdu": np.array([-70., 0., 70., -35., 35.])}
ARCJET_FLAG = "potentially_degraded_due_to_arcjet_firing_qf"
PRODUCT_NAMES = {"mag": "dn_magn-l2-avg1m", "ephe": "dn_ephe-l2-orb1m"}


def pitch_angles(b_brf, species):
    """Use the study's BRF rotation and incoming telescope look vectors."""
    field = np.asarray(b_brf)
    if field.ndim != 2 or field.shape[1] != 3:
        raise ValueError("BRF magnetic field must have shape (time, 3)")
    beta = LOOK_BETA[species]
    total = np.linalg.norm(field, axis=1)
    # Rotate old BRF (Bx, By, Bz) to (-Bz, By, Bx), as in the study.
    with np.errstate(divide="ignore", invalid="ignore"):
        theta_b = np.arccos(field[:, 0] / total)
        phi_b = np.arctan2(field[:, 1], -field[:, 2])
        bx = np.sin(theta_b) * np.cos(phi_b)
        by = np.sin(theta_b) * np.sin(phi_b)
        bz = np.cos(theta_b)
        degrees_to_radians = np.pi / 180.0
        theta_v = np.pi - (90.0 - np.zeros(5)) * degrees_to_radians
        phi_v = np.pi + beta * degrees_to_radians
        result = np.empty((len(field), 5), dtype=float)
        for telescope in range(5):
            cosine = bx * (np.sin(theta_v[telescope]) * np.cos(phi_v[telescope]))
            cosine += by * (np.sin(theta_v[telescope]) * np.sin(phi_v[telescope]))
            cosine += bz * np.cos(theta_v[telescope])
            result[:, telescope] = np.arccos(cosine) / degrees_to_radians
    return result


def _values(dataset, name):
    if name not in dataset.variables:
        raise ValueError("Missing geometry variable: " + name)
    return np.ma.filled(dataset[name][:], np.nan)


def _arcjet_mask(variable):
    """Decode this named CF bit flag; other DQF bits are not a study veto."""
    meanings = getattr(variable, "flag_meanings", "").split()
    masks = getattr(variable, "flag_masks", ())
    values = getattr(variable, "flag_values", ())
    if ARCJET_FLAG not in meanings or len(masks) != len(meanings) or len(values) != len(meanings):
        raise ValueError("MAG DQF lacks the arcjet flag definition")
    position = meanings.index(ARCJET_FLAG)
    flags = np.ma.filled(variable[:], 0)
    return (flags & masks[position]) == values[position]


def _daily_orbit(orbit, times, grid):
    """Preserve the orbit IQR rule and per-column <=5-missing interpolation."""
    frame = pd.DataFrame(orbit, columns=["lat", "long", "radius"], index=times)
    first, third = frame.quantile(0.25), frame.quantile(0.75)
    spread = third - first
    frame = frame[~((frame < first - 1.5 * spread) |
                    (frame > third + 1.5 * spread)).any(axis=1)]
    frame = frame.reindex(grid)
    for column in frame:
        if frame[column].isna().sum() <= 5:
            frame[column] = frame[column].interpolate(method="time", limit_direction="both").bfill().ffill()
    return frame.to_numpy()


def _llr_to_mlt(orbit, grid):
    """Keep the original SpacePy RLL->SM backend and transformation defaults."""
    valid = np.isfinite(orbit).all(axis=1)
    result = np.full(len(grid), np.nan)
    if not valid.any():
        return result
    from spacepy.coordinates import Coords
    from spacepy.time import Ticktock

    llr = orbit[valid].copy()
    llr[:, 2] /= 6371 * 1e3  # meters to Earth radii, preserving source precision
    ticks = Ticktock(grid[valid].to_numpy(dtype="datetime64[s]").tolist(), "UTC")
    sm = Coords(llr[:, [2, 0, 1]], "RLL", "sph", ["Re", "deg", "deg"],
                ticks, use_irbem=False).convert("SM", "car").data
    x, y = sm[:, 0], sm[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = np.rad2deg(np.arctan(y / x))
    theta += 180.0 * ((x < 0) & (y > 0)) - 180.0 * ((x < 0) & (y < 0))
    mlt = 12.0 + theta * 12.0 / 180.0
    mlt[(x < 0) & (y == 0)] = 0.0
    result[valid] = mlt
    return result


def read_geometry(path, product):
    """Read MAG or ephemeris; missing BRF minutes retain missing pitch angles.

    MAG masks declared fill values and the arcjet bit, matching arcjet_flag=0
    in the study. Ephemeris provides local time with unavailable pitch angles.
    The root preprocessor applies the separately preserved PAD substitution.
    """
    if product not in PRODUCT_NAMES:
        raise ValueError("Geometry product must be 'mag' or 'ephe'")
    path = Path(path)
    prefix = r"(?:dn|sci)_" + re.escape(PRODUCT_NAMES[product].split("_", 1)[1])
    pattern = prefix + r"_g16_d(\d{8})_v(\d+[-_]\d+[-_]\d+)\.nc"
    match = re.fullmatch(pattern, path.name)
    if match is None:
        raise ValueError("Expected a GOES-16 " + product + " filename: " + path.name)
    day = datetime.strptime(match[1], "%Y%m%d")
    midnight = (day - datetime(2000, 1, 1, 12)).total_seconds()
    grid = pd.date_range(day, periods=1440, freq="min")

    with Dataset(path) as dataset:
        seconds = _values(dataset, "time")
        units = getattr(dataset["time"], "units", None)
        if units not in (TIME_UNITS, TIME_UNITS.removesuffix(" UTC")):
            raise ValueError("Geometry timestamps must use the NOAA J2000 epoch")
        offsets = seconds - midnight
        if (seconds.ndim != 1 or not seconds.size or not np.isfinite(seconds).all()
                or np.any(np.diff(seconds) <= 0) or np.any(offsets < 0)
                or np.any(offsets >= 86400) or np.any(offsets % 60 != 0)):
            raise ValueError("Geometry timestamps must follow the filename day's UTC minute grid")
        times = pd.DatetimeIndex(j2000_to_datetime(seconds))
        orbit_name = "orbit_llr_geo" if product == "mag" else "geo_llr"
        orbit = _values(dataset, orbit_name)
        orbit_units = "degrees, degrees, meters" if product == "mag" else "deg, deg, km"
        if getattr(dataset[orbit_name], "units", None) != orbit_units:
            raise ValueError("Expected " + orbit_name + " in " + orbit_units)
        if orbit.shape != (len(times), 3):
            raise ValueError("Geometry orbit must have shape (time, 3)")
        if product == "ephe":
            orbit[:, 2] *= 1e3  # ephemeris kilometers to meters
        pitches = {species: np.full((1440, 5), np.nan) for species in LOOK_BETA}
        if product == "mag":
            field = _values(dataset, "b_brf")
            if getattr(dataset["b_brf"], "units", None) != "nT":
                raise ValueError("Expected BRF magnetic field in nT")
            if field.shape != (len(times), 3):
                raise ValueError("BRF magnetic field must have shape (time, 3)")
            try:
                if "DQF" not in dataset.variables:
                    raise ValueError("MAG has no DQF variable")
                arcjet = _arcjet_mask(dataset["DQF"])
                if arcjet.shape != (len(times),):
                    raise ValueError("MAG DQF must match the time grid")
                field[arcjet] = np.nan
            except ValueError as error:
                warnings.warn(str(error) + "; arcjet masking unavailable", RuntimeWarning)
            for species in LOOK_BETA:
                frame = pd.DataFrame(pitch_angles(field, species), index=times)
                pitches[species] = frame.reindex(grid).to_numpy()

    return {"j2000": midnight + np.arange(1440, dtype=float) * 60,
            "mlt": _llr_to_mlt(_daily_orbit(orbit, times, grid), grid),
            "pitch_fedu": pitches["fedu"], "pitch_fpdu": pitches["fpdu"],
            "source": {"filename": path.name, "date": day.date().isoformat(),
                       "version": match[2], "product": product}}
