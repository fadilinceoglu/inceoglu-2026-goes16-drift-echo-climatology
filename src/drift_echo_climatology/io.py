"""Read the Level-2 observations and compact assets used by this paper."""

from datetime import date, datetime, timedelta
from pathlib import Path
import re

import numpy as np
from netCDF4 import Dataset


ASSET_DIR = Path(__file__).resolve().parent / "assets"
MPSH_NAME = re.compile(
    r"^sci_mpsh-l2-avg1m_g16_d(?P<day>\d{8})_v(?P<version>\d+-\d+-\d+)\.nc$"
)
MPSH_VARIABLES = {
    "j2000": "time",
    "fedu": "AvgDiffElectronFlux",
    "fpdu": "AvgDiffProtonFlux",
    "fedu_energies": "DiffElectronEffectiveEnergy",
    "fpdu_energies": "DiffProtonEffectiveEnergy",
}
TIME_UNITS = "seconds since 2000-01-01 12:00:00 UTC"


def read_mpsh(path):
    """Read a GOES-16 one-minute v2 product without recalibrating its fluxes.

    Fluxes and effective energies are retained at their stored precision.
    Negative fluxes, including the product's fill values, become NaN as in
    the original study reader. Instrumental Poisson errors are not needed
    by this paper's CLEAN significance test or figures.
    """
    path = Path(path)
    match = MPSH_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError("Expected a GOES-16 one-minute MPS-HI filename: " + path.name)
    version = tuple(int(part) for part in match["version"].split("-"))
    if version[0] != 2:
        raise ValueError("This paper reader supports Level-2 v2 inputs only: " + path.name)
    day = datetime.strptime(match["day"], "%Y%m%d").date()

    with Dataset(path) as dataset:
        dataset.set_auto_mask(False)
        missing = set(MPSH_VARIABLES.values()) - set(dataset.variables)
        if missing:
            raise ValueError("Missing MPS-HI variables: " + ", ".join(sorted(missing)))
        if getattr(dataset["time"], "units", None) != TIME_UNITS:
            raise ValueError("Expected MPS-HI timestamps in " + TIME_UNITS)
        result = {key: np.array(dataset[name][:], copy=True)
                  for key, name in MPSH_VARIABLES.items()}
        for species in ("fedu", "fpdu"):
            energy_name = MPSH_VARIABLES[species + "_energies"]
            if getattr(dataset[energy_name], "units", None) != "keV":
                raise ValueError("Expected effective energies in keV: " + energy_name)
            result[species + "_units"] = getattr(
                dataset[MPSH_VARIABLES[species]], "units", "")

    time = result["j2000"]
    if time.ndim != 1 or not time.size or not np.isfinite(time).all():
        raise ValueError("MPS-HI timestamps must be a nonempty finite one-dimensional array")
    midnight = (datetime.combine(day, datetime.min.time()) - datetime(2000, 1, 1, 12)).total_seconds()
    offsets = time - midnight
    if (np.any(np.diff(time) <= 0) or np.any(offsets < 0)
            or np.any(offsets >= 86400) or np.any(offsets % 60 != 0)):
        raise ValueError("MPS-HI timestamps must follow the filename day's UTC minute grid")
    for species in ("fedu", "fpdu"):
        flux = result[species]
        energies = result[species + "_energies"]
        if (flux.ndim != 3 or flux.shape[0] != time.size
                or flux.shape[1] != 5 or energies.shape != flux.shape[1:]):
            raise ValueError("Inconsistent telescope/channel dimensions for " + species)
        if not energies.size or not np.isfinite(energies).all() or np.any(energies <= 0):
            raise ValueError("Effective energies must be finite and positive for " + species)
        flux[flux < 0] = np.nan
    result["source"] = {"filename": path.name, "date": day.isoformat(),
                        "version": match["version"]}
    return result


def j2000_to_datetime(seconds):
    """Convert NOAA seconds since 2000-01-01 12:00 UTC to naive UTC datetimes."""
    epoch = datetime(2000, 1, 1, 12)
    return np.array([epoch + timedelta(seconds=float(value)) for value in seconds])


def uses_pitch_climatology(day):
    """Preserve the study's GOES-16 commissioning-period substitution."""
    return date(2017, 1, 1) <= day < date(2017, 4, 12)


def load_pitch_climatology(path=None):
    """Load only when an early-2017 day needs the preserved 95th-percentile model."""
    path = ASSET_DIR / "goes16_pitch_climatology.npz" if path is None else Path(path)
    with np.load(path, allow_pickle=False) as asset:
        percentile = int(asset["percentile"])
        arrays = {species: np.array(asset[name], copy=True) for species, name in (
            ("fedu", "electron_pitch_deg"), ("fpdu", "proton_pitch_deg"))}
    if percentile != 95:
        raise ValueError("The preserved study climatology uses percentile 95")
    if any(array.shape != (1440, 5) or not np.isfinite(array).all()
           for array in arrays.values()):
        raise ValueError("Pitch climatology must contain two finite 1440-by-5 arrays")
    return arrays


def read_omni_imf(path):
    """Read one NASA OMNI CDF using each variable's declared validity metadata.

    Monthly source files are kept intact. The conditioning stage will combine
    them and apply the paper's time interval. No IQR filter or guessed common
    fill-value list is applied to valid IMF observations.
    """
    import cdflib
    import pandas as pd

    with cdflib.CDF(path) as cdf:
        result = {"time": cdflib.cdfepoch.to_datetime(cdf.varget("Epoch"))}
        for name in ("BX_GSE", "BY_GSM", "BZ_GSM"):
            values = np.array(cdf.varget(name), dtype=float, copy=True)
            attrs = cdf.varattsget(name)
            if values.ndim != 1 or values.size != len(result["time"]):
                raise ValueError("OMNI variable must match the Epoch grid: " + name)
            if "FILLVAL" not in attrs:
                raise ValueError("OMNI variable lacks FILLVAL metadata: " + name)
            fill = float(np.asarray(attrs["FILLVAL"]).reshape(-1)[0])
            valid = np.isfinite(values) & (values != fill)
            if "VALIDMIN" in attrs:
                valid &= values >= float(np.asarray(attrs["VALIDMIN"]).reshape(-1)[0])
            if "VALIDMAX" in attrs:
                valid &= values <= float(np.asarray(attrs["VALIDMAX"]).reshape(-1)[0])
            values[~valid] = np.nan
            result[name] = values
    return pd.DataFrame(result)
