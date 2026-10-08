# Data provenance

## Public observations

GOES-16 observations come from the [NOAA Level-2 archive](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/data/):

| Input | Archive subdirectory | Required fields |
|---|---|---|
| MPS-HI | `mpsh-l2-avg1m_science/YYYY/MM/` | `time`, `AvgDiffElectronFlux`, `AvgDiffProtonFlux`, both `Diff*EffectiveEnergy` arrays |
| Magnetometer | `magn-l2-avg1m/YYYY/MM/` | `time`, `b_epn`, `b_brf`, `orbit_llr_geo` |
| Ephemeris | `ephe-l2-orb1m/YYYY/MM/` | `time`, `geo_llr` |

The inspected MPS-HI archive contains 3,012 daily files from January 8, 2017 to
April 7, 2025: 3,002 v2-0-3 and ten v2-0-2 files. This is an availability
inventory, not a claim that every minute or calendar day has valid observations.
The downloader rediscovers current source availability rather than relying on a
fixed list. Downloaded observations and their local manifests are ignored by Git.

The MPS-HI reader preserves stored flux/energy precision and replaces negative
flux values with NaN, matching the working study reader. Effective energies
are taken from each file: they vary across product versions and calibration
changes, so fixed energy tables would change the calculation. Fluxes, energies,
and timestamps matched the original reader exactly on seven archived examples,
including v2-0-2 and both sides of the August 2023 calibration transition.

Checks against stored study results found different lowest-channel proton
energies on January 24, 2018, April 19, 2024, and February 9, 2025. The stored
values match the older CDRL79 calibration; the current files use CDRL88 values.
The reader retains file-provided energies. This input discrepancy remains for
review before reproducing the complete historical catalog.

Calibration LUTs and instrumental Poisson-error arrays are not consumed by the
current paper's detection, selection, or plots. They are omitted. CLEAN retains
the study's separate AR(1) residual-noise significance calculation.

Monthly one-minute IMF observations come from the [NASA OMNI archive](https://cdaweb.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro2_1min/).
The reader consumes `Epoch`, `BX_GSE`, `BY_GSM`, and `BZ_GSM`. It masks declared
CDF fill values, nonfinite values, and observations outside declared validity
bounds. It preserves valid disturbances removed by the old generic IQR and
fill-pattern cleaner. This is an explicit input correction; GOES fluxes,
effective energies, and pitch-angle asset values are unchanged.

Against the original full export, metadata masking retains 247 valid `BZ_GSM`
observations removed by its generic IQR filter. The other two components and
all common finite values agree exactly. Inclusive window component medians,
derived clock angles, and sample counts matched the original on all six verified
cached days. A directly downloaded August 2019 monthly CDF also matched that
month of the original export exactly.

The plot catalog contains only valid selected channel rows. It retains
overlapping windows and adds the clock angle derived after taking component
medians, plus the number of OMNI timestamps in the window. The sample count
remains zero when a component has no finite observations, as in the working
code. The original export's inclusive bounds are January 1, 2017 00:00 UTC
through April 30, 2025
00:00 UTC; later padded window portions have no IMF samples beyond that cutoff.

## Pitch-angle climatology

The package includes only the two required `(1440, 5)` pitch-angle arrays in
`src/drift_echo_climatology/assets/goes16_pitch_climatology.npz` (54,016 bytes),
loaded with `allow_pickle=False`. They match the consumed source slices exactly.
See the adjacent [asset provenance](../src/drift_echo_climatology/assets/README.md)
for the source hash and export contract.

The working detector uses this model unconditionally for January 1 through
April 11, 2017 and repeats its first daily rows into midnight padding. Later
magnetometer failures do not automatically activate this substitution. The
existing code selects the 95th percentile; R1 describes a median. This first
release preserves the coded values and records that discrepancy for a later
scientific check while the pipeline is brought up.

Publication citation, software licensing, and provider/output reuse notices
will be completed with the public-release metadata. No raw observations or
precomputed detection catalog are bundled.
