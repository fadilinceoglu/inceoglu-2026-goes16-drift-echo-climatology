# Drift Echoes at Geostationary Orbit: An Eight-Year Climatology from GOES-16

This repository contains the calculations and figure-generation code for
Inceoglu, Rodriguez, and Kress (2026). The analysis downloads GOES-16 particle,
magnetometer, and ephemeris observations, preprocesses the measurements, detects
drift echoes with CLEAN, selects consistent multi-channel sequences, and
reproduces the main and supporting figures.

Paper DOI:

## Results in scope

- `outputs/figures/Figure_1.png`: representative electron and proton fluxes,
  first-iteration spectra, and selected energy–period sequences.
- `outputs/figures/Figure_2.png`: theoretical electron and proton drift periods.
- `outputs/figures/Figure_3.png` and `Figure_4.png`: period–MLT distributions
  conditioned on pitch angle and energy rank.
- `outputs/figures/Figure_5.png`–`Figure_8.png` and
  `Figure_SM_1_clock_000.png`–`Figure_SM_20_clock_345.png`: distributions
  conditioned on the IMF clock angle in 15° bins.
- `data/catalog.csv`: valid selected channel detections with window-based
  OMNI clock angles.

The study interval is January 1, 2017 through April 30, 2025. Source data are
retrieved from [NOAA NCEI](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/data/)
and [NASA OMNI](https://cdaweb.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro2_1min/).
The 28 paper figures are included under `outputs/figures/`. Downloaded observations
and intermediate data remain local and are ignored by Git.

## Environment

Use Python 3.9 with the pinned dependencies. From the repository root:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
```

## Reproduce

Run the complete source-to-figure workflow with:

```bash
python scripts/reproduce.py
```

Full-interval CLEAN is computationally intensive. Existing acquisition,
preprocessing, detection, and selection outputs are reused when their inputs and
settings match. The stages can also be run separately:

```bash
python scripts/download_data.py
python scripts/preprocess_data.py
python scripts/detect_echoes.py
python scripts/select_events.py
python scripts/prepare_plot_data.py
python scripts/make_figures.py
```

With existing selected CSV/JSON tables, prepared NPZ files, and the full study's
monthly OMNI inputs, the last two commands regenerate the figures without
repeating CLEAN. The original OMNI cleaner computes global quartiles, so all
study months are required even for a shorter GOES date range.
Figure 1 requires the prepared June 16 and November 23, 2019 observations and
their valid Telescope 2 sequences. All paths default to this repository.
See [reproduction commands](docs/REPRODUCTION.md) for date limits and local paths.

## Calculation conventions

- CLEAN uses eight-hour electron and four-hour proton windows with 75% overlap,
  5,000 AR(1) noise simulations, and at most 50 iterations per window.
- Multi-channel selection requires at least four decreasing periods across
  available neighboring energies below 1,050 keV. The catalog counts detection
  windows; overlapping windows remain separate.
- IMF clock angles are derived from median magnetic-field components over each
  inclusive UTC detection window after applying the original OMNI cleaner,
  including its global 10-IQR rule for `BZ_GSM`.

The original Monte Carlo seeds were not recorded. The implementation uses
reproducible per-window seeds with a default master seed of 2026; detections near
the significance threshold can differ from the historical catalog. Input versions
and retained scientific conventions are described in
[data provenance](docs/DATA_PROVENANCE.md).

## Repository contents

```text
src/drift_echo_climatology/   scientific calculations and plotting
  assets/                    required pitch-angle climatology
scripts/                     complete workflow and individual stages
data/                        downloaded observations and generated tables, ignored
outputs/figures/             eight main and twenty supporting figures, included
docs/                        reproduction and data provenance
tests/                       focused scientific and execution checks
```

Run the checks with:

```bash
python -m unittest discover -s tests -v
```

## Citation

Inceoglu, F., Rodriguez, J. V., and Kress, B. T. (2026).
*Drift Echoes at Geostationary Orbit: An Eight-Year Climatology from GOES-16*.

Please cite the paper and this repository when using these calculations.

## License

The analysis code and original documentation are licensed under the
[MIT License](LICENSE).
