# Reproduction commands

Run `python scripts/reproduce.py` from the repository root for the complete
workflow. Defaults are GOES-16 and January 1, 2017 through April 30, 2025.
`--data-dir /path/to/raw` can reuse an existing source archive.

## Paths

```text
data/raw/GOES16/{mpsh,mag,ephe}/   NOAA daily files
data/raw/omni/                   NASA monthly CDF files
data/processed/GOES16/            prepared numerical NPZ files
data/detected/GOES16/             daily CLEAN candidate CSV/JSON pairs
data/selected/GOES16/             daily selected sequence CSV/JSON pairs
data/catalog.csv                 valid channel rows and OMNI clock angles
outputs/figures/                 eight main and twenty supporting figures
```

All default paths are anchored to the repository containing the scripts.
Source files and generated tables are ignored by Git. The 28 paper figures under
`outputs/figures/` are included in the repository.

## Bounded checks

`--start` and `--end` select inclusive UTC anchor dates. Download the next day
too for the detector's midnight padding:

```bash
python scripts/download_data.py --start 2019-06-16 --end 2019-06-17
python scripts/preprocess_data.py --start 2019-06-16 --end 2019-06-16
python scripts/detect_echoes.py --start 2019-06-16 --end 2019-06-16
python scripts/select_events.py --start 2019-06-16 --end 2019-06-16
python scripts/prepare_plot_data.py --start 2019-06-16 --end 2019-06-16
```

Use `download_data.py --list-only` to inspect source URLs without downloading.
`--products omni` downloads only OMNI. Its files cover whole months; the catalog
uses the original export's timestamp bounds, ending at April 30, 2025 00:00 UTC.

Preprocessing accepts `--data-dir` and `--output-dir`; CLEAN accepts
`--prepared-dir` and `--output-dir`; selection accepts `--candidate-dir` and
`--output-dir`. Existing outputs with changed inputs or settings require
`--force` on the relevant stage.

For a single CLEAN window at the full scientific settings:

```bash
python scripts/detect_echoes.py --start 2019-06-16 --end 2019-06-16 \
  --species fedu --telescopes 2 --channels 1 6 --window-start 14 \
  --output-dir outputs/clean-check
```

Selectors are one-based; window starts are UTC hours. Restricted or reduced
CLEAN runs require a separate output directory and are marked partial. To use
them downstream, selection and plot-data preparation require `--allow-partial`
and an explicit separate output location.

## Figures

```bash
python scripts/prepare_plot_data.py --selected-dir /path/to/selected \
  --omni-dir /path/to/omni --output data/catalog.csv
python scripts/make_figures.py --catalog data/catalog.csv \
  --prepared-dir data/processed/GOES16 --output-dir outputs/figures
```

Plot-data preparation rebuilds the small catalog from checked selected tables
and the latest local monthly OMNI versions. It requires every month needed by
the selected windows and reports missing inputs.

The full figure set needs both Figure 1 example days and their valid sequences;
a catalog from a shorter interval alone cannot supply those examples. Energy
ranks and histogram scales are computed from the complete supplied valid
species catalogs before pitch-angle or clock-angle filtering. Subset checks
therefore produce subset distributions, not the complete paper climatology.
