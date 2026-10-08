# Reproduction commands

Run `python scripts/reproduce.py` from the repository root for the complete
workflow. Defaults are GOES-16 and January 1, 2017 through April 30, 2025.
`--data-dir /path/to/raw` can reuse an existing source archive.
`reproduce.py` downloads the complete study's OMNI months even when `--start`
and `--end` restrict the GOES dates, because the original IMF cleaner requires
global quartiles over the full study interval.

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

`--start` and `--end` select inclusive UTC GOES anchor dates. Download the next
GOES day too for the detector's midnight padding, and all study OMNI months for
the original global cleaner:

```bash
python scripts/download_data.py --start 2019-06-16 --end 2019-06-17 \
  --products mpsh mag ephe
python scripts/download_data.py --start 2017-01-01 --end 2025-04-30 \
  --products omni
python scripts/preprocess_data.py --start 2019-06-16 --end 2019-06-16
python scripts/detect_echoes.py --start 2019-06-16 --end 2019-06-16
python scripts/select_events.py --start 2019-06-16 --end 2019-06-16
python scripts/prepare_plot_data.py --start 2019-06-16 --end 2019-06-16
```

Use `download_data.py --list-only` to inspect source URLs without downloading.
`--products omni` downloads only OMNI. Its files cover whole months; the catalog
concatenates the full study interval and trims to the original export's timestamp
bounds, ending at April 30, 2025 00:00 UTC, before applying the original cleaner
once. A one-month OMNI download is insufficient for a date-range sample.

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
and the latest local monthly OMNI versions. It requires every month from January
2017 through April 2025, even for a short selected-event interval, and reports
missing inputs. This preserves the original cleaner's global quartiles before
computing inclusive-window component medians and clock angles.

The full figure set needs both Figure 1 example days and their valid sequences;
a catalog from a shorter interval alone cannot supply those examples. Energy
ranks and histogram scales are computed from the complete supplied valid
species catalogs before pitch-angle or clock-angle filtering. Subset checks
therefore produce subset distributions, not the complete paper climatology.
