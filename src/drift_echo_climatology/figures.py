"""Generate the main and supporting figures from the study catalog."""

import argparse
import os
from pathlib import Path
import sys


def main(argv=None, root=None):
    root = Path.cwd() if root is None else Path(root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=root / "data" / "catalog.csv")
    parser.add_argument("--prepared-dir", type=Path, default=root / "data" / "processed" / "GOES16")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "figures")
    args = parser.parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    from .catalog import load_catalog
    from .example_figures import plot_figure_1, plot_theory
    from .distribution_figures import plot_distributions, plot_clock_distributions

    try:
        catalog = load_catalog(args.catalog)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        paths = [plot_figure_1(args.prepared_dir, catalog, args.output_dir),
                 plot_theory(args.output_dir)]
        paths.extend(plot_distributions(catalog, args.output_dir))
        paths.extend(plot_clock_distributions(catalog, args.output_dir))
        print(f"Generated {len(paths)} paper figures from {catalog['event_id'].nunique()} "
              f"valid windows and {len(catalog)} channel rows: {args.output_dir}")
    except Exception as exc:
        print(f"Figure generation failed: {exc}", file=sys.stderr)
        return 1
    return 0
