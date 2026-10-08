"""Paper distributions of valid detection windows in MLT, period and IMF angle."""

from pathlib import Path
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MLT_BIN_WIDTH = 1.0
LOG_PER_BIN_WIDTH = 0.02
MLT_TICKS = [0, 3, 6, 9, 12, 15, 18, 21, 24]
COLORMAP = "PuBu"
MLT_QUARTILES = [(3, 9, 3, 9), (9, 15, 9, 15), (15, 21, 15, 21), (21, 3, 21, 27)]
PITCH_SECTIONS = [
    ("60°-120°", [(60.0, 120.0)]),
    ("120°-150° and 30°-60°", [(120.0, 150.0), (30.0, 60.0)]),
    ("150°-180° and 0°-30°", [(150.0, 180.0), (0.0, 30.0)]),
]
CLOCK_BIN_WIDTH = 15.0
CLOCK_BIN_CENTERS = np.arange(0.0, 360.0, CLOCK_BIN_WIDTH)
PAPER_ANGLES = {60.0: 5, 150.0: 6, 240.0: 7, 330.0: 8}
PEAK_PERCENTILE = 95


def add_energy_rank(frame):
    """Rank the available effective energies within each telescope."""
    frame = frame.copy()
    frame["energy_channel"] = pd.to_numeric(frame["energy_channel"], errors="coerce")
    frame["telescope_no"] = pd.to_numeric(frame["telescope_no"], errors="coerce")
    rank_map = {}
    for telescope, group in frame.groupby("telescope_no"):
        for rank, energy in enumerate(np.sort(group["energy_channel"].dropna().unique())):
            rank_map[(float(telescope), float(energy))] = rank
    frame["energy_rank"] = [
        rank_map.get((float(telescope), float(energy)), np.nan)
        for telescope, energy in zip(frame["telescope_no"], frame["energy_channel"])
    ]
    return frame


def selected_ranks_and_labels(frame):
    available = np.sort(frame["energy_rank"].dropna().unique().astype(int))
    if available.size == 0:
        raise ValueError("No valid energy ranks found")
    ranks = [int(available[0]), int(available[len(available) // 2]), int(available[-1])]
    labels = {}
    for rank in ranks:
        energies = np.sort(frame.loc[frame["energy_rank"] == rank, "energy_channel"].dropna().unique())
        if energies.size == 0:
            labels[rank] = "No energy"
        elif np.isclose(energies.min(), energies.max()):
            labels[rank] = f"{energies.min():.0f} keV"
        else:
            labels[rank] = f"{energies.min():.0f}-{energies.max():.0f} keV"
    return ranks, labels


def log_period_edges_for_df(frame, log_per_bin_width=LOG_PER_BIN_WIDTH):
    periods = pd.to_numeric(frame["max_echo_cand_per"], errors="coerce").to_numpy()
    periods = periods[np.isfinite(periods) & (periods > 0)]
    if periods.size == 0:
        raise ValueError("No valid periods found for histogramming")
    values = np.log10(periods)
    lower = np.floor(values.min() / log_per_bin_width) * log_per_bin_width
    upper = np.ceil(values.max() / log_per_bin_width) * log_per_bin_width
    return np.arange(lower, upper + log_per_bin_width, log_per_bin_width)


def build_span_count_histogram(frame, mlt_edges, log_per_edges):
    """Give each detection weight one, shared across its wrapped MLT span."""
    start = pd.to_numeric(frame["t1_mlt_echo"], errors="coerce").to_numpy()
    end = pd.to_numeric(frame["t2_mlt_echo"], errors="coerce").to_numpy()
    periods = pd.to_numeric(frame["max_echo_cand_per"], errors="coerce").to_numpy()
    finite = np.isfinite(start) & np.isfinite(end) & np.isfinite(periods) & (periods > 0)
    start = np.mod(np.asarray(start[finite], dtype=float), 24.0)
    end = np.mod(np.asarray(end[finite], dtype=float), 24.0)
    periods = np.log10(periods[finite])
    period_indices = np.digitize(periods, log_per_edges) - 1
    valid = (period_indices >= 0) & (period_indices < len(log_per_edges) - 1)
    start, end, period_indices = start[valid], end[valid], period_indices[valid]
    centers = 0.5 * (mlt_edges[:-1] + mlt_edges[1:])
    centers_48 = np.concatenate([centers, centers + 24.0])
    histogram = np.zeros((len(log_per_edges) - 1, len(centers)), dtype=float)
    for first, last, period_index in zip(start, end, period_indices):
        last_unwrapped = first + (last - first) % 24.0
        in_span = (centers_48 >= first) & (centers_48 <= last_unwrapped)
        indices = np.unique(np.where(in_span)[0] % len(centers))
        if indices.size:
            histogram[period_index, indices] += 1.0 / indices.size
    return histogram


def compute_quartile_medians(frame):
    start = pd.to_numeric(frame["t1_mlt_echo"], errors="coerce").to_numpy()
    end = pd.to_numeric(frame["t2_mlt_echo"], errors="coerce").to_numpy()
    periods = pd.to_numeric(frame["max_echo_cand_per"], errors="coerce").to_numpy()
    finite = np.isfinite(start) & np.isfinite(end) & np.isfinite(periods) & (periods > 0)
    start = np.asarray(start[finite], dtype=float)
    end = np.asarray(end[finite], dtype=float)
    periods = periods[finite]
    midpoint = (start + (end - start) % 24.0 / 2.0) % 24.0
    medians = {}
    for index, (first, last, _, _) in enumerate(MLT_QUARTILES):
        mask = ((midpoint >= first) & (midpoint < last) if first < last
                else (midpoint >= first) | (midpoint < last))
        if mask.sum() > 0:
            medians[index] = np.median(periods[mask])
    return medians


def pitch_mask(angles, intervals):
    mask = np.zeros(len(angles), dtype=bool)
    for lower, upper in intervals:
        mask |= ((angles >= lower) & (angles <= upper) if upper >= 180.0
                 else (angles >= lower) & (angles < upper))
    return mask


def subset_for_panel(frame, intervals, rank):
    angles = pd.to_numeric(frame["median_pitch_angle_deg"], errors="coerce").to_numpy()
    ranks = pd.to_numeric(frame["energy_rank"], errors="coerce").to_numpy()
    mask = np.isfinite(angles) & np.isfinite(ranks) & pitch_mask(angles, intervals) & (ranks == rank)
    return frame.loc[mask].reset_index(drop=True)


def centered_angle_bin_mask(values, center_deg, bin_width=CLOCK_BIN_WIDTH):
    wrapped = np.mod(np.asarray(values, dtype=float), 360.0)
    delta = ((wrapped - float(center_deg) + 180.0) % 360.0) - 180.0
    return np.isfinite(wrapped) & (delta >= -0.5 * bin_width) & (delta < 0.5 * bin_width)


def peak_contour_level(display_data):
    positive = display_data[(display_data > 0) & np.isfinite(display_data)]
    return np.percentile(positive, PEAK_PERCENTILE) if positive.size else None


def _species_reference(catalog, species):
    # Rank and period limits come from the complete valid species population;
    # they must not be recomputed inside pitch or clock-angle panels.
    frame = catalog.loc[(catalog["species"] == species) & (catalog["category"] == "valid")].copy()
    frame = add_energy_rank(frame)
    ranks, labels = selected_ranks_and_labels(frame)
    return frame, ranks, labels, log_period_edges_for_df(frame)


def _display_vmax(displays):
    maximum = max((values[values > 0].max() for values in displays if np.any(values > 0)), default=1.0)
    return maximum if maximum > 0 else 1.0


def _style():
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})


def _mlt_axis(axis, panel_index):
    axis.set_xlim(0, 24)
    axis.set_xticks(MLT_TICKS)
    axis.grid(True, alpha=0.25)
    axis.text(0.02, 0.98, f"({string.ascii_lowercase[panel_index]})", transform=axis.transAxes,
              ha="left", va="top", fontsize=9)


def _overlay_quartiles(axis, frame):
    for index, median in compute_quartile_medians(frame).items():
        _, _, first, last = MLT_QUARTILES[index]
        segments = [(21, 24), (0, 3)] if last > 24 else [(first, last)]
        for left, right in segments:
            level = np.log10(median)
            axis.hlines(level, left, right, colors="red", linewidths=1.5, zorder=5)
            axis.text(0.5 * (left + right), level - 0.05, f"{median:.1f} min", color="red",
                      fontsize=6, fontweight="bold", ha="center", va="top", rotation=90, zorder=6)


def plot_distributions(catalog, output_dir):
    """Save Figures 3 and 4 without collapsing overlapping detection windows."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    mlt_edges = np.arange(0.0, 24.0 + MLT_BIN_WIDTH, MLT_BIN_WIDTH)
    for species, figure_number in (("fedu", 3), ("fpdu", 4)):
        frame, ranks, labels, period_edges = _species_reference(catalog, species)
        panels = {}
        for row, (_, intervals) in enumerate(PITCH_SECTIONS):
            for column, rank in enumerate(ranks):
                selected = subset_for_panel(frame, intervals, rank)
                display = np.log10(build_span_count_histogram(selected, mlt_edges, period_edges) + 1.0)
                panels[row, column] = display, selected
        vmax = _display_vmax([panel[0] for panel in panels.values()])
        _style()
        figure, axes = plt.subplots(3, 3, figsize=(7.5, 7.5), sharex=True, sharey=True,
                                    constrained_layout=True)
        try:
            for column, rank in enumerate(ranks):
                axes[0, column].set_title(labels[rank], pad=6)
            for row, (label, _) in enumerate(PITCH_SECTIONS):
                for column in range(3):
                    axis = axes[row, column]
                    display, selected = panels[row, column]
                    image = axis.pcolormesh(mlt_edges, period_edges, display, cmap=COLORMAP,
                                           vmin=0.0, vmax=vmax, shading="flat")
                    _mlt_axis(axis, row * 3 + column)
                    if column == 0:
                        axis.set_ylabel(f"{label}\nlog$_{{10}}$(Period [min])")
                    if row == 2:
                        axis.set_xlabel("MLT (hr)")
                    _overlay_quartiles(axis, selected)
            colorbar = figure.colorbar(image, ax=axes, pad=0.02, shrink=0.95)
            colorbar.set_label(r"$\log_{10}(\mathrm{count} + 1)$")
            destination = output_dir / f"Figure_{figure_number}.png"
            figure.savefig(destination, dpi=300, bbox_inches="tight")
            paths.append(destination)
        finally:
            plt.close(figure)
    return paths


def _overlay_peak_contour(axis, display, mlt_edges, period_edges):
    threshold = peak_contour_level(display)
    if threshold is None:
        return
    mlt_centers = 0.5 * (mlt_edges[:-1] + mlt_edges[1:])
    period_centers = 0.5 * (period_edges[:-1] + period_edges[1:])
    safe = np.where(np.isfinite(display), display, 0.0)
    if threshold >= safe.max():
        return  # No bins exceed the percentile in a sparse or flat panel.
    axis.contour(mlt_centers, period_centers, safe, levels=[threshold],
                 colors="red", linewidths=1.2, zorder=5)


def plot_clock_distributions(catalog, output_dir):
    """Save Figures 5–8 and the twenty remaining 15-degree IMF clock bins."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    references = [_species_reference(catalog, species) for species in ("fedu", "fpdu")]
    clocks = [pd.to_numeric(reference[0]["clock_angle_deg_median"], errors="coerce").to_numpy(dtype=float)
              for reference in references]
    mlt_edges = np.arange(0.0, 24.0 + MLT_BIN_WIDTH, MLT_BIN_WIDTH)
    ymin, ymax = min(reference[3][0] for reference in references), max(reference[3][-1] for reference in references)
    paths, supplement = [], 1
    for center in CLOCK_BIN_CENTERS:
        _style()
        figure, axes = plt.subplots(2, 3, figsize=(7.5, 5), sharex=True, sharey=True,
                                    constrained_layout=True)
        try:
            for column, label in enumerate(("Low-energy rank", "Middle-energy rank", "High-energy rank")):
                axes[0, column].set_title(label, pad=6)
            for row, ((frame, ranks, _, period_edges), values, species_label) in enumerate(
                    zip(references, clocks, ("FEDU (electrons)", "FPDU (protons)"))):
                conditioned = frame.loc[centered_angle_bin_mask(values, center)].reset_index(drop=True)
                panels = []
                for rank in ranks:
                    rank_values = pd.to_numeric(conditioned["energy_rank"], errors="coerce").to_numpy()
                    selected = conditioned.loc[np.isfinite(rank_values) & (rank_values == rank)].reset_index(drop=True)
                    display = (np.log10(build_span_count_histogram(selected, mlt_edges, period_edges) + 1.0)
                               if len(selected) else np.full((len(period_edges) - 1, len(mlt_edges) - 1), np.nan))
                    panels.append((display, len(selected)))
                vmax = _display_vmax([panel[0] for panel in panels])
                row_image = None
                for column, (display, count) in enumerate(panels):
                    axis = axes[row, column]
                    if np.any(np.isfinite(display)):
                        row_image = axis.pcolormesh(mlt_edges, period_edges, display, cmap=COLORMAP,
                                                   vmin=0.0, vmax=vmax, shading="flat")
                    _mlt_axis(axis, row * 3 + column)
                    axis.set_ylim(ymin, ymax)
                    if column == 0:
                        axis.set_ylabel(f"{species_label}\nlog10(Period [min])")
                    if row == 1:
                        axis.set_xlabel("MLT (hr)")
                    if count == 0:
                        axis.text(0.5, 0.5, "No detections", transform=axis.transAxes, ha="center", va="center")
                    else:
                        _overlay_peak_contour(axis, display, mlt_edges, period_edges)
                if row_image is not None:
                    colorbar = figure.colorbar(row_image, ax=axes[row, :], pad=0.02, shrink=0.95)
                    colorbar.set_label(r"$\log_{10}(\mathrm{count} + 1)$")
            if center in PAPER_ANGLES:
                name = f"Figure_{PAPER_ANGLES[center]}.png"
            else:
                name = f"Figure_SM_{supplement}_clock_{int(round(center)) % 360:03d}.png"
                supplement += 1
            destination = output_dir / name
            figure.savefig(destination, dpi=300, bbox_inches="tight")
            paths.append(destination)
        finally:
            plt.close(figure)
    return paths
