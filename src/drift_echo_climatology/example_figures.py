"""Render the paper's pipeline examples and theoretical drift-period curves."""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CLEAN_FFT_LENGTH, DRIFT_CONSTANT_SECONDS, LSHELL, REST_ENERGY_KEV, WINDOW_MINUTES
from .preprocessing import load_prepared


CHANNEL_COLORS = ("#6a3d9a", "#e31a1c", "#b8860b", "#1f78b4", "#6b8e23")


def _first_iteration_spectrum(values):
    """Use the detector's interpolation, Hann taper and amplitude normalization."""
    signal = pd.Series(np.asarray(values, dtype=float)).interpolate(
        method="linear", limit=3, limit_direction="both").to_numpy()
    if signal.ndim != 1 or signal.size < 2 or signal.size > CLEAN_FFT_LENGTH or not np.isfinite(signal).all():
        raise ValueError("The example spectrum requires a finite detection window")
    size = signal.size
    left = (CLEAN_FFT_LENGTH - size) // 2
    right = CLEAN_FFT_LENGTH - size - left
    tapered = np.pad(np.hanning(size) * signal, (left, right), mode="constant", constant_values=0)
    frequency = np.fft.rfftfreq(CLEAN_FFT_LENGTH, 60.0)
    amplitude = (2.0 / (size * 0.5)) * np.abs(np.fft.rfft(tapered))
    return frequency, amplitude


def _period_spectrum(frequency, amplitude):
    period = np.full(frequency.shape, np.nan, dtype=float)
    positive = frequency > 0
    period[positive] = (1.0 / frequency[positive]) / 60.0
    keep = positive & np.isfinite(period) & (period >= 5.0) & (period <= 240.0)
    order = np.argsort(period[keep])
    return period[keep][order], amplitude[keep][order]


def _drift_period_minutes(energy, pitch_degrees, species):
    normalized = np.asarray(energy, dtype=float) / REST_ENERGY_KEV[species]
    angle = pitch_degrees * np.pi / 180.0
    seconds = (DRIFT_CONSTANT_SECONDS[species] / LSHELL
               * ((normalized + 1) / (normalized * (normalized + 2)))
               * (1 - 0.3333 * np.sin(angle) ** 0.62))
    return seconds / 60.0


def _example_event(catalog, day, species, start_hour):
    date = pd.Timestamp(day)
    dates = pd.to_datetime(catalog["date"], errors="raise")
    starts = pd.to_datetime(catalog["date1_utc_echo"], errors="raise")
    event = catalog.loc[(catalog["category"] == "valid") & (catalog["species"] == species)
                        & (catalog["satellite"] == "goes16") & (catalog["telescope_no"] == 2)
                        & (dates.dt.normalize() == date)
                        & (starts == date + pd.Timedelta(hours=start_hour))].copy()
    if event.empty:
        raise ValueError(f"Missing valid Figure 1 example: {day}, {species}, Telescope 2, {start_hour:02d}:00 UTC")
    return event.sort_values("energy_channel").reset_index(drop=True)


def _energy_period_panel(axis, event):
    axis.plot(event.energy_channel, event.max_echo_cand_per, "k.", label="CLEAN DE")
    axis.plot(event.energy_channel, event.expected_drift_per, color="tab:orange", marker=".",
              linestyle="", label="DE from dipole + alpha")
    axis.plot(event.energy_channel, event.est_drift_per_a0_adj, color="tab:red", linewidth=2,
              label=r"DE from dipole + adj $\alpha_0$")
    axis.plot(event.energy_channel, event.est_drift_per_a90_adj, color="tab:blue", linewidth=2,
              label=r"DE from dipole + adj $\alpha_{90}$")
    axis.set_xlabel("Energy Channel (keV)")
    axis.set_ylabel("Period (min)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper right")


def plot_figure_1(prepared_dir, catalog, output_dir):
    """Save Figure_1.png from prepared days and their valid sequences.

    The top panels use the anchor day's raw flux. The spectra use the stored
    high-pass series, which was filtered over the complete padded day before
    extracting these fixed study windows. All catalog periods are in minutes.
    """
    examples = (("2019-06-16", "fedu", 14), ("2019-11-23", "fpdu", 11))
    inputs = []
    for day, species, hour in examples:
        path = Path(prepared_dir) / ("g16_d" + day.replace("-", "") + ".npz")
        prepared = load_prepared(path)
        if str(prepared["date"]) != day:
            raise ValueError("Prepared example filename and anchor date disagree")
        event = _example_event(catalog, day, species, hour)
        inputs.append((prepared, event))
    style = {"agg.path.chunksize": 10000, "font.size": 9, "axes.titlesize": 9,
             "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7}
    with plt.rc_context(style):
        figure, axes = plt.subplots(3, 2, figsize=(7.5, 9.5))
        try:
            for column, ((day, species, hour), (prepared, event)) in enumerate(zip(examples, inputs)):
                start = pd.Timestamp(day) + pd.Timedelta(hours=hour)
                end = start + pd.Timedelta(minutes=WINDOW_MINUTES[species])
                time = pd.to_datetime(prepared["j2000"][:1440], unit="s", origin="2000-01-01T12:00:00")
                energies = prepared[species + "_energies"][1]
                units = str(prepared[species + "_units"])
                letter = "E" if species == "fedu" else "P"
                axes[0, column].axvspan(start, end, color="gray", alpha=0.18, zorder=0)
                for channel, color in enumerate(CHANNEL_COLORS):
                    label = f"{letter}{channel + 1} ({energies[channel]:.0f} keV)"
                    axes[0, column].plot(time, prepared[species][:1440, 1, channel],
                                         lw=0.8, color=color, label=label)
                    values = prepared[species + "_hp"][hour * 60:hour * 60 + WINDOW_MINUTES[species], 1, channel]
                    frequency, amplitude = _first_iteration_spectrum(values)
                    period, amplitude = _period_spectrum(frequency, amplitude)
                    axes[1, column].plot(period, amplitude, lw=1.0, color=color,
                                         label=f"{letter}{channel + 1} FFT ({energies[channel]:.0f} keV)")
                axes[0, column].set_yscale("log")
                axes[0, column].set_ylabel(units)
                axes[0, column].grid(True, alpha=0.25)
                axes[0, column].legend(loc="upper left", ncol=2)
                axes[0, column].xaxis.set_major_locator(mdates.HourLocator(interval=3))
                axes[0, column].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                axes[0, column].set_xlabel("UT (hour:min)")
                axes[1, column].set_yscale("log")
                axes[1, column].set_ylabel(f"FFT Amplitude ({units})")
                axes[1, column].set_xlabel("Period (min)")
                axes[1, column].grid(True, alpha=0.25)
                axes[1, column].legend(loc="upper right", ncol=2)
                _energy_period_panel(axes[2, column], event)
            for axis, label in zip(axes.flat, ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")):
                axis.text(0.02, 0.02, label, transform=axis.transAxes,
                          ha="left", va="bottom", fontsize=9)
            figure.tight_layout(rect=[0, 0, 1, 0.98])
            output = Path(output_dir) / "Figure_1.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output, dpi=300)
        finally:
            plt.close(figure)
    return output


def plot_theory(output_dir):
    """Save Figure_2.png using the paper's Walt dipole relation at L=6.6."""
    style = {"font.size": 14, "axes.labelsize": 16, "xtick.labelsize": 14,
             "ytick.labelsize": 14, "legend.fontsize": 14}
    with plt.rc_context(style):
        figure, axis = plt.subplots(figsize=(7, 7))
        try:
            for species, name, minimum, color in (("fedu", "electrons", 50.0, "purple"),
                                                   ("fpdu", "protons", 80.0, "orange")):
                energies = np.linspace(minimum, 1000.0, 1000)
                for pitch, linestyle in ((90.0, "-"), (0.0, ":")):
                    axis.plot(energies, _drift_period_minutes(energies, pitch, species), color=color,
                              linestyle=linestyle, linewidth=3,
                              label=rf"{name}, $\alpha_{{eq}} = {pitch:.0f}^\circ$")
            axis.set_xlim(0, 1000.0)
            axis.set_ylim(0, 215)
            axis.set_xlabel("particle kinetic energy (keV)")
            axis.set_ylabel("drift period (minutes)")
            axis.minorticks_on()
            axis.grid(True, which="major", color="0.82", linewidth=0.8)
            axis.grid(True, which="minor", color="0.90", linewidth=0.4)
            axis.legend(loc="upper right", frameon=True)
            figure.tight_layout()
            output = Path(output_dir) / "Figure_2.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output, dpi=200)
        finally:
            plt.close(figure)
    return output
