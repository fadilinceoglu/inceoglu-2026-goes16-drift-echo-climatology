"""The study's residual-subtraction CLEAN detector with an explicit RNG."""

from numbers import Integral

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from .config import (CLEAN_SIMULATIONS, CLEAN_MAX_ITERATIONS,
                     CLEAN_SIGNIFICANCE, CLEAN_FFT_LENGTH)


def _peak_model(frequency, amplitude, center, width):
    return amplitude * np.sinc((frequency - center) / width) ** 2


def _fit_ar1(values):
    centered = np.asarray(values) - np.nanmean(values)
    centered = centered[np.isfinite(centered)]
    if centered.size < 20:
        raise RuntimeError("Too short for reliable AR(1)")
    r0 = np.dot(centered, centered) / centered.size
    r1 = np.dot(centered[1:], centered[:-1]) / (centered.size - 1)
    phi = np.clip(r1 / (r0 + 1e-12), -0.99, 0.99)
    return phi, np.sqrt(max((1 - phi ** 2) * r0, 1e-12))


def _simulate_ar1(phi, sigma, size, simulations, rng):
    innovations = rng.normal(0, sigma, (simulations, size))
    noise = np.zeros_like(innovations)
    for index in range(1, size):
        noise[:, index] = phi * noise[:, index - 1] + innovations[:, index]
    return noise


def clean_window(values, time_seconds, *, rng, simulations=CLEAN_SIMULATIONS,
                 max_iterations=CLEAN_MAX_ITERATIONS, significance=CLEAN_SIGNIFICANCE,
                 fft_length=CLEAN_FFT_LENGTH):
    """Return significant components from a finite one-minute flux window.

    Time is in seconds since the anchor UTC midnight, preserving fitted phase.
    Use a local numpy.random.RandomState for reproducible draws.
    Instrumental Poisson errors do not enter this residual-noise calculation.
    """
    values = np.asarray(values)
    time_seconds = np.asarray(time_seconds, dtype=float)
    if (values.ndim != 1 or time_seconds.ndim != 1 or values.size < 2
            or values.size != time_seconds.size or not np.isfinite(values).all()
            or not np.isfinite(time_seconds).all()):
        raise ValueError("CLEAN requires equal finite one-dimensional arrays with at least two samples")
    if not np.allclose(np.diff(time_seconds), 60.0, rtol=0, atol=1e-8):
        raise ValueError("CLEAN requires a uniform increasing one-minute time grid")
    for name, value in (("simulations", simulations), ("max_iterations", max_iterations),
                        ("fft_length", fft_length)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(name + " must be a positive integer")
    if fft_length < values.size:
        raise ValueError("fft_length must be at least the window length")
    if not np.isfinite(significance) or not 0 <= significance <= 1:
        raise ValueError("significance must be between zero and one")
    if not isinstance(rng, np.random.RandomState):
        raise TypeError("rng must be a numpy.random.RandomState")

    residual = values.copy() if np.issubdtype(values.dtype, np.floating) else values.astype(float)
    size = values.size
    step = np.mean(np.diff(time_seconds))
    initial_std = np.nanstd(residual)
    resolution = 1 / (size * step)
    result = pd.DataFrame(columns=["freq", "amp", "phase"])
    rejected = []
    gain = 0.5

    for iteration in range(max_iterations):
        taper = np.hanning(size)
        pad_left = (fft_length - size) // 2
        pad_right = fft_length - size - pad_left
        signal = np.pad(taper * residual, (pad_left, pad_right), mode="constant", constant_values=0)
        frequency = np.fft.rfftfreq(fft_length, step)
        amplitude = (2.0 / (size * gain)) * np.abs(np.fft.rfft(signal))

        # The first iteration preserves the original broad AR(1) fallback;
        # later iterations fall back only for a too-short residual.
        if iteration == 0:
            try:
                phi, sigma = _fit_ar1(residual)
                noise = _simulate_ar1(phi, sigma, size, simulations, rng)
            except Exception:
                std = min(np.nanstd(residual), initial_std * 2)
                noise = rng.normal(0, std, (simulations, size))
        else:
            try:
                phi, sigma = _fit_ar1(residual)
                noise = _simulate_ar1(phi, sigma, size, simulations, rng)
            except RuntimeError:
                noise = rng.normal(0, np.nanstd(residual), (simulations, size))
        noise = np.pad(noise * taper[None, :], ((0, 0), (pad_left, pad_right)),
                       mode="constant", constant_values=0)
        noise_amplitude = ((2.0 / (size * gain)) * np.abs(np.fft.rfft(noise, axis=1))).T
        peaks, _ = find_peaks(amplitude, height=np.median(noise_amplitude, axis=1)
                             + 4 * np.std(noise_amplitude, axis=1))
        if peaks.size == 0:
            break
        peak = peaks[np.argmax(amplitude[peaks])]
        peak_amplitude, peak_frequency = amplitude[peak], frequency[peak]
        fitted = False
        for width in range(50, 4, -1):
            left, right = max(0, peak - width), min(len(frequency), peak + width + 1)
            x, y = frequency[left:right], amplitude[left:right]
            if len(x) < 5:
                continue
            guess = [np.max(y), frequency[peak], 2 / (size * step)]
            try:
                parameters, _ = curve_fit(_peak_model, x, y, p0=guess,
                                          bounds=([0, frequency[0], 0],
                                                  [np.inf, frequency[-1], np.inf]))
                peak_frequency, peak_amplitude = parameters[1], parameters[0]
                fitted = True
                break
            except (RuntimeError, ValueError):
                continue
        if not fitted:
            peak_frequency = frequency[peak]
            peak_amplitude = amplitude[peak]
            if 0 < peak < len(amplitude) - 1:
                polynomial = np.polyfit(frequency[peak - 1:peak + 2], amplitude[peak - 1:peak + 2], 2)
                peak_frequency = -polynomial[1] / (2 * polynomial[0])

        nearest = np.argmin(np.abs(frequency - peak_frequency))
        exceedance = np.sum(noise_amplitude[nearest] > peak_amplitude) / simulations
        if exceedance > significance:
            rejected.append(peak_frequency)
            continue
        # Preserve the study's iteration behavior: rejected/duplicate peaks
        # leave the residual unchanged and may be selected again next time.
        if not result.empty:
            differences = np.abs(result["freq"] - peak_frequency)
            if (differences < resolution).any() or any(abs(value - peak_frequency) < resolution
                                                      for value in rejected):
                rejected.append(peak_frequency)
                continue

        cosine = np.cos(2 * np.pi * peak_frequency * time_seconds)
        sine = np.sin(2 * np.pi * peak_frequency * time_seconds)
        coefficients, _, _, _ = np.linalg.lstsq(np.vstack([cosine, sine]).T, residual, rcond=None)
        c, d = coefficients
        fitted_amplitude = np.sqrt(c ** 2 + d ** 2)
        phase = np.arctan2(-d, c)
        result.loc[iteration, "freq"] = peak_frequency
        result.loc[iteration, "amp"] = fitted_amplitude
        result.loc[iteration, "phase"] = phase
        residual -= fitted_amplitude * np.cos(2 * np.pi * peak_frequency * time_seconds + phase)

    result = result.loc[result["freq"] > 1 / (60 * size / 2)].reset_index(drop=True)
    if result.empty:
        return pd.DataFrame(columns=["freq", "amp", "phase", "period"])
    result["period"] = (1 / result["freq"]) / 60
    return result
