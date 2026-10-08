"""Scientific numerical contracts for the two example figures."""

from pathlib import Path
import sys
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drift_echo_climatology.example_figures import (
    _first_iteration_spectrum, _period_spectrum, _drift_period_minutes,
)


class ExampleFigures(unittest.TestCase):
    def test_known_sinusoid_retains_period_and_physical_amplitude(self):
        for size in (240, 480):
            with self.subTest(size=size):
                minutes = np.arange(size)
                signal = 3.0 * np.cos(2 * np.pi * minutes / 60.0)
                frequency, amplitude = _first_iteration_spectrum(signal)
                peak = np.argmax(amplitude)
                self.assertAlmostEqual(frequency[peak], 1 / 3600.0)
                self.assertAlmostEqual(amplitude[peak], 3.0, delta=0.03)
                period, plotted = _period_spectrum(frequency, amplitude)
                self.assertEqual(period[np.argmax(plotted)], 60.0)
                self.assertTrue(np.all(np.diff(period) > 0))
                self.assertTrue(np.all((period >= 5) & (period <= 240)))

    def test_window_gap_rule_and_input_preservation(self):
        signal = np.arange(240, dtype=float)
        signal[80:86] = np.nan
        original = signal.copy()
        _first_iteration_spectrum(signal)
        np.testing.assert_array_equal(signal, original)
        signal[86] = np.nan
        with self.assertRaisesRegex(ValueError, "finite detection window"):
            _first_iteration_spectrum(signal)

    def test_walt_reference_values_and_energy_ordering(self):
        cases = (("fedu", 50.0, 210.28657320127013, 15.014016773829436),
                 ("fpdu", 80.0, 125.59698251810741, 10.05268160135555))
        for species, minimum, first, last in cases:
            with self.subTest(species=species):
                energy = np.linspace(minimum, 1000.0, 1000)
                zero = _drift_period_minutes(energy, 0.0, species)
                equatorial = _drift_period_minutes(energy, 90.0, species)
                self.assertAlmostEqual(zero[0], first, places=11)
                self.assertAlmostEqual(zero[-1], last, places=11)
                self.assertTrue(np.all(np.diff(zero) < 0))
                np.testing.assert_allclose(equatorial, 0.6667 * zero, rtol=1e-15, atol=0)


if __name__ == "__main__":
    unittest.main()
