"""Synthetic signal recovery, null behavior, and reproducible local noise."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drift_echo_climatology.clean import clean_window


class CleanWindow(unittest.TestCase):
    def run_clean(self, values, seed=15, **options):
        settings = dict(simulations=128, max_iterations=8)
        settings.update(options)
        return clean_window(values, np.arange(len(values)) * 60.,
                            rng=np.random.RandomState(seed), **settings)

    def test_known_period_amplitude_and_phase(self):
        samples = np.arange(240)
        result = self.run_clean(8 * np.cos(2 * np.pi * samples / 40 + .4))
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.period.iloc[0], 40., delta=.01)
        self.assertAlmostEqual(result.amp.iloc[0], 8., delta=.001)
        self.assertAlmostEqual(result.phase.iloc[0], .4, delta=.001)
        self.assertAlmostEqual(result.freq.iloc[0], 1 / 2400., delta=1e-7)

    def test_two_components_are_recovered_by_residual_subtraction(self):
        samples = np.arange(480)
        values = (8 * np.cos(2 * np.pi * samples / 40 + .4)
                  + 5 * np.cos(2 * np.pi * samples / 75 - .3))
        result = self.run_clean(values)
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(sorted(result.period), [40., 75.], atol=.02, rtol=0)
        np.testing.assert_allclose(result.amp.to_numpy(dtype=float), [8., 5.], atol=.25, rtol=0)

    def test_white_noise_and_zero_do_not_produce_false_components(self):
        noise = np.random.RandomState(88).normal(size=240)
        for values in (noise, np.zeros(240)):
            with self.subTest(kind="zero" if not values.any() else "white"):
                result = self.run_clean(values)
                self.assertTrue(result.empty)
                self.assertEqual(list(result.columns), ["freq", "amp", "phase", "period"])

    def test_local_seed_repeatability_and_global_rng_independence(self):
        values = 8 * np.cos(2 * np.pi * np.arange(240) / 40 + .4)
        before = values.copy()
        np.random.seed(91)
        state = np.random.get_state()
        first = self.run_clean(values)
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        np.random.seed(44)
        second = self.run_clean(values)
        pd.testing.assert_frame_equal(first, second, check_exact=True)
        np.testing.assert_array_equal(values, before)

    def test_periods_longer_than_half_the_window_are_excluded(self):
        # A240-minute window needs strictly more than two cycles, as in the study.
        for period in (120., 150.):
            values = 8 * np.cos(2 * np.pi * np.arange(240) / period + .4)
            with self.subTest(period=period):
                self.assertTrue(self.run_clean(values, max_iterations=4).empty)

    def test_invalid_windows_and_parameters_fail_explicitly(self):
        rng = np.random.RandomState(15)
        for values, times in (([], []), ([1], [0]), ([1, np.nan], [0, 60]),
                              ([1, 2], [0]), ([1, 2], [0, 30]),
                              ([1, 2], [60, 0]), ([1, 2], [0, np.inf]),
                              ([[1, 2]], [0, 60])):
            with self.subTest(values=values), self.assertRaises(ValueError):
                clean_window(values, times, rng=rng)
        for settings in (dict(simulations=0), dict(simulations=1.5),
                         dict(max_iterations=0), dict(fft_length=1),
                         dict(significance=-.1), dict(significance=np.nan)):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                clean_window([1., 2.], [0., 60.], rng=rng, **settings)
        with self.assertRaisesRegex(TypeError, "RandomState"):
            clean_window([1., 2.], [0., 60.], rng=np.random.default_rng(15))


if __name__ == "__main__":
    unittest.main()
