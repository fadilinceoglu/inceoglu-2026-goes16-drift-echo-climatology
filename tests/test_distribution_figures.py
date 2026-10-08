"""Scientific weighting, rank population, sector and contour contracts."""

import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "drift-distribution-test-mpl"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drift_echo_climatology import distribution_figures as figures


class ScientificDistributions(unittest.TestCase):
    def test_wrapped_span_weights_preserve_duplicate_detection_windows(self):
        # Two detections crossing midnight remain two detections. Another
        # detection occupies two centers; missing/out-of-range data add none.
        frame = pd.DataFrame({"t1_mlt_echo": [23.2, 23.2, 3.2, .6, np.nan, 2., 2.],
                              "t2_mlt_echo": [1.8, 1.8, 4.8, .9, 4., 4., 4.],
                              "max_echo_cand_per": [100., 100., 10., 10., 10., 1000., -1.]})
        actual = figures.build_span_count_histogram(frame, np.arange(25.),
                                                     np.array([.98, 1.02, 1.98, 2.02]))
        expected = np.zeros((3, 24))
        expected[2, [23, 0, 1]] = 2. / 3.
        expected[0, [3, 4]] = .5
        np.testing.assert_array_equal(actual, expected)
        self.assertAlmostEqual(actual.sum(), 3.)

    def test_energy_ranks_use_all_valid_species_rows_before_conditioning(self):
        frame = pd.DataFrame({"species": ["fedu"] * 7 + ["fpdu"],
                              "category": ["valid"] * 6 + ["invalid", "valid"],
                              "telescope_no": [1, 1, 1, 2, 2, 2, 1, 1],
                              "energy_channel": [50., 100., 200., 60., 130., 260., 10., 1.],
                              "max_echo_cand_per": [100., 50., 25., 110., 55., 27., 1000., 5.],
                              "clock_angle_deg_median": [0., 0., 60., 0., 0., 60., 60., 60.]})
        original = frame.copy(deep=True)
        ranked, ranks, labels, edges = figures._species_reference(frame, "fedu")
        self.assertEqual(ranked.energy_rank.to_list(), [0, 1, 2, 0, 1, 2])
        self.assertEqual(ranks, [0, 1, 2])
        self.assertEqual(labels, {0: "50-60 keV", 1: "100-130 keV", 2: "200-260 keV"})
        conditioned = ranked.loc[figures.centered_angle_bin_mask(ranked.clock_angle_deg_median, 60.)]
        self.assertEqual(conditioned.energy_rank.to_list(), [2, 2])
        self.assertAlmostEqual(edges[0], np.floor(np.log10(25.) / .02) * .02)
        pd.testing.assert_frame_equal(frame, original)

    def test_clock_bins_partition_wrapped_boundaries_once(self):
        values = np.array([-7.5, 7.5, 352.5, 367.5, 360., -180., 180., np.nan])
        np.testing.assert_array_equal(figures.centered_angle_bin_mask(values, 0.),
                                      [True, False, True, False, True, False, False, False])
        membership = np.stack([figures.centered_angle_bin_mask(values, center)
                               for center in figures.CLOCK_BIN_CENTERS]).sum(axis=0)
        np.testing.assert_array_equal(membership, [1, 1, 1, 1, 1, 1, 1, 0])

    def test_pitch_sectors_partition_endpoints_with_180_included(self):
        angles = np.array([0., 30., 60., 120., 150., 180., np.nan])
        masks = np.stack([figures.pitch_mask(angles, intervals) for _, intervals in figures.PITCH_SECTIONS])
        np.testing.assert_array_equal(masks.sum(axis=0), [1, 1, 1, 1, 1, 1, 0])
        np.testing.assert_array_equal(np.argmax(masks[:, :6], axis=0), [2, 1, 0, 1, 2, 2])

    def test_quartile_medians_use_wrapped_midpoints_and_half_open_boundaries(self):
        frame = pd.DataFrame({"t1_mlt_echo": [20., 23., 2., 8., 14.],
                              "t2_mlt_echo": [22., 1., 4., 10., 16.],
                              "max_echo_cand_per": [10., 30., 20., 40., 50.]})
        self.assertEqual(figures.compute_quartile_medians(frame), {0: 20., 1: 40., 2: 50., 3: 20.})

    def test_peak_contour_uses_nonzero_display_values(self):
        display = np.array([[0., 1., 2.], [3., np.nan, 0.]])
        self.assertEqual(figures.peak_contour_level(display), np.percentile([1., 2., 3.], 95))
        self.assertIsNone(figures.peak_contour_level(np.array([[0., np.nan]])))
        self.assertEqual(figures._display_vmax([np.zeros((2, 2)), display]), 3.)
        self.assertEqual(figures._display_vmax([np.zeros((2, 2))]), 1.)


if __name__ == "__main__":
    unittest.main()
