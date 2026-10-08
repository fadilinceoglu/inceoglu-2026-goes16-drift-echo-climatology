"""Scientific path, ranking, envelope, and selected-table contracts."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drift_echo_climatology import selection
from drift_echo_climatology.detection import COLUMNS, _typed


def candidates(periods=(80., 60., 40., 20.), energies=(100., 200., 300., 400.),
               species="fedu", start=0, telescope=1):
    midnight = pd.Timestamp("2019-06-16")
    rows = []
    for channel, (energy, period) in enumerate(zip(energies, periods), 1):
        row = {name: 0. for name in COLUMNS}
        row.update(species=species, date=midnight, satellite="goes16",
                   telescope_no=telescope, energy_channel_index=channel,
                   energy_channel=energy, window_start_minute=start, window_seed=123,
                   date1_utc_echo=midnight + pd.Timedelta(minutes=start),
                   date2_utc_echo=midnight + pd.Timedelta(minutes=start + 479),
                   amp=1., phase=0., max_echo_cand_frq=1 / (period * 60),
                   max_echo_cand_per=period, expected_drift_per=period,
                   est_drift_per_a90_adj=period - 10., est_drift_per_a0_adj=period + 10.)
        rows.append(row)
    return _typed(pd.DataFrame(rows, columns=COLUMNS))


class LongestSequences(unittest.TestCase):
    def test_all_peaks_and_all_tied_longest_paths_keep_stack_order(self):
        frame = pd.DataFrame({"energy_channel": [1., 1., 2., 2., 3., 4.],
                              "max_echo_cand_per": [80., 90., 50., 60., 30., 10.],
                              "row_id": np.arange(6)})
        paths, length = selection.longest_sequences(frame)
        self.assertEqual(length, 4)
        self.assertEqual([path.row_id.to_list() for path in paths],
                         [[1, 3, 4, 5], [0, 3, 4, 5], [1, 2, 4, 5], [0, 2, 4, 5]])

    def test_available_energy_neighbors_bridge_missing_nominal_channels(self):
        frame = candidates(energies=(100., 300., 500., 700.))
        frame["energy_channel_index"] = [1, 3, 5, 7]
        paths, length = selection.longest_sequences(frame)
        self.assertEqual(length, 4)
        self.assertEqual(paths[0].energy_channel_index.to_list(), [1, 3, 5, 7])

    def test_local_neighbors_cannot_jump_an_available_blocking_energy(self):
        frame = candidates(periods=(80., 90., 40., 20.))
        self.assertEqual(selection.longest_sequences(frame), [])
        paths, length = selection.longest_sequences(frame, min_len=3)
        self.assertEqual(length, 3)
        self.assertEqual(paths[0].energy_channel.to_list(), [200., 300., 400.])
        self.assertEqual(selection.longest_sequences(candidates(periods=(80., 60., 60., 20.))), [])

    def test_six_decimal_energy_period_dedup_precedes_amplitude_ranking(self):
        frame = candidates(periods=(80., 80.0000001, 60., 40., 20.),
                           energies=(100., 100., 200., 300., 400.))
        frame.loc[0, "amp"] = 1000.
        frame.loc[1, "amp"] = 1.
        paths, length = selection.longest_sequences(frame)
        self.assertEqual(length, 4)
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].iloc[0].amp, 1.)
        self.assertEqual(paths[0].iloc[0].max_echo_cand_per, 80.0000001)


class MetricsAndRanking(unittest.TestCase):
    def test_expected_is_finite_subset_and_advisory_for_classification(self):
        metrics = selection.envelope_metrics([15., 15.00000001, 50., 50.], [0.] * 4, [100.] * 4,
                                             [0., 0., np.nan, np.nan])
        self.assertEqual(metrics["orange_n"], 2)
        self.assertEqual(metrics["frac_within_expected_band"], .5)
        self.assertEqual(selection.classify_metrics(metrics)[0], "valid_gray")
        metrics["orange_n"] = 4
        metrics["exp_norm_bias"] = -100.
        metrics["frac_within_expected_band"] = 0.
        self.assertEqual(selection.classify_metrics(metrics, use_orange=True)[0], "valid_gray")

    def test_missing_expected_uses_infinity_then_curvature_then_amplitude(self):
        smooth = candidates(periods=(80., 40., 20., 10.))
        smooth["expected_drift_per"] = np.nan
        jagged = smooth.copy()
        jagged.loc[0, "max_echo_cand_per"] = 160.
        jagged.loc[0, "est_drift_per_a0_adj"] = 170.
        jagged.loc[0, "est_drift_per_a90_adj"] = 150.
        jagged["amp"] = 1000.
        smooth_score, metrics = selection.score_sequence(smooth)
        jagged_score, _ = selection.score_sequence(jagged)
        self.assertTrue(np.isinf(smooth_score[3]))
        self.assertFalse(metrics["orange_used"])
        self.assertLess(smooth_score, jagged_score)
        louder = smooth.copy()
        louder["amp"] = 10.
        self.assertLess(selection.score_sequence(louder)[0], smooth_score)

    def test_envelope_lexicographically_overrides_large_amplitude(self):
        good = candidates()
        bad = good.copy()
        bad.loc[0, "max_echo_cand_per"] = 120.
        bad["amp"] = 1e20
        self.assertLess(selection.score_sequence(good)[0], selection.score_sequence(bad)[0])

    def test_hard_and_edge_boundary_operators_match_original(self):
        base = selection.envelope_metrics([50.] * 4, [0.] * 4, [100.] * 4)
        for key, threshold in (("frac_outside_envelope_strict", .10),
                               ("frac_outside_envelope_soft", .15),
                               ("mean_severity_outside_soft", .08)):
            with self.subTest(key=key):
                metrics = dict(base, **{key: threshold})
                self.assertEqual(selection.classify_metrics(metrics)[0], "valid_gray")
                metrics[key] = np.nextafter(threshold, np.inf)
                self.assertEqual(selection.classify_metrics(metrics)[0], "invalid")
        for key, threshold in (("frac_outside_envelope_strict", .01),
                               ("frac_outside_envelope_soft", .03),
                               ("mean_severity_outside_soft", .05)):
            with self.subTest(key=key):
                metrics = dict(base, **{key: threshold})
                self.assertEqual(selection.classify_metrics(metrics)[0], "edge")
                metrics[key] = np.nextafter(threshold, np.inf)
                self.assertEqual(selection.classify_metrics(metrics)[0], "valid_gray")
        self.assertEqual(selection.classify_metrics(dict(base, ned_mean=.08))[0], "valid_gray")
        self.assertEqual(selection.classify_metrics(dict(base, ned_mean=np.nextafter(.08, 0)))[0], "edge")


class SelectedTable(unittest.TestCase):
    def test_all_four_categories_and_finite_subset_retains_original_rows(self):
        valid = candidates(start=0)
        edge = candidates(start=120)
        edge["est_drift_per_a90_adj"] = edge.max_echo_cand_per - 1.
        edge["est_drift_per_a0_adj"] = edge.max_echo_cand_per + 19.
        invalid = candidates(start=240)
        invalid.loc[0, "est_drift_per_a0_adj"] = 60.
        single = candidates(start=360)
        single.loc[0, "est_drift_per_a0_adj"] = np.nan
        frame = pd.concat([valid, edge, invalid, single], ignore_index=True)
        original = frame.copy(deep=True)
        selected, stats = selection.select_candidates(frame)
        self.assertEqual(selected.category.drop_duplicates().to_list(), ["valid", "edge", "invalid", "single"])
        self.assertEqual(stats["categories"], dict(valid=1, edge=1, invalid=1, single=1))
        self.assertEqual(stats["events_selected"], 4)
        self.assertEqual(stats["rows_selected"], 16)
        self.assertTrue(selected.sequence_length.eq(4).all())
        self.assertTrue(np.isnan(selected.loc[selected.category == "single", "est_drift_per_a0_adj"]).any())
        pd.testing.assert_frame_equal(frame, original)

    def test_energy_cutoff_is_strict_and_minimum_path_is_four(self):
        below = candidates(energies=(100., 200., 300., np.nextafter(1050., 0.)))
        selected, _ = selection.select_candidates(below)
        self.assertEqual(len(selected), 4)
        below.loc[3, "energy_channel"] = 1050.
        selected, stats = selection.select_candidates(below)
        self.assertTrue(selected.empty)
        self.assertEqual(stats["candidates_below_energy_threshold"], 3)
        self.assertEqual(stats["groups_without_sequence"], 1)

    def test_exact_score_tie_keeps_first_path_from_original_traversal(self):
        frame = candidates(periods=(80., 90., 60., 40., 20.),
                           energies=(100., 100., 200., 300., 400.))
        metrics = selection.envelope_metrics([50.] * 4, [0.] * 4, [100.] * 4)
        with patch.object(selection, "score_sequence", return_value=((0., 0., 0., 0., 0., 0.), metrics)):
            selected, stats = selection.select_candidates(frame)
        self.assertEqual(selected.max_echo_cand_per.to_list(), [90., 60., 40., 20.])
        self.assertEqual(stats["sequences_considered"], 2)

    def test_species_telescope_and_window_order_and_no_cross_window_dedup(self):
        frame = pd.concat([candidates(species="fedu", start=120, telescope=2),
                           candidates(species="fpdu", start=120, telescope=2),
                           candidates(species="fpdu", start=120, telescope=1),
                           candidates(species="fpdu", start=0, telescope=1)], ignore_index=True)
        selected, stats = selection.select_candidates(frame)
        events = selected.drop_duplicates("event_id")
        self.assertEqual(list(zip(events.species, events.telescope_no, events.window_start_minute)),
                         [("fpdu", 1, 120), ("fpdu", 1, 0), ("fpdu", 2, 120), ("fedu", 2, 120)])
        self.assertEqual(stats["events_selected"], 4)
        self.assertTrue(events.event_id.str.startswith("goes16:").all())
        again, _ = selection.select_candidates(frame)
        pd.testing.assert_frame_equal(selected, again, check_exact=True)

    def test_empty_output_preserves_all_columns_and_types(self):
        frame = _typed(pd.DataFrame(columns=COLUMNS))
        selected, stats = selection.select_candidates(frame)
        self.assertEqual(list(selected), selection.SELECTED_COLUMNS)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(selected.date))
        self.assertTrue(pd.api.types.is_integer_dtype(selected.telescope_no))
        self.assertTrue(pd.api.types.is_integer_dtype(selected.sequence_length))
        self.assertTrue(pd.api.types.is_float_dtype(selected.max_echo_cand_per))
        self.assertEqual(stats["groups_total"], 0)
        self.assertEqual(stats["events_selected"], 0)


if __name__ == "__main__":
    unittest.main()
