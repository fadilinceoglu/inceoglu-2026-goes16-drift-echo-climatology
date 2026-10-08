"""Window and flat-table contracts without expensive noise simulations."""

from datetime import datetime
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drift_echo_climatology import detection


def prepared_day(samples=1440):
    day = datetime(2019, 6, 16)
    midnight = (day - datetime(2000, 1, 1, 12)).total_seconds()
    data = {"date": np.array(day.date().isoformat()),
            "j2000": midnight + np.arange(samples) * 60.,
            "mlt": (np.arange(samples) / 60. + 3.) % 24.}
    for species, mass, constant in (("fedu", 510.998950, 1.557e4),
                                    ("fpdu", 938272.089, 8.481)):
        energies = np.tile([100., 200.], (5, 1))
        normalized = energies / mass
        drift = constant / 6.6 * (normalized + 1) / (normalized * (normalized + 2))
        flux = np.broadcast_to(np.arange(samples)[:, None, None], (samples, 5, 2)).astype(float).copy()
        data.update({species: flux.copy(), species + "_hp": flux,
                     species + "_energies": energies,
                     species + "_drift_seconds_0": drift,
                     species + "_drift_seconds_90": drift * (1 - 0.3333),
                     "pitch_" + species: np.full((samples, 5), 60.)})
    return data


def one_peak(values, seconds, **settings):
    return pd.DataFrame({"freq": [1 / 1800.], "amp": [2.], "phase": [0.25], "period": [30.]})


class DetectionContract(unittest.TestCase):
    def detect(self, prepared, **selectors):
        settings = dict(species=("fedu",), telescopes=[1], channels=[1], starts=[0])
        settings.update(selectors)
        return detection.detect_day(prepared, **settings)

    def test_window_starts_preserve_endpoint_rule_and_anchor_day(self):
        for samples, species, step, count, last in (
                (1440, "fedu", 120, 8, 840), (1440, "fpdu", 60, 20, 1140),
                (1920, "fedu", 120, 12, 1320), (1920, "fpdu", 60, 24, 1380)):
            with self.subTest(samples=samples, species=species):
                starts = detection.window_starts(samples, species)
                np.testing.assert_array_equal(starts, np.arange(count) * step)
                self.assertEqual(starts[-1], last)
                self.assertTrue((starts < 1440).all())

    def test_coded_bidirectional_interpolation_fills_six_but_skips_seven(self):
        for gap, processed in ((6, 1), (7, 0)):
            with self.subTest(gap=gap):
                prepared = prepared_day()
                prepared["fedu_hp"][100:100 + gap, 0, 0] = np.nan
                with patch.object(detection, "clean_window", side_effect=one_peak) as clean:
                    table, stats = self.detect(prepared)
                self.assertEqual(clean.call_count, processed)
                self.assertEqual(len(table), processed)
                self.assertEqual(stats["windows_skipped_missing_flux"], 1 - processed)
                self.assertEqual(stats["windows_processed"], processed)
                if processed:
                    np.testing.assert_array_equal(clean.call_args.args[0], np.arange(480))
                self.assertTrue(np.isnan(prepared["fedu_hp"][100:100 + gap, 0, 0]).all())

    def test_missing_pitch_retains_spectral_candidates_and_minute_units(self):
        prepared = prepared_day()
        prepared["pitch_fedu"][:] = np.nan
        with patch.object(detection, "clean_window", side_effect=one_peak), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            table, stats = self.detect(prepared)
        row = table.iloc[0]
        self.assertEqual(stats["candidates"], 1)
        for name in ("median_pitch_angle_deg", "expected_drift_per", "energy_keV_from_period"):
            self.assertTrue(pd.isna(row[name]), name)
        self.assertEqual(row["max_echo_cand_per"], 30.)
        self.assertEqual(row["max_echo_cand_frq"], 1 / 1800.)
        self.assertEqual(row["phase"], 0.25)
        drift0 = prepared["fedu_drift_seconds_0"][0, 0] / 60
        drift90 = prepared["fedu_drift_seconds_90"][0, 0] / 60
        self.assertEqual(row["est_drift_per_a0"], drift0)
        self.assertEqual(row["est_drift_per_a90"], drift90)
        self.assertEqual(row["est_drift_per_a0_adj"], 1.25 * drift0)
        self.assertEqual(row["est_drift_per_a90_adj"], drift90 / 3)
        self.assertEqual(row["energy_channel"], 100.)
        self.assertEqual(row["energy_channel_index"], 1)

    def test_expected_period_and_inverse_energy_preserve_seconds_conversion(self):
        prepared = prepared_day()
        for species in ("fedu", "fpdu"):
            with self.subTest(species=species):
                seconds = prepared[species + "_drift_seconds_0"][0, 0] * (1 - 0.3333 * np.sin(np.deg2rad(60.)) ** 0.62)
                def peak_at_expected_period(values, time, **settings):
                    return pd.DataFrame({"freq": [1 / seconds], "amp": [2.], "phase": [0.], "period": [seconds / 60]})
                with patch.object(detection, "clean_window", side_effect=peak_at_expected_period):
                    table, _ = self.detect(prepared, species=(species,))
                self.assertEqual(table.iloc[0]["expected_drift_per"], seconds / 60)
                self.assertAlmostEqual(table.iloc[0]["energy_keV_from_period"], 100., places=8)

    def test_midnight_window_preserves_day_relative_phase_origin_and_endpoints(self):
        prepared = prepared_day(1920)
        with patch.object(detection, "clean_window", side_effect=one_peak) as clean:
            table, _ = self.detect(prepared, starts=[1320])
        np.testing.assert_array_equal(clean.call_args.args[1], np.arange(1320, 1800) / 60. * 3600)
        row = table.iloc[0]
        self.assertEqual(row["date"], pd.Timestamp("2019-06-16"))
        self.assertEqual(row["date1_utc_echo"], pd.Timestamp("2019-06-16T22:00:00"))
        self.assertEqual(row["date2_utc_echo"], pd.Timestamp("2019-06-17T05:59:00"))
        self.assertEqual(row["t1_mlt_echo"], prepared["mlt"][1320])
        self.assertEqual(row["t2_mlt_echo"], prepared["mlt"][1799])

    def test_window_rng_is_stable_for_reordered_and_individually_selected_jobs(self):
        def random_peak(values, time, **settings):
            result = one_peak(values, time)
            result["amp"] = settings["rng"].uniform(1., 2.)
            return result
        prepared = prepared_day()
        with patch.object(detection, "clean_window", side_effect=random_peak):
            first, _ = self.detect(prepared, telescopes=[2, 1], channels=[2, 1], starts=[120, 0])
            reordered, _ = self.detect(prepared, telescopes=[1, 2], channels=[1, 2], starts=[0, 120])
            alone, _ = self.detect(prepared, telescopes=[2], channels=[2], starts=[120])
        pd.testing.assert_frame_equal(first, reordered, check_exact=True)
        selected = first[(first.telescope_no == 2) & (first.energy_channel_index == 2)
                         & (first.window_start_minute == 120)].reset_index(drop=True)
        pd.testing.assert_frame_equal(selected, alone, check_exact=True)
        self.assertEqual(first.window_seed.nunique(), len(first))
        self.assertTrue(((first.window_seed >= 0) & (first.window_seed < 2 ** 32)).all())

    def test_empty_candidates_keep_typed_schema_and_readable_csv_headers(self):
        empty = pd.DataFrame(columns=["freq", "amp", "phase", "period"])
        with patch.object(detection, "clean_window", return_value=empty):
            table, stats = self.detect(prepared_day())
        self.assertTrue(table.empty)
        self.assertEqual(stats["candidates"], 0)
        self.assertEqual(stats["windows_processed"], 1)
        self.assertEqual(list(table), detection.COLUMNS)
        required = {"date", "satellite", "telescope_no", "energy_channel", "t1_mlt_echo", "t2_mlt_echo",
                    "date1_utc_echo", "date2_utc_echo", "amp", "phase", "max_echo_cand_frq",
                    "max_echo_cand_per", "expected_drift_per", "energy_keV_from_period",
                    "median_pitch_angle_deg", "est_drift_per_a0", "est_drift_per_a90",
                    "est_drift_per_a0_adj", "est_drift_per_a90_adj", "dec_year",
                    "species", "energy_channel_index", "window_start_minute", "window_seed"}
        self.assertTrue(required.issubset(table.columns))
        for name in detection.DATE_COLUMNS:
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(table[name]), name)
        for name in detection.INTEGER_COLUMNS:
            self.assertTrue(pd.api.types.is_integer_dtype(table[name]), name)
        self.assertTrue(pd.api.types.is_float_dtype(table.max_echo_cand_per))
        restored = detection.load_candidates(io.StringIO(table.to_csv(index=False)))
        pd.testing.assert_frame_equal(table, restored)


if __name__ == "__main__":
    unittest.main()
