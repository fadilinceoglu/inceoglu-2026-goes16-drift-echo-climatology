"""IMF window medians, monthly coverage, and the checked plotting catalog."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drift_echo_climatology import catalog, classification, detection, selection


def selected_day():
    rows = []
    for species in ("fedu", "fpdu"):
        for channel, period in enumerate((80., 60., 40., 20.), 1):
            row = {name: 0. for name in detection.COLUMNS}
            row.update(species=species, date="2019-06-16", satellite="goes16", telescope_no=1,
                       energy_channel_index=channel, energy_channel=channel * 100., window_seed=123,
                       date1_utc_echo="2019-06-16T00:00:00", date2_utc_echo="2019-06-16T07:59:00",
                       max_echo_cand_per=period, max_echo_cand_frq=1 / (60 * period), amp=1.,
                       est_drift_per_a90_adj=0., est_drift_per_a0_adj=100., expected_drift_per=period)
            rows.append(row)
    return selection.select_candidates(pd.DataFrame(rows, columns=detection.COLUMNS))[0]


def omni_day(day="2019-06-16", periods=480):
    return pd.DataFrame({"time": pd.date_range(day, periods=periods, freq="min"),
                         "BX_GSE": np.ones(periods), "BY_GSM": np.ones(periods),
                         "BZ_GSM": np.ones(periods)})


class IMFWindows(unittest.TestCase):
    def test_inclusive_components_then_angle_and_shared_window_alignment(self):
        omni = omni_day(periods=4)
        omni["BX_GSE"] = [5., np.nan, 1., 2.]
        omni["BY_GSM"] = [-1., 2., 9., 0.]
        omni["BZ_GSM"] = [2., 1., -4., 3.]
        windows = pd.DataFrame({"date1_utc_echo": [omni.time[0], omni.time[0], omni.time[1]],
                                "date2_utc_echo": [omni.time[2], omni.time[2], omni.time[3]]}, index=[8, 3, 7])
        result = catalog.window_imf_medians(windows, omni)
        self.assertEqual(result.index.to_list(), [8, 3, 7])
        self.assertEqual(result.imf_samples.to_list(), [3, 3, 3])
        self.assertEqual(result.loc[8, "BX_GSE_median"], 3.)
        self.assertEqual(result.loc[8, "BY_GSM_median"], 2.)
        self.assertEqual(result.loc[8, "BZ_GSM_median"], 1.)
        self.assertEqual(result.loc[8, "clock_angle_deg_median"], np.mod(np.degrees(np.arctan2(2., 1.)), 360))
        pd.testing.assert_series_equal(result.loc[8], result.loc[3], check_names=False)

    def test_independent_finite_components_and_original_no_data_gate(self):
        omni = omni_day(periods=3)
        omni["BX_GSE"], omni["BY_GSM"], omni["BZ_GSM"] = [1., np.nan, np.nan], [np.nan, 1., np.nan], [np.nan, np.nan, 1.]
        windows = pd.DataFrame({"date1_utc_echo": [omni.time[0]], "date2_utc_echo": [omni.time[2]]})
        result = catalog.window_imf_medians(windows, omni)
        self.assertEqual(result.clock_angle_deg_median.iloc[0], 45.)
        self.assertEqual(result.imf_samples.iloc[0], 3)
        omni["BX_GSE"] = np.nan
        result = catalog.window_imf_medians(windows, omni)
        self.assertTrue(result[catalog.MEDIAN_COLUMNS + ["clock_angle_deg_median"]].isna().all().all())
        self.assertEqual(result.imf_samples.iloc[0], 0)

    def test_invalid_time_grid_and_endpoints_fail_but_missing_minutes_are_counted(self):
        windows = pd.DataFrame({"date1_utc_echo": [pd.Timestamp("2019-06-16")],
                                "date2_utc_echo": [pd.Timestamp("2019-06-16T00:02:00")]})
        omni = omni_day(periods=3)
        self.assertEqual(catalog.window_imf_medians(windows, omni.iloc[[0, 2]]).imf_samples.iloc[0], 2)
        for bad in (omni.iloc[[0, 0]], omni.iloc[::-1], omni.assign(time=omni.time + pd.Timedelta(seconds=1))):
            with self.assertRaises(ValueError):
                catalog.window_imf_medians(windows, bad)
        windows["date2_utc_echo"] = pd.NaT
        with self.assertRaises(ValueError):
            catalog.window_imf_medians(windows, omni)

    def test_global_cleaning_needs_all_study_months_even_for_subset_windows(self):
        windows = pd.DataFrame({"date1_utc_echo": pd.to_datetime(["2019-06-30T22:00", "2025-04-30T00:00"]),
                                "date2_utc_echo": pd.to_datetime(["2019-07-01T05:59", "2025-05-01T07:59"])})
        months = catalog._needed_months(windows)
        self.assertEqual(len(months), 100)
        self.assertEqual(months[0].isoformat(), "2017-01-01")
        self.assertEqual(months[-1].isoformat(), "2025-04-01")
        self.assertEqual(catalog._needed_months(windows.iloc[:0]), [])


class PlotCatalog(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.selected_dir = self.root / "selected"
        self.omni_dir = self.root / "omni"
        self.omni_dir.mkdir()
        self.source = self.selected_dir / "g16_d20190616.csv"
        self.output = self.root / "catalog.csv"
        self.frame = selected_day()
        for month in catalog._needed_months(self.frame):
            (self.omni_dir / ("omni_hro2_1min_" + month.strftime("%Y%m%d") + "_v01.cdf")).write_text("trusted test source")
        self.month = self.omni_dir / "omni_hro2_1min_20190601_v01.cdf"
        self.metadata = {"complete_day": True, "units": {name: "minutes" for name in catalog.PERIOD_COLUMNS},
                         "amplitude_units": {"fedu": "electrons", "fpdu": "protons"}}
        classification.save_selected(self.source, self.frame, self.metadata)
        self.arguments = ["--start", "2019-06-16", "--end", "2019-06-16",
                          "--selected-dir", str(self.selected_dir), "--omni-dir", str(self.omni_dir),
                          "--output", str(self.output)]

    def run_cli(self, arguments=None, omni=None):
        def read_month(path):
            if path.name.startswith("omni_hro2_1min_20190601_"):
                return omni_day() if omni is None else omni
            month = path.name.split("_")[3]
            return omni_day(day=month[:4] + "-" + month[4:6] + "-01", periods=1)

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), patch.object(
                catalog, "read_omni_imf", side_effect=read_month):
            return catalog.main(self.arguments if arguments is None else arguments, root=self.root)

    def test_real_core_round_trip_keeps_rows_events_and_small_schema(self):
        self.assertEqual(self.run_cli(), 0)
        result = catalog.load_catalog(self.output)
        pd.testing.assert_frame_equal(result[selection.SELECTED_COLUMNS], self.frame, check_exact=True)
        self.assertEqual(list(result), catalog.CATALOG_COLUMNS)
        self.assertEqual(result.event_id.nunique(), 2)
        self.assertTrue(result.clock_angle_deg_median.eq(45.).all())
        self.assertTrue(result.imf_samples.eq(480).all())
        self.assertTrue(result.attrs["provenance"]["complete_detection_days"])
        self.assertEqual(result.attrs["provenance"]["rows"], 8)
        self.assertEqual(len(result.attrs["provenance"]["omni_sources"]), 100)
        self.assertFalse(list(self.root.glob(".*.part")))

    def test_nonvalid_and_empty_inputs_keep_typed_empty_catalog_without_omni(self):
        self.frame["category"] = "edge"
        classification.save_selected(self.source, self.frame, self.metadata)
        self.month.unlink()
        self.assertEqual(self.run_cli(), 0)
        result = catalog.load_catalog(self.output)
        self.assertTrue(result.empty)
        self.assertEqual(str(result.date.dtype), "datetime64[ns]")
        self.assertEqual(str(result.imf_samples.dtype), "int64")
        self.assertEqual(str(result.clock_angle_deg_median.dtype), "float64")

    def test_numeric_latest_month_version_and_required_months(self):
        newer = self.omni_dir / "omni_hro2_1min_20190601_v10.cdf"
        newer.write_text("latest")
        (self.omni_dir / "omni_hro2_1min_20190601_v2.cdf").write_text("older")
        with patch.object(catalog, "read_omni_imf", return_value=omni_day()) as reader:
            _, sources = catalog._read_months(self.omni_dir, [pd.Timestamp("2019-06-01").date()])
        self.assertEqual(reader.call_args.args[0], newer)
        self.assertEqual(sources[0]["version"], 10)
        (self.omni_dir / "omni_hro2_1min_20190701_v01.cdf").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "2019-07-01"):
            catalog._read_months(self.omni_dir, [pd.Timestamp("2019-07-01").date()])

    def test_iqr_is_applied_once_after_months_are_combined(self):
        january = omni_day(day="2017-01-01", periods=129)
        february = omni_day(day="2017-02-01", periods=128)
        january["BZ_GSM"] = np.r_[np.tile([-1., 1.], 64), 40.]
        february["BZ_GSM"] = np.tile([-50., 50.], 64)
        self.assertTrue(np.isnan(catalog.clean_omni_imf(january)["BZ_GSM"].iloc[-1]))
        with patch.object(catalog, "read_omni_imf", side_effect=[january, february]):
            result, _ = catalog._read_months(self.omni_dir,
                [pd.Timestamp("2017-01-01").date(), pd.Timestamp("2017-02-01").date()])
        self.assertEqual(result["BZ_GSM"].iloc[128], 40.)

    def test_missing_or_invalid_sources_preserve_previous_catalog(self):
        self.assertEqual(self.run_cli(), 0)
        previous = {path: path.read_bytes() for path in (self.output, self.output.with_suffix(".json"))}
        self.month.unlink()
        self.assertEqual(self.run_cli(), 1)
        self.source.write_text("damaged selected table")
        self.assertEqual(self.run_cli(), 1)
        for path, value in previous.items():
            self.assertEqual(path.read_bytes(), value)

    def test_partial_input_needs_output_and_retains_partial_provenance(self):
        self.metadata["complete_day"] = False
        classification.save_selected(self.source, self.frame, self.metadata)
        self.assertEqual(self.run_cli(), 1)
        self.assertEqual(self.run_cli(self.arguments + ["--allow-partial"]), 0)
        self.assertFalse(catalog.load_catalog(self.output).attrs["provenance"]["complete_detection_days"])
        with self.assertRaises(SystemExit):
            self.run_cli(["--selected-dir", str(self.selected_dir), "--allow-partial"])

    def test_reader_checks_hash_units_and_shared_window_values(self):
        self.assertEqual(self.run_cli(), 0)
        self.output.write_text("damaged catalog")
        with self.assertRaises(ValueError):
            catalog.load_catalog(self.output)
        self.assertEqual(self.run_cli(), 0)
        metadata = json.loads(self.output.with_suffix(".json").read_text())
        metadata["units"]["clock_angle_deg_median"] = "radians"
        self.output.with_suffix(".json").write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
            catalog.load_catalog(self.output)
        self.assertEqual(self.run_cli(), 0)
        result = catalog.load_catalog(self.output)
        result.loc[0, "clock_angle_deg_median"] = 90.
        with self.assertRaisesRegex(ValueError, "Shared UTC windows"):
            catalog._typed_catalog(result)

    def test_exact_global_time_cutoff_and_no_inputs(self):
        path = self.omni_dir / "omni_hro2_1min_20250401_v01.cdf"
        path.write_text("source")
        omni = pd.DataFrame({"time": pd.to_datetime(["2025-04-29T23:59", "2025-04-30T00:00", "2025-04-30T00:01"]),
                             "BX_GSE": [1.] * 3, "BY_GSM": [1.] * 3, "BZ_GSM": [1.] * 3})
        with patch.object(catalog, "read_omni_imf", return_value=omni):
            combined, _ = catalog._read_months(self.omni_dir, [pd.Timestamp("2025-04-01").date()])
        self.assertEqual(combined.time.to_list(), omni.time.iloc[:2].to_list())
        missing_output = self.root / "missing" / "catalog.csv"
        self.assertEqual(self.run_cli(["--selected-dir", str(self.root / "unknown"), "--output", str(missing_output)]), 1)
        self.assertFalse(missing_output.parent.exists())


if __name__ == "__main__":
    unittest.main()
