"""Selection stage persistence, provenance, and bounded-run separation."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drift_echo_climatology import classification, detection


class ClassificationCLI(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.candidate_dir = self.root / "candidates"
        self.output_dir = self.root / "selected"
        self.source = self.candidate_dir / "g16_d20190616.csv"
        self.output = self.output_dir / self.source.name
        rows = []
        for channel, period in enumerate((80., 60., 40., 20.), 1):
            row = {name: 0. for name in detection.COLUMNS}
            row.update(species="fedu", date="2019-06-16", satellite="goes16",
                       telescope_no=2, energy_channel_index=channel, energy_channel=100. * channel,
                       window_start_minute=0, window_seed=channel, date1_utc_echo="2019-06-16T00:00:00",
                       date2_utc_echo="2019-06-16T07:59:00", max_echo_cand_per=period,
                       max_echo_cand_frq=1 / (60 * period), amp=2., expected_drift_per=period,
                       est_drift_per_a0=80., est_drift_per_a90=30.,
                       est_drift_per_a0_adj=100., est_drift_per_a90_adj=10.,
                       median_pitch_angle_deg=90., energy_keV_from_period=100. * channel,
                       dec_year=2019.455)
            rows.append(row)
        self.candidates = detection._typed(pd.DataFrame(rows, columns=detection.COLUMNS))
        self.metadata = {"complete_day": True,
                         "parameters": {"simulations": 5000, "max_iterations": 50, "significance": 0.1,
                                        "species": ["fedu", "fpdu"], "telescopes": None,
                                        "channels": None, "starts": None, "random_seed": 2026},
                         "amplitude_units": {"fedu": "electrons/(cm^2 sr keV s)"}}
        detection.save_candidates(self.source, self.candidates, self.metadata)
        self.arguments = ["--start", "2019-06-16", "--end", "2019-06-16",
                          "--candidate-dir", str(self.candidate_dir), "--output-dir", str(self.output_dir)]

    def run_cli(self, arguments=None):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return classification.main(self.arguments if arguments is None else arguments, root=self.root)

    def test_real_selection_round_trip_and_reuse_preserve_window_and_components(self):
        self.assertEqual(self.run_cli(), 0)
        table = classification.load_selected(self.output)
        pd.testing.assert_frame_equal(table[detection.COLUMNS], self.candidates)
        self.assertEqual(table.category.unique().tolist(), ["valid"])
        self.assertEqual(table.event_id.nunique(), 1)
        self.assertEqual(table.sequence_length.tolist(), [4] * 4)
        before = [p.stat().st_mtime_ns for p in (self.output, self.output.with_suffix(".json"))]
        with patch.object(classification, "select_candidates", side_effect=AssertionError("cache not reused")):
            self.assertEqual(self.run_cli(), 0)
        self.assertEqual(before, [p.stat().st_mtime_ns for p in (self.output, self.output.with_suffix(".json"))])
        record = json.loads(self.output.with_suffix(".json").read_text())
        self.assertTrue(record["complete_day"])
        self.assertEqual(record["statistics"]["events_selected"], 1)
        self.assertEqual(record["detection_provenance"]["parameters"]["random_seed"], 2026)
        self.assertEqual(record["units"]["max_echo_cand_per"], "minutes")

    def test_empty_and_too_short_inputs_produce_typed_empty_tables(self):
        for count in (0, 3):
            with self.subTest(candidates=count):
                detection.save_candidates(self.source, self.candidates.iloc[:count], self.metadata)
                self.assertEqual(self.run_cli(self.arguments + ["--force"]), 0)
                table = classification.load_selected(self.output)
                self.assertTrue(table.empty)
                self.assertEqual(list(table), classification.SELECTED_COLUMNS)
                self.assertEqual(str(table.sequence_length.dtype), "int64")
                self.assertEqual(str(table.date.dtype), "datetime64[ns]")

    def test_partial_input_requires_explicit_separate_destination_and_remains_partial(self):
        self.metadata["complete_day"] = False
        detection.save_candidates(self.source, self.candidates, self.metadata)
        self.assertEqual(self.run_cli(), 1)
        self.assertFalse(self.output_dir.exists())
        self.assertEqual(self.run_cli(self.arguments + ["--allow-partial"]), 0)
        self.assertFalse(json.loads(self.output.with_suffix(".json").read_text())["complete_day"])
        with self.assertRaises(SystemExit):
            self.run_cli(["--candidate-dir", str(self.candidate_dir), "--allow-partial"])
        self.metadata["complete_day"] = True
        self.metadata["parameters"]["simulations"] = 100
        detection.save_candidates(self.source, self.candidates, self.metadata)
        self.assertEqual(self.run_cli(self.arguments + ["--force"]), 1)

    def test_changed_input_and_metadata_require_force_and_damaged_inputs_are_rejected(self):
        self.assertEqual(self.run_cli(), 0)
        previous = self.output.read_bytes()
        self.candidates.loc[0, "amp"] = 3.
        detection.save_candidates(self.source, self.candidates, self.metadata)
        self.assertEqual(self.run_cli(), 1)
        self.assertEqual(self.output.read_bytes(), previous)
        self.assertEqual(self.run_cli(self.arguments + ["--force"]), 0)
        self.metadata["parameters"]["random_seed"] = 42
        detection.save_candidates(self.source, self.candidates, self.metadata)
        self.assertEqual(self.run_cli(), 1)
        self.assertEqual(self.run_cli(self.arguments + ["--force"]), 0)
        self.source.write_text("damaged candidate data")
        self.assertEqual(self.run_cli(self.arguments + ["--force"]), 1)

    def test_failed_recomputation_and_bad_selected_output_cannot_silently_resume(self):
        self.assertEqual(self.run_cli(), 0)
        previous = {p: p.read_bytes() for p in (self.output, self.output.with_suffix(".json"))}
        with patch.object(classification, "select_candidates", side_effect=RuntimeError("calculation failed")):
            self.assertEqual(self.run_cli(self.arguments + ["--force"]), 1)
        for path, contents in previous.items():
            self.assertEqual(path.read_bytes(), contents)
        self.output.write_text("damaged selected data")
        self.assertEqual(self.run_cli(), 1)
        self.assertEqual(self.run_cli(self.arguments + ["--force"]), 0)
        self.assertFalse(list(self.output_dir.glob(".*.part")))

    def test_bad_units_or_dates_and_same_input_output_directory_are_rejected(self):
        record = json.loads(self.source.with_suffix(".json").read_text())
        record["units"]["expected_drift_per"] = "seconds"
        self.source.with_suffix(".json").write_text(json.dumps(record))
        self.assertEqual(self.run_cli(), 1)
        self.candidates["date"] = pd.Timestamp("2019-06-17")
        detection.save_candidates(self.source, self.candidates, self.metadata)
        self.assertEqual(self.run_cli(), 1)
        with self.assertRaises(SystemExit):
            self.run_cli(["--candidate-dir", str(self.candidate_dir), "--output-dir", str(self.candidate_dir)])

    def test_no_inputs_does_not_create_selected_directory(self):
        self.assertEqual(self.run_cli(["--candidate-dir", str(self.root / "missing"),
                                       "--output-dir", str(self.output_dir)]), 1)
        self.assertFalse(self.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
