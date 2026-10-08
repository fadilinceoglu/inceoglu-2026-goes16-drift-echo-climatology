"""Candidate persistence, empty days, and safe resumption of detection."""

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
from drift_echo_climatology import detection
from drift_echo_climatology.preprocessing import save_prepared


class DetectionCLI(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.prepared_dir = self.root / "prepared"
        self.output_dir = self.root / "candidates"
        self.prepared_path = self.prepared_dir / "g16_d20190616.npz"
        midnight = (pd.Timestamp("2019-06-16") - pd.Timestamp("2000-01-01T12:00:00")).total_seconds()
        self.prepared = {"j2000": midnight + np.arange(1440) * 60., "date": np.array("2019-06-16"),
                         "schema_version": np.array(1), "padding_minutes": np.array(0),
                         "mlt": np.zeros(1440)}
        for species in ("fedu", "fpdu"):
            for suffix in ("", "_hp"):
                self.prepared[species + suffix] = np.zeros((1440, 5, 1))
            self.prepared["pitch_" + species] = np.full((1440, 5), 90.)
            for suffix, value in (("_energies", 100.), ("_cutoff_minutes", 180.),
                                  ("_drift_seconds_0", 3600.), ("_drift_seconds_90", 2400.)):
                self.prepared[species + suffix] = np.full((5, 1), value)
        save_prepared(self.prepared_path, self.prepared)
        self.arguments = ["--start", "2019-06-16", "--end", "2019-06-16",
                          "--prepared-dir", str(self.prepared_dir), "--output-dir", str(self.output_dir)]
        self.empty = detection._typed(pd.DataFrame(columns=detection.COLUMNS))
        self.stats = {"windows_total": 0, "windows_processed": 0, "windows_skipped_missing_flux": 0,
                      "candidates": 0}
        self.csv = self.output_dir / "g16_d20190616.csv"

    def run_cli(self, arguments=None):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return detection.main(self.arguments if arguments is None else arguments, root=self.root)

    def test_empty_day_is_written_with_complete_schema_and_reused(self):
        with patch.object(detection, "detect_day", return_value=(self.empty, self.stats)) as process:
            self.assertEqual(self.run_cli(), 0)
            modified = self.csv.stat().st_mtime_ns
            self.assertEqual(self.run_cli(), 0)
            self.assertEqual(process.call_count, 1)
            self.assertEqual(self.csv.stat().st_mtime_ns, modified)
        restored = detection.load_candidates(self.csv)
        pd.testing.assert_frame_equal(restored, self.empty)
        record = json.loads(self.csv.with_suffix(".json").read_text())
        self.assertTrue(record["complete_day"])
        self.assertEqual(record["parameters"]["simulations"], 5000)
        self.assertEqual(record["parameters"]["max_iterations"], 50)
        self.assertEqual(record["parameters"]["random_seed"], 2026)
        self.assertFalse(list(self.output_dir.glob(".*.part")))

    def test_changed_input_and_damaged_output_cannot_resume(self):
        with patch.object(detection, "detect_day", return_value=(self.empty, self.stats)):
            self.assertEqual(self.run_cli(), 0)
            original = self.csv.read_bytes()
            self.prepared["fedu_hp"][0, 0, 0] = 1.
            save_prepared(self.prepared_path, self.prepared)
            self.assertEqual(self.run_cli(), 1)
            self.assertEqual(self.csv.read_bytes(), original)
            self.assertEqual(self.run_cli(self.arguments + ["--force"]), 0)
            self.csv.write_text("broken CSV")
            self.assertEqual(self.run_cli(), 1)

    def test_changed_seed_requires_deliberate_recomputation(self):
        with patch.object(detection, "detect_day", return_value=(self.empty, self.stats)):
            self.assertEqual(self.run_cli(), 0)
            changed = self.arguments + ["--random-seed", "91"]
            self.assertEqual(self.run_cli(changed), 1)
            self.assertEqual(self.run_cli(changed + ["--force"]), 0)
        record = json.loads(self.csv.with_suffix(".json").read_text())
        self.assertEqual(record["parameters"]["random_seed"], 91)

    def test_failure_preserves_previous_candidate_pair(self):
        with patch.object(detection, "detect_day", return_value=(self.empty, self.stats)):
            self.assertEqual(self.run_cli(), 0)
        original = {path: path.read_bytes() for path in (self.csv, self.csv.with_suffix(".json"))}
        with patch.object(detection, "detect_day", side_effect=RuntimeError("calculation failed")):
            self.assertEqual(self.run_cli(self.arguments + ["--force"]), 1)
        for path, content in original.items():
            self.assertEqual(path.read_bytes(), content)

    def test_restricted_runs_are_tagged_and_require_explicit_destination(self):
        with patch.object(detection, "detect_day", return_value=(self.empty, self.stats)):
            self.assertEqual(self.run_cli(self.arguments + ["--species", "fedu", "--window-start", "14"]), 0)
        record = json.loads(self.csv.with_suffix(".json").read_text())
        self.assertFalse(record["complete_day"])
        self.assertEqual(record["parameters"]["starts"], [840])
        arguments = ["--prepared-dir", str(self.prepared_dir), "--species", "fedu"]
        with self.assertRaises(SystemExit):
            self.run_cli(arguments)

    def test_no_inputs_does_not_create_a_candidate_directory(self):
        arguments = ["--prepared-dir", str(self.root / "missing"), "--output-dir", str(self.output_dir)]
        self.assertEqual(self.run_cli(arguments), 1)
        self.assertFalse(self.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
