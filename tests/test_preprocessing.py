"""Padding, conditioning and restart checks using small numerical fixtures."""

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from netCDF4 import Dataset
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drift_echo_climatology import preprocessing as prep


class Preprocessing(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def mpsh(self, day, offsets=None, directory=None):
        directory = self.root if directory is None else directory
        directory.mkdir(parents=True, exist_ok=True)
        day = datetime.fromisoformat(day)
        offsets = np.arange(1440) * 60 if offsets is None else np.asarray(offsets)
        path = directory / f"sci_mpsh-l2-avg1m_g16_d{day:%Y%m%d}_v2-0-3.nc"
        midnight = (day - datetime(2000, 1, 1, 12)).total_seconds()
        with Dataset(path, "w") as dataset:
            for name, size in (("time", len(offsets)), ("telescope", 5), ("channel", 1)):
                dataset.createDimension(name, size)
            time = dataset.createVariable("time", "f8", ("time",))
            time.units = "seconds since 2000-01-01 12:00:00 UTC"
            time[:] = midnight + offsets
            for species in ("Electron", "Proton"):
                flux = dataset.createVariable("AvgDiff" + species + "Flux", "f4",
                                              ("time", "telescope", "channel"))
                flux.units = "particles/(cm^2 sr keV s)"
                flux[:] = np.full((len(offsets), 5, 1), day.day, dtype="float32")
                energies = dataset.createVariable("Diff" + species + "EffectiveEnergy", "f4",
                                                  ("telescope", "channel"))
                energies.units = "keV"
                energies[:] = 76.2265625
        return path

    def geometry(self, day, product):
        day = datetime.fromisoformat(str(day))
        midnight = (day - datetime(2000, 1, 1, 12)).total_seconds()
        return {"j2000": midnight + np.arange(1440) * 60.,
                "mlt": np.arange(1440) / 60.,
                "pitch_fedu": np.full((1440, 5), day.day, dtype=float),
                "pitch_fpdu": np.full((1440, 5), day.day, dtype=float)}

    def prepare(self, day="2019-06-16", following=None, offsets=None):
        inputs = {"mpsh_path": self.mpsh(day), "mag_path": day}
        if following:
            inputs.update(next_mpsh_path=self.mpsh(following, offsets), next_mag_path=following)
        with patch.object(prep, "read_geometry", side_effect=self.geometry):
            return prep.prepare_day(**inputs)

    def test_padding_uses_next_calendar_day_minutes_and_keeps_gaps(self):
        result = self.prepare(following="2019-06-17", offsets=[0, 60, 479 * 60, 480 * 60])
        self.assertEqual(result["j2000"].size, 1920)
        np.testing.assert_array_equal(result["fedu"][:1440], 16.)
        np.testing.assert_array_equal(result["fedu"][[1440, 1441, 1919]], 17.)
        self.assertTrue(np.isnan(result["fedu"][1442:1919]).all())
        self.assertTrue(np.isnan(result["fedu_hp"][1442:1919]).all())
        np.testing.assert_array_equal(result["pitch_fedu"][1440:], 17.)
        np.testing.assert_array_equal(result["fedu_cutoff_minutes"], 180.)

    def test_padding_cannot_jump_over_a_missing_day(self):
        with self.assertRaisesRegex(ValueError, "next calendar day"):
            self.prepare(following="2019-06-18")

    def test_invalid_next_day_can_still_prepare_the_primary_day(self):
        source = self.mpsh("2019-06-16")
        broken = self.root / "sci_mpsh-l2-avg1m_g16_d20190617_v2-0-3.nc"
        broken.write_bytes(b"broken netcdf")
        with patch.object(prep, "read_geometry", side_effect=self.geometry), redirect_stderr(io.StringIO()):
            result = prep.prepare_day(source, next_mpsh_path=broken, mag_path="2019-06-16")
        self.assertEqual(result["j2000"].size, 1440)

    def test_wrong_day_primary_geometry_is_rejected(self):
        with patch.object(prep, "read_geometry", side_effect=self.geometry):
            with self.assertRaisesRegex(ValueError, "Primary geometry"):
                prep.prepare_day(self.mpsh("2019-06-16"), mag_path="2019-06-17")

    def test_no_padding_and_safe_file_round_trip(self):
        result = self.prepare()
        self.assertEqual(result["j2000"].size, 1440)
        self.assertEqual(int(result["padding_minutes"]), 0)
        destination = self.root / "prepared.npz"
        prep.save_prepared(destination, result)
        loaded = prep.load_prepared(destination)
        for name in result:
            np.testing.assert_array_equal(loaded[name], result[name])
        self.assertFalse(list(self.root.glob(".*.part")))

    def test_high_pass_retains_short_period_and_removes_long_period(self):
        minutes = np.arange(1440)
        for period, minimum, maximum in ((20, 0.99, 1.01), (300, 0., 0.1)):
            wave = np.sin(2 * np.pi * minutes / period)
            filtered = prep.high_pass(wave, 180.)
            amplitude = 2 * np.mean(filtered[600:1200] * wave[600:1200])
            self.assertGreaterEqual(amplitude, minimum)
            self.assertLessEqual(amplitude, maximum)
        wave[700] = np.nan
        self.assertTrue(np.isnan(prep.high_pass(wave, 180.)[700]))

    def test_commissioning_substitution_follows_anchor_day_into_padding(self):
        result = self.prepare("2017-04-11", following="2017-04-12")
        climate = prep.load_pitch_climatology()
        for species in prep.SPECIES:
            np.testing.assert_array_equal(result["pitch_" + species][:1440], climate[species])
            np.testing.assert_array_equal(result["pitch_" + species][1440:], climate[species][:480])

    def test_ephemeris_fallback_keeps_missing_pitch_after_commissioning(self):
        def ephemeris(day, product):
            result = self.geometry(day, product)
            result["pitch_fedu"][:] = np.nan
            result["pitch_fpdu"][:] = np.nan
            return result
        with patch.object(prep, "read_geometry", side_effect=ephemeris):
            result = prep.prepare_day(self.mpsh("2019-06-16"), ephe_path="2019-06-16")
        self.assertTrue(np.isfinite(result["mlt"]).all())
        self.assertTrue(np.isnan(result["pitch_fedu"]).all())
        self.assertEqual(str(result["geometry_source"]), "ephe")

    def test_commissioning_asset_is_included_in_resume_identity(self):
        source = self.mpsh("2017-04-11")
        with patch.object(prep, "ASSET_DIR", self.root):
            asset = self.root / "goes16_pitch_climatology.npz"
            asset.write_bytes(b"first model")
            first = prep._provenance({"mpsh_path": source})
            asset.write_bytes(b"changed model")
            self.assertNotEqual(prep._provenance({"mpsh_path": source}), first)

    def test_cli_resumes_valid_outputs_and_refuses_changed_sources(self):
        raw = self.root / "raw"
        source = self.mpsh("2019-06-16", directory=raw / "GOES16" / "mpsh")
        prepared = self.prepare()
        destination = self.root / "processed"
        arguments = ["--start", "2019-06-16", "--end", "2019-06-16",
                     "--data-dir", str(raw), "--output-dir", str(destination)]
        with patch.dict(prep.os.environ, {}), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with patch.object(prep, "prepare_day", return_value=prepared) as process:
                self.assertEqual(prep.main(arguments, root=self.root), 0)
                written = destination / "g16_d20190616.npz"
                modified = written.stat().st_mtime_ns
                self.assertEqual(prep.main(arguments, root=self.root), 0)
                self.assertEqual(process.call_count, 1)
                self.assertEqual(written.stat().st_mtime_ns, modified)
                with Dataset(source, "r+") as dataset:
                    dataset["AvgDiffElectronFlux"][0, 0, 0] = 99.
                self.assertEqual(prep.main(arguments, root=self.root), 1)
                self.assertEqual(written.stat().st_mtime_ns, modified)
                self.assertEqual(prep.main(arguments + ["--force"], root=self.root), 0)
                self.assertEqual(process.call_count, 2)


if __name__ == "__main__":
    unittest.main()
