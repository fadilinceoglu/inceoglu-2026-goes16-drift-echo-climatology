"""Geometry definitions and file alignment without requiring raw archives."""

from datetime import datetime
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from netCDF4 import Dataset
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drift_echo_climatology import geometry


class Geometry(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def fixture(self, product="mag", offsets=None, flags=True):
        if offsets is None:
            offsets = np.arange(1440) * 60
        prefix = geometry.PRODUCT_NAMES[product]
        path = self.root / (prefix + "_g16_d20190616_v2-0-2.nc")
        midnight = (datetime(2019, 6, 16) - datetime(2000, 1, 1, 12)).total_seconds()
        with Dataset(path, "w") as dataset:
            dataset.createDimension("time", len(offsets))
            dataset.createDimension("coordinate", 3)
            times = dataset.createVariable("time", "f8", ("time",))
            times.units = geometry.TIME_UNITS
            times[:] = midnight + offsets
            orbit = dataset.createVariable("orbit_llr_geo" if product == "mag" else "geo_llr",
                                           "f8", ("time", "coordinate"), fill_value=-9999.)
            orbit.units = "degrees, degrees, meters" if product == "mag" else "deg, deg, km"
            orbit[:] = [0., -75., 42164000. if product == "mag" else 42164.]
            if product == "mag":
                field = dataset.createVariable("b_brf", "f4", ("time", "coordinate"),
                                               fill_value=-9999.)
                field.units = "nT"
                field[:] = [0., 0., 1.]
                if len(offsets) > 3:
                    field[2, :] = -9999.
                if flags:
                    dqf = dataset.createVariable("DQF", "u4", ("time",))
                    dqf.flag_meanings = "other_qf " + geometry.ARCJET_FLAG
                    dqf.flag_masks = np.array([1, 2], dtype="uint32")
                    dqf.flag_values = np.array([1, 2], dtype="uint32")
                    dqf[:] = 0
                    dqf[0] = 1  # non-arcjet quality bit is not an added veto
                    if len(offsets) > 1:
                        dqf[1] = 3  # combined bits still mask the arcjet row
        return path

    def test_telescope_order_and_field_reversal(self):
        forward = np.array([[0., 0., 1.]])
        for species, expected in (("fedu", [35., 35., 70., 0., 70.]),
                                  ("fpdu", [70., 0., 70., 35., 35.])):
            actual = geometry.pitch_angles(forward, species)
            np.testing.assert_allclose(actual[0], expected, atol=1e-12)
            np.testing.assert_allclose(geometry.pitch_angles(-forward, species), 180. - actual,
                                       atol=1e-12)
        self.assertTrue(np.isnan(geometry.pitch_angles(np.zeros((1, 3)), "fedu")).all())

    def test_mag_fill_arcjet_and_daily_alignment(self):
        offsets = np.delete(np.arange(1440) * 60, 4)
        path = self.fixture(offsets=offsets)
        # Coordinate arithmetic is tested separately; this isolates row identities.
        with patch.object(geometry, "_llr_to_mlt", return_value=np.arange(1440, dtype=float)):
            result = geometry.read_geometry(path, "mag")
        self.assertEqual(result["j2000"].shape, (1440,))
        np.testing.assert_array_equal(np.diff(result["j2000"]), np.full(1439, 60.))
        for key in ("pitch_fedu", "pitch_fpdu"):
            self.assertEqual(result[key].shape, (1440, 5))
            self.assertTrue(np.isfinite(result[key][0]).all())
            self.assertTrue(np.isnan(result[key][[1, 2, 4]]).all())
            self.assertTrue(np.isfinite(result[key][3]).all())

    def test_ephemeris_conversion_and_unavailable_pitch(self):
        path = self.fixture("ephe")
        with patch.object(geometry, "_llr_to_mlt", return_value=np.full(1440, 12.)) as convert:
            result = geometry.read_geometry(path, "ephe")
        np.testing.assert_array_equal(convert.call_args[0][0][:, 2], np.full(1440, 42164000.))
        self.assertTrue(np.isnan(result["pitch_fedu"]).all())
        self.assertTrue(np.isnan(result["pitch_fpdu"]).all())

    def test_missing_dqf_retains_original_warning_policy(self):
        with patch.object(geometry, "_llr_to_mlt", return_value=np.zeros(1440)):
            with self.assertWarnsRegex(RuntimeWarning, "arcjet masking unavailable"):
                result = geometry.read_geometry(self.fixture(flags=False), "mag")
        self.assertTrue(np.isfinite(result["pitch_fedu"][1]).all())

    def test_early_mag_underscore_version_and_prefixes_are_supported(self):
        path = self.fixture()
        early = path.with_name(path.name.replace("v2-0-2", "v2_0_0"))
        path.rename(early)
        with patch.object(geometry, "_llr_to_mlt", return_value=np.zeros(1440)):
            result = geometry.read_geometry(early, "mag")
        self.assertEqual(result["source"]["version"], "2_0_0")
        science = early.with_name(early.name.replace("dn_", "sci_", 1))
        early.rename(science)
        with patch.object(geometry, "_llr_to_mlt", return_value=np.zeros(1440)):
            self.assertEqual(geometry.read_geometry(science, "mag")["source"]["filename"],
                             science.name)

    def test_orbit_interpolation_does_not_fill_long_gaps(self):
        grid = geometry.pd.date_range("2019-06-16", periods=1440, freq="min")
        orbit = np.tile([0., -75., 42164000.], (1440, 1))
        orbit[10:16] = np.nan
        cleaned = geometry._daily_orbit(orbit, grid, grid)
        self.assertTrue(np.isnan(cleaned[10:16]).all())
        orbit[15] = [0., -75., 42164000.]
        self.assertTrue(np.isfinite(geometry._daily_orbit(orbit, grid, grid)).all())

    def test_all_missing_orbit_has_no_fabricated_mlt(self):
        grid = geometry.pd.date_range("2019-06-16", periods=1440, freq="min")
        result = geometry._llr_to_mlt(np.full((1440, 3), np.nan), grid)
        self.assertTrue(np.isnan(result).all())


if __name__ == "__main__":
    unittest.main()
