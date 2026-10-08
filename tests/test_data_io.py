"""Small source-file fixtures exercising the scientific input contract."""

from datetime import date, datetime
from pathlib import Path
import sys
import tempfile
import unittest

import cdflib
from cdflib import cdfwrite
from netCDF4 import Dataset
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drift_echo_climatology import io as data_io


class InputReaders(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def mpsh(self, version="2-0-3", time_units=data_io.TIME_UNITS, offsets=(0, 60), energy=100):
        path = self.root / ("sci_mpsh-l2-avg1m_g16_d20190616_v" + version + ".nc")
        epoch = datetime(2000, 1, 1, 12)
        seconds = (datetime(2019, 6, 16) - epoch).total_seconds()
        with Dataset(path, "w") as dataset:
            for name, size in (("time", len(offsets)), ("telescope", 5), ("channel", 2)):
                dataset.createDimension(name, size)
            times = dataset.createVariable("time", "f8", ("time",))
            times.units = time_units
            times[:] = seconds + np.asarray(offsets)
            for species in ("Electron", "Proton"):
                flux = dataset.createVariable("AvgDiff" + species + "Flux", "f4",
                                              ("time", "telescope", "channel"))
                flux.units = species.lower() + "s/(cm^2 sr keV s)"
                flux[:] = np.float32(1.234567)
                flux[0, 0, 0] = -1e31
                energies = dataset.createVariable("Diff" + species + "EffectiveEnergy", "f4",
                                                  ("telescope", "channel"))
                energies.units = "keV"
                energies[:] = energy
        return path

    def omni(self, include_fill=True):
        path = self.root / "omni.cdf"
        epochs = cdflib.cdfepoch.compute_epoch(
            [[2019, 6, 16, 0, minute, 0, 0] for minute in range(6)])
        values = [-80, 99.99, 9999.99, 101, np.nan, -100]
        with cdfwrite.CDF(path, delete=True) as cdf:
            for name in ("Epoch", "BX_GSE", "BY_GSM", "BZ_GSM"):
                specification = dict(Variable=name, Num_Elements=1, Rec_Vary=True, Dim_Sizes=[],
                                     Data_Type=cdfwrite.CDF.CDF_EPOCH if name == "Epoch"
                                     else cdfwrite.CDF.CDF_DOUBLE)
                attrs = {"VALIDMIN": -100.0, "VALIDMAX": 100.0}
                if include_fill:
                    attrs["FILLVAL"] = 9999.99
                cdf.write_var(specification, var_attrs={} if name == "Epoch" else attrs,
                              var_data=epochs if name == "Epoch" else values)
        return path

    def test_flux_precision_missing_values_and_embedded_energies(self):
        for version, energy in (("2-0-2", 99.1362686), ("2-0-3", 98.7927094)):
            with self.subTest(version=version):
                result = data_io.read_mpsh(self.mpsh(version, energy=energy))
                for species in ("fedu", "fpdu"):
                    self.assertEqual(result[species].dtype, np.dtype("float32"))
                    self.assertTrue(np.isnan(result[species][0, 0, 0]))
                    self.assertEqual(result[species][1, 0, 0], np.float32(1.234567))
                    np.testing.assert_array_equal(result[species + "_energies"],
                                                  np.full((5, 2), energy, dtype="float32"))
                self.assertEqual(data_io.j2000_to_datetime(result["j2000"])[0],
                                 datetime(2019, 6, 16))

    def test_unsupported_v1_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "v2 inputs only"):
            data_io.read_mpsh(self.mpsh("1-0-2"))

    def test_timestamp_contract_allows_gaps_and_rejects_wrong_metadata(self):
        self.assertEqual(len(data_io.read_mpsh(self.mpsh(offsets=(0, 120)))["j2000"]), 2)
        for offsets in ((60, 0), (0, 30), (0, 86400)):
            with self.subTest(offsets=offsets), self.assertRaisesRegex(ValueError, "UTC minute grid"):
                data_io.read_mpsh(self.mpsh(offsets=offsets))
        with self.assertRaisesRegex(ValueError, "timestamps in"):
            data_io.read_mpsh(self.mpsh(time_units="days since 1970-01-01"))

    def test_invalid_effective_energies_are_rejected(self):
        for energy in (0, -9999, np.nan):
            with self.subTest(energy=energy), self.assertRaisesRegex(ValueError, "finite and positive"):
                data_io.read_mpsh(self.mpsh(energy=energy))

    def test_climatology_is_safe_and_commissioning_dates_are_explicit(self):
        arrays = data_io.load_pitch_climatology()
        self.assertEqual(set(arrays), {"fedu", "fpdu"})
        for array in arrays.values():
            self.assertEqual(array.shape, (1440, 5))
            self.assertEqual(array.dtype, np.dtype("float64"))
            self.assertTrue(np.isfinite(array).all())
        for day, expected in ((date(2016, 12, 31), False), (date(2017, 1, 1), True),
                              (date(2017, 4, 11), True), (date(2017, 4, 12), False)):
            self.assertEqual(data_io.uses_pitch_climatology(day), expected)

    def test_omni_masks_declared_invalid_data_and_retains_valid_disturbances(self):
        result = data_io.read_omni_imf(self.omni())
        expected = [-80, 99.99, np.nan, np.nan, np.nan, -100]
        for name in ("BX_GSE", "BY_GSM", "BZ_GSM"):
            np.testing.assert_allclose(result[name], expected, equal_nan=True)
        self.assertEqual(result["time"].iloc[0], datetime(2019, 6, 16))

    def test_omni_requires_fill_metadata(self):
        with self.assertRaisesRegex(ValueError, "FILLVAL"):
            data_io.read_omni_imf(self.omni(include_fill=False))


if __name__ == "__main__":
    unittest.main()
