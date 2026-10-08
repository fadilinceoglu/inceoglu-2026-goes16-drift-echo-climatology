"""Downloader contracts: source selection and safe local file replacement."""

from contextlib import redirect_stderr, redirect_stdout
from datetime import date
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drift_echo_climatology import acquisition as download
PAYLOAD = b"a small mocked science file"


class Response(io.BytesIO):
    def __init__(self, payload):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}


class DownloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.data_dir = Path(temporary.name)
        self.record = {
            "product": "mpsh", "date": "2017-01-08", "source_version": [2, 0, 3],
            "filename": "sci_mpsh-l2-avg1m_g16_d20170108_v2-0-3.nc",
            "url": "https://example.invalid/science.nc",
        }
        self.destination = self.data_dir / "GOES16" / "mpsh" / self.record["filename"]
        self.destination.parent.mkdir(parents=True)

    def acquire(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            download.download_file(self.record, self.data_dir)

    def manifest(self):
        path = self.data_dir / "download_manifest.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_daily_selection_is_numeric_bounded_and_version_two_only(self):
        names = [
            "sci_mpsh-l2-avg1m_g16_d20170108_v2-2-0.nc",
            "sci_mpsh-l2-avg1m_g16_d20170108_v2-10-0.nc",
            "sci_mpsh-l2-avg1m_g16_d20170108_v3-0-0.nc",
            "sci_mpsh-l2-avg1m_g16_d20170109_v2-0-3.nc",
            "sci_mpsh-l2-avg1m_g16_d20170107_v2-0-3.nc",
            "sci_mpsh-l2-avg1m_g16_d20170110_v2-0-3.nc",
            "https://example.com/sci_mpsh-l2-avg1m_g16_d20170108_v2-99-0.nc",
        ]
        html = "".join(f'<a href="{name}">file</a>' for name in names)
        index = download.GOES_ROOT + "mpsh-l2-avg1m_science/2017/01/"
        records = download.select_latest(html, index, "mpsh", date(2017, 1, 8), date(2017, 1, 9))
        self.assertEqual([item["date"] for item in records], ["2017-01-08", "2017-01-09"])
        self.assertEqual(records[0]["source_version"], [2, 10, 0])
        self.assertEqual(records[0]["url"], index + names[1])
        for product, prefix in (("mag", "dn_magn-l2-avg1m"), ("ephe", "dn_ephe-l2-orb1m")):
            with self.subTest(product=product):
                html = f'<a href="{prefix}_g16_d20170108_v0-0-3.nc">file</a>'
                self.assertEqual(len(download.select_latest(html, index, product,
                                                           date(2017, 1, 8), date(2017, 1, 8))), 1)

    def test_omni_selection_keeps_whole_months_overlapping_requested_days(self):
        names = ["omni_hro2_1min_20161201_v01.cdf", "omni_hro2_1min_20170101_v01.cdf",
                 "omni_hro2_1min_20170101_v10.cdf", "omni_hro2_1min_20170201_v01.cdf",
                 "omni_hro2_1min_20170202_v01.cdf", "omni_hro2_1min_20170301_v01.cdf"]
        html = "".join(f'<a href="{name}">file</a>' for name in names)
        records = download.select_latest(html, download.OMNI_ROOT + "2017/", "omni",
                                         date(2017, 1, 15), date(2017, 2, 2))
        self.assertEqual([item["date"] for item in records], ["2017-01-01", "2017-02-01"])
        self.assertEqual(records[0]["source_version"], [10])
        self.assertEqual(list(download.month_starts(date(2024, 12, 31), date(2025, 1, 1))),
                         [date(2024, 12, 1), date(2025, 1, 1)])

    def test_cli_uses_checkout_raw_directory_and_accepts_custom_directory(self):
        root = self.data_dir / "checkout"
        custom = self.data_dir / "custom"
        html = f'<a href="{self.record["filename"]}">file</a>'
        args = ["--start", "2017-01-08", "--end", "2017-01-08", "--products", "mpsh"]
        for extra, expected in (([], root / "data" / "raw"), (["--data-dir", str(custom)], custom)):
            with self.subTest(directory=expected), \
                    patch.object(download, "fetch_listing", return_value=html), \
                    patch.object(download, "download_file") as transfer, \
                    patch.object(download, "urlopen") as network, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(download.main(args + extra, root=root), 0)
                self.assertEqual(transfer.call_args.args[1], expected)
                transfer.assert_called_once()
                network.assert_not_called()
                self.assertFalse(expected.exists())

    def test_valid_local_file_resumes_without_network(self):
        self.destination.write_bytes(PAYLOAD)
        with patch.object(download, "validate_file") as validate, patch.object(download, "urlopen") as network:
            self.acquire()
        validate.assert_called_once_with(self.destination, "mpsh")
        network.assert_not_called()
        self.assertEqual(self.destination.read_bytes(), PAYLOAD)
        self.assertEqual(self.manifest()[0]["status"], "existing_valid")
        self.assertEqual(self.manifest()[0]["sha256"], hashlib.sha256(PAYLOAD).hexdigest())

    def test_failed_replacement_preserves_existing_file_and_removes_parts(self):
        self.destination.write_bytes(b"previous file")
        with patch.object(download, "validate_file", side_effect=ValueError("invalid container")), \
                patch.object(download, "urlopen", side_effect=lambda *a, **k: Response(PAYLOAD)) as network, \
                patch.object(download.time, "sleep"):
            with self.assertRaises(RuntimeError):
                self.acquire()
        self.assertEqual(network.call_count, download.ATTEMPTS)
        self.assertEqual(self.destination.read_bytes(), b"previous file")
        self.assertEqual(list(self.destination.parent.glob(".*.part")), [])
        self.assertFalse((self.data_dir / "download_manifest.jsonl").exists())

    def test_successful_transfer_replaces_after_validation_and_records_digest(self):
        self.destination.write_bytes(b"previous file")
        older = self.destination.parent / "older_version.nc"
        older.write_bytes(b"keep this older version")

        def validate(path, product):
            self.assertEqual(self.destination.read_bytes(), b"previous file")
            if path == self.destination:
                raise ValueError("invalid existing container")
            self.assertEqual(Path(path).read_bytes(), PAYLOAD)

        with patch.object(download, "validate_file", side_effect=validate), \
                patch.object(download, "urlopen", side_effect=lambda *a, **k: Response(PAYLOAD)):
            self.acquire()
        self.assertEqual(self.destination.read_bytes(), PAYLOAD)
        self.assertEqual(older.read_bytes(), b"keep this older version")
        self.assertEqual(list(self.destination.parent.glob(".*.part")), [])
        entry = self.manifest()[0]
        self.assertEqual(entry["status"], "downloaded")
        self.assertEqual(entry["sha256"], hashlib.sha256(PAYLOAD).hexdigest())
        for field in ("url", "filename", "date", "product", "source_version"):
            self.assertEqual(entry[field], self.record[field])


if __name__ == "__main__":
    unittest.main()
