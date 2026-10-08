"""Download GOES-16 and OMNI source files for the paper's study interval.

Dates are inclusive. OMNI files cover whole months; readers must apply their
own exact timestamp limits. --list-only reads directory indexes without writes.
MPSH acquisition selects source version 2, which the public reader supports.
"""

import argparse
from datetime import date, datetime, timezone
import hashlib
from http.client import HTTPException
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import ssl
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from .config import STUDY_START, STUDY_END


GOES_ROOT = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/data/"
OMNI_ROOT = "https://cdaweb.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro2_1min/"
PRODUCT_DIRS = {"mpsh": "mpsh-l2-avg1m_science", "mag": "magn-l2-avg1m", "ephe": "ephe-l2-orb1m"}
PATTERNS = {
    "mpsh": re.compile(r"sci_mpsh-l2-avg1m_g16_d(\d{8})_v(\d+)-(\d+)-(\d+)\.nc"),
    "mag": re.compile(r"(?:dn|sci)_magn-l2-avg1m_g16_d(\d{8})_v(\d+)[-_](\d+)[-_](\d+)\.nc"),
    "ephe": re.compile(r"(?:dn|sci)_ephe-l2-orb1m_g16_d(\d{8})_v(\d+)[-_](\d+)[-_](\d+)\.nc"),
    "omni": re.compile(r"omni_hro2_1min_(\d{8})_v(\d+)\.cdf"),
}
TIMEOUT = 60
ATTEMPTS = 3
CHUNK_SIZE = 1024 * 1024


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def month_starts(start, end):
    current = start.replace(day=1)
    while current <= end:
        yield current
        current = date(current.year + (current.month == 12), current.month % 12 + 1, 1)


def _request(url):
    return Request(url, headers={"User-Agent": "DriftEchoStudy/1.0", "Accept-Encoding": "identity"})


def fetch_listing(url):
    """Return HTML, None for an unavailable index, or raise after retries."""
    for attempt in range(ATTEMPTS):
        try:
            with urlopen(_request(url), timeout=TIMEOUT, context=ssl.create_default_context()) as response:
                return response.read().decode("utf-8")
        except HTTPError as exc:
            if exc.code == 404:
                return None
            error = exc
            if exc.code not in (408, 429, 500, 502, 503, 504):
                raise
        except (URLError, OSError, UnicodeError, HTTPException) as exc:
            error = exc
        if attempt + 1 < ATTEMPTS:
            time.sleep(2 ** attempt)
    raise error


def select_latest(html, index_url, product, start, end):
    """Select the highest numeric version for each date/month in an index."""
    parser = _Links()
    parser.feed(html)
    latest = {}
    for href in parser.hrefs:
        parsed = urlparse(urljoin(index_url, href))
        if parsed.scheme != "https" or parsed.netloc != urlparse(index_url).netloc:
            continue
        filename = parsed.path.rsplit("/", 1)[-1]
        match = PATTERNS[product].fullmatch(filename)
        if not match:
            continue
        try:
            file_date = datetime.strptime(match[1], "%Y%m%d").date()
        except ValueError:
            continue
        if product == "omni":
            if file_date.day != 1 or not start.replace(day=1) <= file_date <= end:
                continue
        elif not start <= file_date <= end:
            continue
        version = tuple(int(part) for part in match.groups()[1:])
        if product == "mpsh" and version[0] != 2:
            continue
        record = {"product": product, "date": file_date.isoformat(), "filename": filename,
                  "source_version": list(version), "url": urljoin(index_url, filename)}
        previous = latest.get(file_date)
        if previous is None or (version, filename) > (tuple(previous["source_version"]), previous["filename"]):
            latest[file_date] = record
    return [latest[key] for key in sorted(latest)]


def validate_file(path, product):
    """Check container readability and the fields needed by the paper readers."""
    if path.stat().st_size == 0:
        raise ValueError("empty file")
    if product == "omni":
        import cdflib
        with cdflib.CDF(path) as cdf:
            for field in ("Epoch", "BX_GSE", "BY_GSM", "BZ_GSM"):
                info = cdf.varinq(field)
                last = info["Last_Rec"] if isinstance(info, dict) else info.Last_Rec
                if last < 0:
                    raise ValueError(f"empty variable: {field}")
                cdf.varget(field, startrec=0, endrec=0)
                cdf.varget(field, startrec=last, endrec=last)
    else:
        import netCDF4
        required = {"mpsh": ("time", "AvgDiffElectronFlux", "AvgDiffProtonFlux",
                             "DiffElectronEffectiveEnergy", "DiffProtonEffectiveEnergy"),
                    "mag": ("time", "b_epn", "b_brf", "orbit_llr_geo"),
                    "ephe": ("time", "geo_llr")}[product]
        with netCDF4.Dataset(path) as dataset:
            for field in required:
                variable = dataset.variables[field]
                if not variable.size:
                    raise ValueError(f"empty variable: {field}")
                variable[0]
                variable[-1]
            timestamps = dataset.variables["time"]
            if timestamps.ndim != 1:
                raise ValueError("time must have one dimension")
            timestamps[:]
            if product == "mpsh":
                for species in ("Electron", "Proton"):
                    flux = dataset.variables[f"AvgDiff{species}Flux"]
                    energy = dataset.variables[f"Diff{species}EffectiveEnergy"]
                    if flux.ndim != 3 or flux.shape[0] != len(timestamps) or flux.shape[1] != 5:
                        raise ValueError(f"{species} flux must have shape [time, 5, channels]")
                    if energy.shape != flux.shape[1:]:
                        raise ValueError(f"{species} energies must match the flux telescope/channel axes")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_manifest(data_dir, record, destination, status, digest):
    entry = dict(record, path=destination.relative_to(data_dir).as_posix(), status=status,
                 sha256=digest, checked_at_utc=datetime.now(timezone.utc).isoformat())
    with (data_dir / "download_manifest.jsonl").open("a", encoding="utf-8") as manifest:
        manifest.write(json.dumps(entry, sort_keys=True) + "\n")


def download_file(record, data_dir):
    """Resume a valid file, otherwise stream, validate and atomically replace it."""
    product = record["product"]
    subdir = Path("omni") if product == "omni" else Path("GOES16") / product
    destination = data_dir / subdir / record["filename"]
    if destination.exists():
        try:
            validate_file(destination, product)
        except ImportError:
            raise
        except Exception as exc:
            print(f"Invalid local file, downloading replacement: {destination.name}: {exc}", file=sys.stderr)
        else:
            _record_manifest(data_dir, record, destination, "existing_valid", _sha256(destination))
            print(f"Keep validated file: {destination.name}")
            return
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(ATTEMPTS):
        temporary = None
        try:
            digest = hashlib.sha256()
            received = 0
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix="." + destination.name + ".",
                                             suffix=".part", delete=False) as output:
                temporary = Path(output.name)
                with urlopen(_request(record["url"]), timeout=TIMEOUT, context=ssl.create_default_context()) as response:
                    length = response.headers.get("Content-Length")
                    for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
                        output.write(chunk)
                        digest.update(chunk)
                        received += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            if not received or (length is not None and received != int(length)):
                raise ValueError(f"incomplete transfer: received {received} bytes; expected {length}")
            validate_file(temporary, product)
            os.replace(temporary, destination)
            temporary = None
            _record_manifest(data_dir, record, destination, "downloaded", digest.hexdigest())
            print(f"Downloaded and validated: {destination.name}")
            return
        except ImportError:
            raise
        except Exception as exc:
            error = exc
            print(f"Attempt {attempt + 1}/{ATTEMPTS} failed for {destination.name}: {exc}", file=sys.stderr)
            if isinstance(exc, HTTPError) and exc.code not in (408, 429, 500, 502, 503, 504):
                break
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        if attempt + 1 < ATTEMPTS:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"could not acquire {destination.name}") from error


def _date_argument(value):
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a date in YYYY-MM-DD format") from exc


def main(argv=None, root=None):
    root = Path.cwd() if root is None else Path(root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=_date_argument, default=STUDY_START, help="first study day (inclusive)")
    parser.add_argument("--end", type=_date_argument, default=STUDY_END, help="last study day (inclusive)")
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "raw")
    parser.add_argument("--products", nargs="+", choices=tuple(PATTERNS), default=list(PATTERNS))
    parser.add_argument("--list-only", action="store_true", help="list source URLs; do not download or create files")
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("--start must be on or before --end")
    planned = completed = failures = 0
    for product in dict.fromkeys(args.products):
        if product == "omni":
            indexes = [f"{OMNI_ROOT}{year}/" for year in range(args.start.year, args.end.year + 1)]
        else:
            indexes = [f"{GOES_ROOT}{PRODUCT_DIRS[product]}/{month:%Y/%m}/" for month in month_starts(args.start, args.end)]
        for index in indexes:
            try:
                html = fetch_listing(index)
                if html is None:
                    print(f"No archive index available (HTTP 404): {index}", file=sys.stderr)
                    continue
                records = select_latest(html, index, product, args.start, args.end)
                if not records:
                    print(f"No matching {product} files in requested interval: {index}", file=sys.stderr)
                for record in records:
                    planned += 1
                    if args.list_only:
                        print(record["url"])
                    else:
                        try:
                            download_file(record, args.data_dir)
                            completed += 1
                        except Exception as exc:
                            failures += 1
                            print(f"FAILED: {record['url']}: {exc}", file=sys.stderr)
            except Exception as exc:
                failures += 1
                print(f"FAILED archive index: {index}: {exc}", file=sys.stderr)
    print(f"Found {planned} source files; validated {completed}; failures {failures}. "
          "Archive availability does not guarantee complete daily or minute coverage.", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
