"""Pull only the ``basic_pick_place`` HDF5 episodes out of a remote EgoDex zip.

The EgoDex archives are tens of GB each and we need one task from them, so this
streams the zip's central directory over HTTP (via ``remotezip``) and fetches the
byte ranges for the matching members only -- no full download, no local unzip.

Fill in ``EGODEX_ZIP_URL`` (or pass ``--url``) before running; the script refuses
to do anything with the placeholder still in place.

    uv run python data/download_egodex.py --list          # inspect what matches
    uv run python data/download_egodex.py                 # extract to data/egodex_raw/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from remotezip import RemoteZip

# Direct URL to an EgoDex part zip, e.g. a HuggingFace .../resolve/main/part1.zip
# link. Must be a plain HTTP(S) URL that supports range requests.
EGODEX_ZIP_URL = URL = "https://ml-site.cdn-apple.com/datasets/egodex/test.zip"

TASK = "basic_pick_place"
SUFFIX = ".hdf5"
OUTPUT_DIR = Path(__file__).resolve().parent / "egodex_raw"


def matching_members(zf: RemoteZip, task: str, suffix: str) -> list[str]:
    """Names of the archive members belonging to ``task`` with the given suffix."""
    return sorted(
        name
        for name in zf.namelist()
        if suffix == Path(name).suffix and task in Path(name).parts[:-1]
    )


def extract(url: str, task: str, suffix: str, output_dir: Path, list_only: bool) -> int:
    with RemoteZip(url) as zf:
        members = matching_members(zf, task, suffix)
        if not members:
            print(f"no members matched task={task!r} suffix={suffix!r}", file=sys.stderr)
            return 1

        total = sum(zf.getinfo(name).file_size for name in members)
        print(f"{len(members)} member(s), {total / 1e9:.2f} GB uncompressed")
        if list_only:
            for name in members:
                print(f"  {name}")
            return 0

        output_dir.mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(members, 1):
            # Flatten <...>/basic_pick_place/<episode>.hdf5 to <episode>.hdf5.
            destination = output_dir / Path(name).name
            if destination.exists():
                print(f"[{i}/{len(members)}] skip {destination.name} (exists)")
                continue
            print(f"[{i}/{len(members)}] {destination.name}")
            with zf.open(name) as src, open(destination, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)

    print(f"done -> {output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=EGODEX_ZIP_URL, help="EgoDex part zip URL")
    parser.add_argument("--task", default=TASK, help=f"task directory (default: {TASK})")
    parser.add_argument("--suffix", default=SUFFIX, help=f"file suffix (default: {SUFFIX})")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--list", action="store_true", help="print matching members without downloading"
    )
    args = parser.parse_args()

    if not args.url:
        parser.error(
            "no zip URL: set EGODEX_ZIP_URL in this file or pass --url <https://...>"
        )

    return extract(args.url, args.task, args.suffix, args.output_dir, args.list)


if __name__ == "__main__":
    raise SystemExit(main())
