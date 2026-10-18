"""Rebuild the Parquet footer of a file whose writer was killed before close().

`lerobot` keeps one `pq.ParquetWriter` open per shard and writes the footer only
on `close()`. The generation run died to SIGHUP, so every shard's data parquet
has intact row groups and no footer, and pyarrow refuses the file outright.

Row groups are self-contained: each column chunk is a dictionary page followed
by data pages, laid out contiguously, and every page carries a Thrift header
giving its own compressed size. So the chunk boundaries can be recovered by
walking the page headers, and the footer rebuilt around them. The schema,
`created_by`, `column_orders` and the Arrow `key_value_metadata` blob are taken
from a reference file written by the same ShardWriter with the same features.

Validated by `--self-test`, which truncates the reference's own footer, rebuilds
it, and asserts the recovered table equals the original.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from fastparquet.cencoding import NumpyIO, ThriftObject, from_buffer

MARKER = b"PAR1"
DICTIONARY_PAGE = 2
NCOLS = 7  # observation.state, action, timestamp, frame_index, episode_index, index, task_index


def read_footer(buf: bytes) -> ThriftObject:
    flen = int.from_bytes(buf[-8:-4], "little")
    raw = np.frombuffer(buf[-8 - flen : -8], "uint8").copy()
    return from_buffer(NumpyIO(raw), "FileMetaData")


def scan_pages(buf: bytes) -> list[dict]:
    """Every complete page in the file body, in order.

    Stops at the first page whose header or data runs past the end of the file:
    that is the write that was interrupted, and everything after it is garbage.
    """
    arr = np.frombuffer(buf, "uint8").copy()
    io = NumpyIO(arr)
    io.seek(len(MARKER))
    pages = []
    while io.tell() < len(buf):
        start = io.tell()
        try:
            ph = from_buffer(io, "PageHeader")
        except Exception:
            break  # header itself is truncated
        body = io.tell()
        end = body + ph.compressed_page_size
        if end > len(buf):
            break  # page data is truncated
        nvals = (
            ph.dictionary_page_header.num_values
            if ph.type == DICTIONARY_PAGE
            else ph.data_page_header.num_values
        )
        pages.append(
            {
                "start": start,
                "body": body,
                "end": end,
                "type": ph.type,
                "num_values": nvals,
                "compressed": ph.compressed_page_size,
                "uncompressed": ph.uncompressed_page_size,
            }
        )
        io.seek(end)
    return pages


def group_chunks(pages: list[dict]) -> list[list[dict]]:
    """Split the page stream into column chunks at each dictionary page."""
    chunks: list[list[dict]] = []
    for p in pages:
        if p["type"] == DICTIONARY_PAGE or not chunks:
            chunks.append([p])
        else:
            chunks[-1].append(p)
    return chunks


def build_column(chunk: list[dict], ref_meta: ThriftObject) -> ThriftObject:
    """One ColumnChunk, with offsets and sizes measured off the real pages.

    Type, encodings, codec and path come from the reference column: they are a
    property of the schema and the writer's settings, not of this file.
    """
    dict_pages = [p for p in chunk if p["type"] == DICTIONARY_PAGE]
    data_pages = [p for p in chunk if p["type"] != DICTIONARY_PAGE]
    first = chunk[0]["start"]
    meta = ThriftObject.from_fields(
        "ColumnMetaData",
        type=ref_meta.type,
        path_in_schema=ref_meta.path_in_schema,
        encodings=ref_meta.encodings,
        codec=ref_meta.codec,
        num_values=sum(p["num_values"] for p in data_pages),
        data_page_offset=data_pages[0]["start"],
        dictionary_page_offset=dict_pages[0]["start"] if dict_pages else None,
        # Header bytes count toward the chunk size, so measure span, not payload.
        total_compressed_size=chunk[-1]["end"] - first,
        total_uncompressed_size=sum(p["uncompressed"] + (p["body"] - p["start"]) for p in chunk),
        key_value_metadata=[],
        i32list=[1, 4],
    )
    return ThriftObject.from_fields("ColumnChunk", file_offset=first, meta_data=meta)


def rebuild(src: Path, ref: Path, dst: Path, verbose: bool = True) -> dict:
    buf = src.read_bytes()
    if buf[:4] != MARKER:
        raise SystemExit(f"{src}: not a parquet file")
    ref_fmd = read_footer(ref.read_bytes())
    ref_cols = ref_fmd.row_groups[0].columns
    if len(ref_cols) != NCOLS:
        raise SystemExit(f"{ref}: expected {NCOLS} columns, got {len(ref_cols)}")

    chunks = group_chunks(scan_pages(buf))
    complete = len(chunks) // NCOLS
    if complete == 0:
        raise SystemExit(f"{src}: no complete row group recovered")

    row_groups, total_rows = [], 0
    for g in range(complete):
        cols = [
            build_column(chunks[g * NCOLS + c], ref_cols[c].meta_data) for c in range(NCOLS)
        ]
        # timestamp is a flat column, so its value count is the row count; the
        # two list columns carry 8 values per row and would overcount.
        rows = cols[2].meta_data.num_values
        for c in (3, 4, 5, 6):
            if cols[c].meta_data.num_values != rows:
                raise SystemExit(f"{src}: row group {g} column {c} disagrees on row count")
        for c in (0, 1):
            if cols[c].meta_data.num_values != rows * 8:
                raise SystemExit(f"{src}: row group {g} column {c} is not 8-wide")
        row_groups.append(
            ThriftObject.from_fields(
                "RowGroup",
                num_rows=rows,
                columns=cols,
                total_byte_size=sum(c.meta_data.total_uncompressed_size for c in cols),
            )
        )
        total_rows += rows

    fmd = ThriftObject.from_fields(
        "FileMetaData",
        version=ref_fmd.version,
        schema=ref_fmd.schema,
        num_rows=total_rows,
        row_groups=row_groups,
        key_value_metadata=ref_fmd.key_value_metadata,
        created_by=ref_fmd.created_by,
        column_orders=ref_fmd.column_orders,
        i32list=[1],
    )

    body_end = chunks[complete * NCOLS - 1][-1]["end"]
    foot = fmd.to_bytes()
    dst.write_bytes(buf[:body_end] + foot + len(foot).to_bytes(4, "little") + MARKER)

    dropped = len(chunks) - complete * NCOLS
    if verbose:
        print(
            f"  {src.parent.parent.parent.name}: {complete} row groups, {total_rows} rows"
            f"{f', dropped {dropped} partial chunks' if dropped else ''}"
        )
    return {"row_groups": complete, "rows": total_rows, "dropped_chunks": dropped}


def self_test(ref: Path, tmp: Path) -> None:
    """Strip the reference's footer, rebuild it, and require an identical table."""
    import pyarrow.parquet as pq

    buf = ref.read_bytes()
    flen = int.from_bytes(buf[-8:-4], "little")
    headless = tmp / "headless.parquet"
    headless.write_bytes(buf[: -8 - flen])  # body only, exactly as a killed writer leaves it

    out = tmp / "rebuilt.parquet"
    rebuild(headless, ref, out, verbose=False)

    want, got = pq.read_table(ref), pq.read_table(out)
    assert want.equals(got), "recovered table differs from the original"
    print(f"self-test OK: {want.num_rows} rows, {want.num_columns} cols recovered identically")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", type=Path, required=True, help="intact file with the same features")
    ap.add_argument("--tmp", type=Path, required=True)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("targets", nargs="*", type=Path, help="truncated parquet files to repair")
    args = ap.parse_args()

    args.tmp.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        self_test(args.ref, args.tmp)
        if not args.targets:
            return 0

    import pyarrow.parquet as pq

    for t in args.targets:
        backup = t.with_suffix(".parquet.truncated")
        if not backup.exists():
            shutil.copy2(t, backup)  # keep the original bytes until the repair is verified
        out = args.tmp / f"{t.parent.parent.parent.name}.parquet"
        rebuild(backup, args.ref, out)
        f = pq.ParquetFile(out)
        print(f"     verified by pyarrow: {f.metadata.num_rows} rows")
        shutil.move(out, t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
