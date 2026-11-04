"""Byte-level framing integrity of the raw .ubx files (ESSD review, R1 consistency).

A failed Fletcher-8 check at a 0xB5 0x62 position is usually a sync pattern
occurring by chance inside a payload, not corruption, so the number of
checksum failures is not a meaningful loss metric. This script instead
reports, per file, how many bytes lie inside checksum-valid UBX frames and
how many lie between them (unframed runs: corrupted/truncated frames or
non-UBX output), using the same scan as ``ubx.parser.iter_frames``.

Output: tables/T_framing_integrity.csv

Usage:
    python -m verify.framing_integrity [--workers 32]
"""
from __future__ import annotations

import logging
from multiprocessing import Pool

import click
import polars as pl

from analysis._common import load_config, resolve_path, tables_dir
from ubx.parser import iter_frames, open_mmap

log = logging.getLogger(__name__)


def scan(path: str) -> dict:
    n_frames = framed = 0
    gaps = gap_bytes = 0
    prev_end = 0
    with open_mmap(path) as mm:
        size = len(mm)
        for f in iter_frames(mm):
            start = f.payload_offset - 6
            if start > prev_end:
                gaps += 1
                gap_bytes += start - prev_end
            end = f.payload_offset + f.payload_length + 2
            framed += end - start
            n_frames += 1
            prev_end = end
    if size > prev_end:
        gaps += 1
        gap_bytes += size - prev_end
    return {"file": path.split("/")[-1], "bytes": size, "n_frames": n_frames, "framed_bytes": framed,
            "unframed_runs": gaps, "unframed_bytes": gap_bytes}


@click.command()
@click.option("--workers", default=32, show_default=True)
def main(workers: int) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    files = sorted(str(p) for p in resolve_path(cfg["paths"]["ubx_dir"]).glob("*.ubx"))
    with Pool(min(workers, len(files))) as pool:
        rows = pool.map(scan, files)
    df = pl.DataFrame(rows).sort("file")
    df.write_csv(tables_dir(cfg) / "T_framing_integrity.csv")
    tot = df.select(pl.all().exclude("file").sum()).row(0, named=True)
    print(f"files={df.height} bytes={tot['bytes']:,} frames={tot['n_frames']:,} "
          f"unframed_runs={tot['unframed_runs']:,} unframed_bytes={tot['unframed_bytes']:,} "
          f"({tot['unframed_bytes'] / tot['bytes']:.2e} of total)")


if __name__ == "__main__":
    main()
