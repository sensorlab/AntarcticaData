"""Per-day file inventory and missing-data summary (ESSD review, R1/R2).

For every daily file this reports the raw ``.ubx`` byte count, the first and
last UBX-NAV-PVT epoch (UTC), the number of NAV-PVT / RXM-RAWX / MON-RF
epochs, the completeness against the nominal 2 Hz rate over the recorded
span, the epochs missing inside the file and the interval to the previous
file. Missing epochs are counted from the NAV-PVT inter-epoch intervals: an
interval of k nominal periods (0.5 s) means k - 1 missing epochs. Duplicate
epochs (same time tag twice) are counted separately and dropped.

The logger rotates files on its own clock, which drifts from UTC over the
cruise (files start up to ~18 min before 00:00 UTC by April 2026), so file
spans are reported as recorded rather than clipped to the UTC day.

Outputs:
    tables/T_file_inventory.csv        one row per day (deposit supplement)
    tables/T_file_inventory.tex        longtable for the paper appendix
    tables/T_gaps.csv                  one row per interval with missing epochs

Usage:
    python -m verify.file_inventory
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from analysis._common import list_days, load_config, resolve_path, staged_path, tables_dir

log = logging.getLogger(__name__)

NAV_RATE_HZ = 2.0
BOOT_T_NS = 1_735_689_600 * 10**9  # 2025-01-01, as in analysis.anomalies boot_state


def _utc(t_ns: int) -> str:
    return datetime.fromtimestamp(t_ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _nrows(path) -> int:
    return pq.ParquetFile(path).metadata.num_rows if path.exists() else 0


def inventory_day(day: str, cfg: dict) -> tuple[dict, list[dict]]:
    ubx = resolve_path(cfg["paths"]["ubx_dir"]) / f"{day}.ubx"
    pvt = pl.read_parquet(staged_path(day, "nav_pvt", cfg), columns=["t_ns", "fixType"])
    # Drop pre-fix boot-state epochs whose receiver clock is not yet set.
    t_all = pvt.filter(pl.col("t_ns") > BOOT_T_NS)["t_ns"].to_numpy()
    t = np.unique(t_all)                      # sorted, duplicates removed
    n_dup = int(t_all.size - t.size)
    dt = np.diff(t) / 1e9
    missing = np.maximum(np.rint(dt * NAV_RATE_HZ).astype(int) - 1, 0)
    gap_idx = np.flatnonzero(missing > 0)
    gaps = [
        {"day": day, "gap_start_utc": _utc(int(t[i])), "gap_end_utc": _utc(int(t[i + 1])),
         "interval_s": float(dt[i]), "missing_epochs": int(missing[i])}
        for i in gap_idx
    ]
    span_s = (t[-1] - t[0]) / 1e9 if t.size else 0.0
    n_rf = _nrows(staged_path(day, "mon_rf", cfg))
    row = {
        "day": day,
        "ubx_bytes": ubx.stat().st_size if ubx.exists() else None,
        "first_epoch_utc": _utc(int(t[0])) if t.size else None,
        "last_epoch_utc": _utc(int(t[-1])) if t.size else None,
        "span_h": span_s / 3600.0,
        "n_nav_pvt": int(t.size),
        "n_rxm_rawx_obs": _nrows(staged_path(day, "rxm_rawx", cfg)),
        "n_mon_rf_epochs": n_rf // 2,  # two RF blocks per 1 Hz epoch
        "completeness_span": t.size / (span_s * NAV_RATE_HZ + 1) if t.size else 0.0,
        "completeness_24h": t.size / (86_400 * NAV_RATE_HZ),
        "_first_t_ns": int(t[0]) if t.size else None,
        "_last_t_ns": int(t[-1]) if t.size else None,
        "n_missing_epochs": int(missing.sum()),
        "n_intervals_with_missing": len(gaps),
        "longest_interval_s": float(dt.max()) if dt.size else 0.0,
        "n_duplicate_epochs": n_dup,
    }
    return row, gaps


def _tex(df: pl.DataFrame) -> str:
    lines = [
        r"\begin{longtable}{lrllrrrr}",
        r"\caption{Daily file inventory. MB is the size of the raw \texttt{.ubx} file; first/last are the first and last "
        r"\texttt{UBX-NAV-PVT} epoch (UTC); $N_\mathrm{PVT}$ is the number of 2\,Hz navigation epochs; "
        r"completeness is $N_\mathrm{PVT}$ divided by the nominal 172\,800 epochs per 24\,h; ``missing'' is the "
        r"number of epochs missing inside the file; ``prev.'' is the interval between the last epoch of the "
        r"previous file and the first epoch of this one (0.5\,s when no epoch is lost).}\label{tab:inventory}\\",
        r"\tophline",
        r"File & MB & First (UTC) & Last (UTC) & $N_\mathrm{PVT}$ & Compl.\ (\%) & Missing & Prev.\ (s) \\",
        r"\middlehline\endhead",
    ]
    for r in df.iter_rows(named=True):
        mb = f"{r['ubx_bytes'] / 1e6:.0f}" if r["ubx_bytes"] else "--"
        g = r["gap_to_prev_file_s"]
        prev = "--" if g is None else f"{g:.1f}"
        lines.append(
            f"{r['day'][:4]}-{r['day'][4:6]}-{r['day'][6:]} & {mb} & {r['first_epoch_utc'][5:16]} & "
            f"{r['last_epoch_utc'][5:16]} & {r['n_nav_pvt']:,} & {100 * r['completeness_24h']:.3f} & "
            f"{r['n_missing_epochs']} & {prev} \\\\".replace(",", "\\,")
        )
    lines += [r"\bottomhline", r"\end{longtable}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    rows, gaps = [], []
    for d in list_days(cfg):
        r, g = inventory_day(d, cfg)
        rows.append(r)
        gaps.extend(g)
    df = pl.DataFrame(rows).sort("day")
    date = pl.col("day").str.strptime(pl.Date, "%Y%m%d")
    df = df.with_columns(
        pl.when((date - date.shift(1)).dt.total_days() == 1)
        .then((pl.col("_first_t_ns") - pl.col("_last_t_ns").shift(1)) / 1e9)
        .otherwise(None)
        .alias("gap_to_prev_file_s")
    ).drop(["_first_t_ns", "_last_t_ns"])
    out = tables_dir(cfg)
    df.write_csv(out / "T_file_inventory.csv")
    pl.DataFrame(gaps, schema={"day": pl.Utf8, "gap_start_utc": pl.Utf8, "gap_end_utc": pl.Utf8,
                               "interval_s": pl.Float64, "missing_epochs": pl.Int64}).write_csv(out / "T_gaps.csv")
    (out / "T_file_inventory.tex").write_text(_tex(df))
    full = df.filter(pl.col("span_h") > 23.9)
    print(f"days={df.height}  total bytes={df['ubx_bytes'].sum():,}  N_PVT={df['n_nav_pvt'].sum():,}")
    print(f"full-span days={full.height}  partial={df.height - full.height}")
    print(f"missing epochs inside files: {df['n_missing_epochs'].sum()} in {(df['n_missing_epochs'] > 0).sum()} files; "
          f"longest interval {df['longest_interval_s'].max():.3f} s; duplicate epochs {df['n_duplicate_epochs'].sum()}")
    gp = df["gap_to_prev_file_s"].drop_nulls()
    if gp.len():
        lost = (gp * NAV_RATE_HZ).round() - 1
        print(f"file changes: {gp.len()}; without loss {(lost == 0).sum()}; epochs lost {int(lost.sum())}; "
              f"interval median={gp.median():.1f}s max={gp.max():.1f}s")
    print(f"completeness (24 h, full days) min={full['completeness_24h'].min():.5f} "
          f"median={full['completeness_24h'].median():.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
