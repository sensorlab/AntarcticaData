"""Per-observable accuracy ranges for downstream uncertainty budgets (ESSD review, R1).

Converts the receiver-reported UBX-RXM-RAWX standard-deviation codes to
physical units and aggregates percentiles per (gnssId, sigId):

    prStdev  pseudorange   0.01 * 2**n  m
    cpStdev  carrier phase 0.004 * n    cycles  (-> mm via wavelength; n = 15 invalid)
    doStdev  Doppler       0.002 * 2**n Hz      (-> m/s via wavelength)

and the UBX-NAV-PVT solution accuracies (hAcc, vAcc, sAcc, tAcc). Percentiles
are pooled over all days by summing per-day histograms on the discrete code
grids, so memory stays flat across the cruise.

Outputs:
    tables/T_obs_accuracy.csv       (gnssId, sigId, observable, p5, p50, p95, n)
    tables/T_nav_accuracy.csv       (metric, p5, p50, p95, n)

Usage:
    python -m verify.obs_accuracy [--days 20250930,20260127]
"""
from __future__ import annotations

import logging
from collections import defaultdict

import click
import numpy as np
import polars as pl

from analysis._common import C_M_S, gnss_signal_freq, list_days, load_config, staged_path, tables_dir

log = logging.getLogger(__name__)

SIG_LABEL = {
    (0, 0): "GPS L1 C/A", (0, 7): "GPS L5 Q", (2, 0): "Galileo E1 C", (2, 4): "Galileo E5a Q",
    (3, 0): "BeiDou B1I D1", (3, 7): "BeiDou B2a", (1, 0): "SBAS L1", (5, 0): "QZSS L1 C/A",
    (5, 9): "QZSS L5 Q", (6, 0): "GLONASS L1 OF", (7, 0): "NavIC L5",
}
PR_SCALE = 0.01 * 2.0 ** np.arange(16)      # m
CP_SCALE = 0.004 * np.arange(16)            # cycles
DO_SCALE = 0.002 * 2.0 ** np.arange(16)     # Hz
CP_INVALID = 15

# NAV-PVT accuracies are continuous; histogram on fixed log-spaced edges.
NAV_EDGES = {
    "hAcc_m": np.logspace(-3, 3, 1201),
    "vAcc_m": np.logspace(-3, 3, 1201),
    "sAcc_m_s": np.logspace(-4, 2, 1201),
    "tAcc_ns": np.logspace(-1, 6, 1401),
}


def _pct_from_hist(values: np.ndarray, counts: np.ndarray, qs=(0.05, 0.5, 0.95)) -> list[float]:
    order = np.argsort(values)
    v, c = values[order], counts[order]
    cdf = np.cumsum(c) / c.sum()
    return [float(v[np.searchsorted(cdf, q)]) for q in qs]


def accumulate(days: list[str], cfg: dict):
    sig_hist: dict[tuple[int, int, str], np.ndarray] = defaultdict(lambda: np.zeros(16, dtype=np.int64))
    nav_hist = {k: np.zeros(len(e) - 1, dtype=np.int64) for k, e in NAV_EDGES.items()}
    for d in days:
        rawx = pl.read_parquet(
            staged_path(d, "rxm_rawx", cfg),
            columns=["gnssId", "sigId", "prStdev", "cpStdev", "doStdev", "trkStat"],
        )
        for col in ("prStdev", "cpStdev", "doStdev"):
            src = rawx
            if col == "cpStdev":  # only meaningful with valid carrier phase
                src = rawx.filter((pl.col("trkStat") & 0x02) != 0)
            vc = src.group_by(["gnssId", "sigId", col]).agg(pl.len().alias("n"))
            for g, s, code, n in vc.iter_rows():
                sig_hist[(g, s, col)][code & 0x0F] += n
        pvt = pl.read_parquet(staged_path(d, "nav_pvt", cfg), columns=["fixType", "hAcc_mm", "vAcc_mm", "sAcc", "tAcc"])
        pvt = pvt.filter(pl.col("fixType") >= 3)
        cols = {
            "hAcc_m": pvt["hAcc_mm"].to_numpy() / 1e3,
            "vAcc_m": pvt["vAcc_mm"].to_numpy() / 1e3,
            "sAcc_m_s": pvt["sAcc"].to_numpy() / 1e3,  # mm/s
            "tAcc_ns": pvt["tAcc"].to_numpy().astype(float),
        }
        for k, v in cols.items():
            nav_hist[k] += np.histogram(v, bins=NAV_EDGES[k])[0]
        log.info("accumulated %s", d)
    return sig_hist, nav_hist


def summarise(sig_hist, nav_hist) -> tuple[pl.DataFrame, pl.DataFrame]:
    rows = []
    for (g, s, col), counts in sorted(sig_hist.items()):
        lam = C_M_S / gnss_signal_freq(g, s, 0) if g != 6 else C_M_S / 1_602_000_000.0
        if col == "prStdev":
            vals, unit, name = PR_SCALE, "m", "pseudorange"
        elif col == "cpStdev":
            counts = counts.copy()
            counts[CP_INVALID] = 0
            vals, unit, name = CP_SCALE * lam * 1e3, "mm", "carrier phase"
        else:
            vals, unit, name = DO_SCALE * lam, "m/s", "Doppler (range rate)"
        if counts.sum() == 0:
            continue
        p5, p50, p95 = _pct_from_hist(vals, counts)
        rows.append({"gnssId": g, "sigId": s, "signal": SIG_LABEL.get((g, s), f"{g}/{s}"),
                     "observable": name, "unit": unit, "p5": p5, "p50": p50, "p95": p95, "n": int(counts.sum())})
    nav_rows = []
    for k, counts in nav_hist.items():
        e = NAV_EDGES[k]
        centres = np.sqrt(e[:-1] * e[1:])
        p5, p50, p95 = _pct_from_hist(centres, counts)
        nav_rows.append({"metric": k, "p5": p5, "p50": p50, "p95": p95, "n": int(counts.sum())})
    return pl.DataFrame(rows), pl.DataFrame(nav_rows)


@click.command()
@click.option("--days", default=None, help="Comma-separated YYYYMMDD; default all staged days.")
def main(days: str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    day_list = days.split(",") if days else list_days(cfg)
    sig, nav = summarise(*accumulate(day_list, cfg))
    out = tables_dir(cfg)
    sig.write_csv(out / "T_obs_accuracy.csv")
    nav.write_csv(out / "T_nav_accuracy.csv")
    with pl.Config(tbl_rows=60, tbl_cols=12, float_precision=3):
        print(sig.select(["signal", "observable", "unit", "p5", "p50", "p95", "n"]))
        print(nav)


if __name__ == "__main__":
    main()
