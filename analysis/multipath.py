"""Estey & Meertens M1/M2 code multipath linear combinations.

For each ``(gnssId, svId)`` we need both L1 and L2 (or L2-equivalent for
non-GPS) carrier-phase + pseudorange. The combination

    M_i = P_i − (1 + 2/(α−1)) · Φ1·λ1 + (2/(α−1)) · Φ2·λ2     for i=1
    M_2 = P_2 − (2α/(α−1)) · Φ1·λ1 + (2α/(α−1) − 1) · Φ2·λ2

where ``α = (f1/f2)²``, removes the ionospheric and clock components and
leaves ``2 · multipath + 2 · ambiguities`` per arc. We arc-segment on
time gaps, lock-time resets and geometry-free phase jumps, drop arcs
shorter than 5 min, subtract the per-arc mean to remove the ambiguity
term, and report the residual as multipath.

Output: ``derived/multipath/<day>.multipath.parquet`` with columns
``t_ns, gnssId, svId, sig1, sig2, M1, M2, elev, azim, cno1, cno2, _arc``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import polars as pl

from analysis._common import (
    C_M_S,
    derived_dir,
    gnss_signal_freq,
    list_days,
    load_config,
    read_parquet,
    staged_path,
    write_parquet,
)

log = logging.getLogger(__name__)


# Signal-id pairs actually tracked by this ZED-F9P-15B firmware (HPG L1L5
# 1.40) on this expedition. Verified by aggregating ``staging/<day>/
# rxm_rawx.parquet`` group_by (gnssId, sigId): on the 2025-09-30 reference
# day GPS shows sigId 0 + 7, GAL shows 0 + 4, BDS shows 0 + 7. GLO/QZSS/
# SBAS/NavIC track only one band → no dual-frequency multipath.
#
#   GPS:  sigId 0 (L1 C/A)  +  sigId 7 (L5 I)
#   GAL:  sigId 0 (E1 C)    +  sigId 4 (E5a data)
#   BDS:  sigId 0 (B1I D1)  +  sigId 7 (B2a Q)
#
# The combination uses (f1, f2) = (L1-band, L5/E5a/B2a-band) regardless of
# the historical "L2" naming.
DUAL_FREQ_SIG_PAIRS: dict[int, tuple[int, int]] = {
    0: (0, 7),  # GPS  L1 + L5
    2: (0, 4),  # GAL  E1 + E5a
    3: (0, 7),  # BDS  B1I + B2a
}
# Backwards-compatible alias (used by analysis.tec).
L1L2_SIG_PAIRS = DUAL_FREQ_SIG_PAIRS

# Arc segmentation thresholds (see compute_day_multipath).
MP_GAP_NS = int(2.5e9)       # time gap that ends an arc (2 Hz nominal)
MP_GF_SLIP_M = 0.05          # |Δ(Φ1λ1 − Φ2λ2)| between epochs flagging a cycle slip
MP_M1_JUMP_M = 10.0          # |ΔM1| code-outlier break
MP_MIN_ARC_EPOCHS = 600      # 5 min at 2 Hz; shorter arcs are discarded


def compute_day_multipath(day: str, cfg: dict | None = None) -> pl.DataFrame:
    cfg = cfg or load_config()
    rawx = read_parquet(staged_path(day, "rxm_rawx", cfg))
    sat = read_parquet(staged_path(day, "nav_sat", cfg))
    if rawx.is_empty():
        return pl.DataFrame()

    # Pre-attach elevation from NAV-SAT (per gnssId/svId/t_ns; sigId-agnostic)
    sat_small = sat.select(["t_ns", "gnssId", "svId", "elev", "azim"]).unique(
        subset=["t_ns", "gnssId", "svId"]
    )
    rawx = rawx.join(
        sat_small, on=["t_ns", "gnssId", "svId"], how="left"
    )

    # We only need rows with carrier-phase lock (trkStat & 0x02 != 0)
    rawx = rawx.filter((pl.col("trkStat") & 0x02) != 0)

    out_frames: list[pl.DataFrame] = []
    for gnss_id, (sig1, sig2) in L1L2_SIG_PAIRS.items():
        sub = rawx.filter(pl.col("gnssId") == gnss_id)
        if sub.is_empty():
            continue
        l1 = sub.filter(pl.col("sigId") == sig1).select(
            ["t_ns", "svId", "freqId", "prMes", "cpMes", "cno", "locktime_ms", "elev", "azim"]
        ).rename(
            {
                "prMes": "P1",
                "cpMes": "Phi1",
                "cno": "cno1",
                "locktime_ms": "lt1",
            }
        )
        l2 = sub.filter(pl.col("sigId") == sig2).select(
            ["t_ns", "svId", "freqId", "prMes", "cpMes", "cno", "locktime_ms"]
        ).rename({"prMes": "P2", "cpMes": "Phi2", "cno": "cno2", "locktime_ms": "lt2"})

        joined = l1.join(l2, on=["t_ns", "svId", "freqId"], how="inner")
        if joined.is_empty():
            continue

        # Frequencies (Hz) — vectorised. Only GLONASS varies with freqId.
        n = joined.height
        if gnss_id == 6:  # GLO FDMA
            freqId_arr = joined["freqId"].to_numpy().astype(np.int32)
            n_chan = freqId_arr - 7
            if sig1 == 0:
                f1 = 1_602_000_000.0 + n_chan * 562_500.0
            else:
                f1 = 1_246_000_000.0 + n_chan * 437_500.0
            if sig2 == 0:
                f2 = 1_602_000_000.0 + n_chan * 562_500.0
            else:
                f2 = 1_246_000_000.0 + n_chan * 437_500.0
            f1 = f1.astype(np.float64)
            f2 = f2.astype(np.float64)
        else:
            # Constant per (gnss, sig) — broadcast to length n
            f1 = np.full(n, gnss_signal_freq(gnss_id, sig1, 0), dtype=np.float64)
            f2 = np.full(n, gnss_signal_freq(gnss_id, sig2, 0), dtype=np.float64)
        lam1 = C_M_S / f1
        lam2 = C_M_S / f2
        alpha = (f1 / f2) ** 2

        P1 = joined["P1"].to_numpy()
        P2 = joined["P2"].to_numpy()
        Phi1 = joined["Phi1"].to_numpy() * lam1   # cycles → metres
        Phi2 = joined["Phi2"].to_numpy() * lam2

        with np.errstate(invalid="ignore", divide="ignore"):
            denom = alpha - 1.0
            M1 = P1 - (1.0 + 2.0 / denom) * Phi1 + (2.0 / denom) * Phi2
            M2 = P2 - (2.0 * alpha / denom) * Phi1 + (2.0 * alpha / denom - 1.0) * Phi2

        # Per-arc bias removal. Arcs are segmented per (gnssId, svId) on time
        # gaps and cycle-slip jumps (see the GAP_NS / SLIP_M logic below), and
        # the per-arc mean of M1/M2 is subtracted to remove the carrier-phase
        # ambiguity term, leaving the multipath residual. This is the per-arc
        # removal reported in the paper.
        df = pl.DataFrame(
            {
                "t_ns": joined["t_ns"],
                "gnssId": np.full(joined.height, gnss_id, dtype=np.uint8),
                "svId": joined["svId"],
                "sig1": np.full(joined.height, sig1, dtype=np.uint8),
                "sig2": np.full(joined.height, sig2, dtype=np.uint8),
                "M1": M1,
                "M2": M2,
                "elev": joined["elev"],
                "azim": joined["azim"],
                "cno1": joined["cno1"],
                "cno2": joined["cno2"],
                "_GF": Phi1 - Phi2,
                "_lt1": joined["lt1"],
                "_lt2": joined["lt2"],
            }
        ).sort(["gnssId", "svId", "t_ns"])
        # Arc segmentation on carrier-phase continuity only. A new arc
        # starts on (a) a time gap > GAP_NS, (b) a lock-time reset on either
        # band (receiver-declared loss of lock), (c) a geometry-free phase
        # jump > GF_SLIP_M between consecutive epochs (cycle slip on either
        # band; the ionospheric GF rate is a few mm per 0.5 s even under
        # strong activity), or (d) a code-dominated |ΔM1| > M1_JUMP_M
        # outlier. Code noise must NOT be used as a slip detector: an earlier
        # |ΔM1| > 0.1 m rule fired on nearly every 2 Hz epoch, producing
        # one-epoch arcs whose per-arc mean removal zeroed M1 and biased the
        # RMS low by an order of magnitude. Arcs shorter than MIN_ARC_EPOCHS
        # are dropped so the per-arc mean is a stable ambiguity estimate.
        k = ["gnssId", "svId"]
        df = df.with_columns([
            (pl.col("t_ns").diff().over(k).fill_null(0) > MP_GAP_NS).alias("_gap_break"),
            ((pl.col("_lt1").diff().over(k).fill_null(0) < 0)
             | (pl.col("_lt2").diff().over(k).fill_null(0) < 0)).alias("_lock_break"),
            (pl.col("_GF").diff().over(k).abs().fill_null(0) > MP_GF_SLIP_M).alias("_gf_break"),
            (pl.col("M1").diff().over(k).abs().fill_null(0) > MP_M1_JUMP_M).alias("_m1_break"),
        ])
        df = df.with_columns(
            (pl.col("_gap_break") | pl.col("_lock_break") | pl.col("_gf_break") | pl.col("_m1_break"))
            .cum_sum().over(k).alias("_arc")
        )
        df = df.filter(pl.len().over(k + ["_arc"]) >= MP_MIN_ARC_EPOCHS)
        df = df.with_columns([
            (pl.col("M1") - pl.col("M1").mean().over(k + ["_arc"])).alias("M1"),
            (pl.col("M2") - pl.col("M2").mean().over(k + ["_arc"])).alias("M2"),
        ]).drop(["_gap_break", "_lock_break", "_gf_break", "_m1_break", "_GF", "_lt1", "_lt2"])
        out_frames.append(df)
    if not out_frames:
        return pl.DataFrame()
    return pl.concat(out_frames, how="vertical")


def write_day_multipath(day: str, cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    df = compute_day_multipath(day, cfg)
    out = derived_dir(cfg) / "multipath" / f"{day}.multipath.parquet"
    write_parquet(df, out)
    log.info("Wrote %s (%d rows)", out, df.height)
    return out


@click.command()
@click.option("--day", "day", default=None)
def main(day: str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    days = [day] if day else list_days(cfg)
    for d in days:
        write_day_multipath(d, cfg)


if __name__ == "__main__":
    main()
