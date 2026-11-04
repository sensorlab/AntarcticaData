"""Per-day and per-regime code-multipath summary for the paper text.

For every day: M1/M2 RMS above and below 30 deg elevation (from
``derived/multipath``), day-median position and ground speed (from
NAV-PVT). Days are assigned to a regime from position and speed:

    trieste_port      day-median within 20 km of Trieste, median speed < 0.5 m/s
    lyttelton_port    within 20 km of Lyttelton, median speed < 0.5 m/s
    antarctic         day-median latitude south of 60 S
    open_ocean        median speed > 3 m/s and none of the above
    other             everything else

Outputs: tables/T_multipath_daily.csv and a printed per-regime median/IQR.

Usage:
    python -m verify.multipath_summary
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

from analysis._common import derived_dir, haversine_km, list_days, load_config, read_parquet, staged_path, tables_dir

log = logging.getLogger(__name__)

PORTS = {"trieste_port": (45.6131, 13.8092), "lyttelton_port": (-43.606, 172.72)}
PORT_RADIUS_KM = 20.0
STATIONARY_M_S = 0.5
UNDERWAY_M_S = 3.0
HIGH_EL_DEG = 30.0


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x))) if x.size else float("nan")


def day_row(day: str, cfg: dict) -> dict | None:
    p = derived_dir(cfg) / "multipath" / f"{day}.multipath.parquet"
    if not p.exists():
        return None
    mp = read_parquet(p).select(["M1", "M2", "elev"]).drop_nulls()
    hi = mp.filter(pl.col("elev") >= HIGH_EL_DEG)
    lo = mp.filter(pl.col("elev") < HIGH_EL_DEG)
    pvt = read_parquet(staged_path(day, "nav_pvt", cfg)).filter(pl.col("fixType") >= 3).select(
        (pl.col("lat_1e7") / 1e7).median().alias("lat"),
        (pl.col("lon_1e7") / 1e7).median().alias("lon"),
        (pl.col("gSpeed_mm_s") / 1e3).median().alias("speed"),
    ).row(0, named=True)
    regime = "other"
    for name, (la, lo_) in PORTS.items():
        if haversine_km(np.array([pvt["lat"]]), np.array([pvt["lon"]]), np.array([la]), np.array([lo_]))[0] < PORT_RADIUS_KM \
                and pvt["speed"] < STATIONARY_M_S:
            regime = name
    if regime == "other" and pvt["lat"] < -60.0:
        regime = "antarctic"
    elif regime == "other" and pvt["speed"] > UNDERWAY_M_S:
        regime = "open_ocean"
    return {
        "day": day, "regime": regime, **pvt,
        "m1_rms_hi": _rms(hi["M1"].to_numpy()), "m2_rms_hi": _rms(hi["M2"].to_numpy()),
        "m1_rms_lo": _rms(lo["M1"].to_numpy()), "m2_rms_lo": _rms(lo["M2"].to_numpy()),
        "n_hi": hi.height, "n_lo": lo.height,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    rows = [r for d in list_days(cfg) if (r := day_row(d, cfg))]
    df = pl.DataFrame(rows)
    df.write_csv(tables_dir(cfg) / "T_multipath_daily.csv")
    summ = df.group_by("regime").agg(
        pl.len().alias("n_days"),
        *[pl.col(c).median().alias(f"{c}_p50") for c in ("m1_rms_hi", "m2_rms_hi", "m1_rms_lo", "m2_rms_lo")],
        pl.col("m1_rms_hi").quantile(0.25).alias("m1_rms_hi_p25"),
        pl.col("m1_rms_hi").quantile(0.75).alias("m1_rms_hi_p75"),
    ).sort("regime")
    with pl.Config(tbl_cols=12, float_precision=2, tbl_width_chars=160):
        print(summ)
    print(f"cruise median M1 RMS el>=30: {df['m1_rms_hi'].median():.2f} m;  el<30: {df['m1_rms_lo'].median():.2f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
