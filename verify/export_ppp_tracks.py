"""Export the kinematic PPP trajectory as a derived product (optional deposit file).

Reads every PRIDE PPP-AR kin file (via ``ppp.compare``), converts GPS time to
UTC, and writes one Parquet file with all epochs:

    derived/ppp_export/ppp_trajectory_1Hz.parquet
        time_utc (ISO 8601), t_unix_ns, lat_deg, lon_deg, h_ell_m, nsat, pdop

Positions refer to the antenna reference point (the RINEX antenna offset is
zero and PRIDE applies the antenna phase-centre corrections). Epochs are not
filtered; users can apply the paper's selection (nsat > 4 and 0 < pdop < 4).

Usage:
    python -m verify.export_ppp_tracks
"""
from __future__ import annotations

import logging

import polars as pl

from analysis._common import derived_dir, list_days, load_config
from ppp.compare import _find_pride_kin, _read_pride_kin

log = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    parts, days = [], []
    for d in list_days(cfg):
        kin = _find_pride_kin(d, cfg)
        if kin is None:
            continue
        df = _read_pride_kin(kin)
        if df.is_empty():
            continue
        parts.append(df)
        days.append(d)
    if not parts:
        print("no PRIDE kin files found")
        return 1
    out = (
        pl.concat(parts)
        .unique(subset="t_ns", keep="first")
        .sort("t_ns")
        .rename({"t_ns": "t_unix_ns", "lat": "lat_deg", "lon": "lon_deg", "h_ell": "h_ell_m"})
        .with_columns(pl.from_epoch("t_unix_ns", time_unit="ns").dt.strftime("%Y-%m-%dT%H:%M:%S%.3fZ").alias("time_utc"))
        .select(["time_utc", "t_unix_ns", "lat_deg", "lon_deg", "h_ell_m", "nsat", "pdop"])
    )
    out_dir = derived_dir(cfg) / "ppp_export"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ppp_trajectory_1Hz.parquet"
    out.write_parquet(path, compression="zstd")
    print(f"days={len(days)} epochs={out.height:,} -> {path} ({path.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
