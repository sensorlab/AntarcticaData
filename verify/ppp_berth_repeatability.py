"""Within-berth horizontal repeatability of the PPP solution (ESSD revision).

Uses the same stationary windows as the real-time analysis
(``verify.igs_compare``: within ``radius_km`` of a port reference, ground
speed < 0.1 m/s for >= 30 min) and the same 50 m berth grouping
(``figures.fig_berth_repeatability``). For every window it takes the median
PPP position and the median real-time position over the *same* matched
epochs (``derived/ppp/<day>/diff.parquet`` from ``ppp.compare``, filtered
like ``ppp.percentiles``: nsat > 4 and 0 < pDOP < 4). Per berth it reports
the horizontal standard deviation of the window medians for both solutions
and the median real-time minus PPP horizontal offset.

Vertical scatter is not reported: the antenna height of a moored ship
changes with tide and loading, so it does not measure positioning
precision.

Outputs:
    tables/T_ppp_berth_windows.csv         one row per stationary window
    tables/T_ppp_berth_repeatability.csv   one row per berth

Usage:
    python -m verify.ppp_berth_repeatability
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

from analysis._common import derived_dir, list_days, load_config, read_parquet, tables_dir
from figures.fig_berth_repeatability import _enu_metres, _greedy_cluster
from verify.igs_compare import EXAMPLE_REFS, _port_windows

log = logging.getLogger(__name__)

EPS_M = 50.0              # berth grouping, as in fig_berth_repeatability
MIN_WINDOWS = 3           # windows per berth
MIN_PPP_EPOCHS = 600      # >= 10 min of 1 Hz PPP epochs inside a window
ANCHORAGE_SIGMA_M = 5.0   # clusters scattered more than this are anchorage areas, not berths


def _windows_for_day(day: str, cfg: dict) -> list[dict]:
    track_p = derived_dir(cfg) / "track" / f"{day}.track.parquet"
    diff_p = derived_dir(cfg) / "ppp" / day / "diff.parquet"
    if not track_p.exists() or not diff_p.exists():
        return []
    track = read_parquet(track_p)
    diff = read_parquet(diff_p)
    if track.is_empty() or diff.is_empty():
        return []
    diff = diff.filter((pl.col("nsat") > 4) & (pl.col("pdop") > 0.0) & (pl.col("pdop") < 4.0))
    t = track["t_ns"].to_numpy()
    rows = []
    for ref in EXAMPLE_REFS:
        for s, e in _port_windows(track, ref):
            sub = diff.filter((pl.col("t_ns") >= int(t[s])) & (pl.col("t_ns") <= int(t[e])))
            if sub.height < MIN_PPP_EPOCHS:
                continue
            rows.append({
                "port": ref["name"], "day": day,
                "t_start_ns": int(t[s]), "t_end_ns": int(t[e]), "n_ppp": sub.height,
                "rt_lat": float(sub["lat_ob"].median()), "rt_lon": float(sub["lon_ob"].median()),
                "ppp_lat": float(sub["lat_pp"].median()), "ppp_lon": float(sub["lon_pp"].median()),
                # de/dn are PPP minus real-time; report real-time minus PPP
                "rt_minus_ppp_e_m": -float(sub["de_m"].median()),
                "rt_minus_ppp_n_m": -float(sub["dn_m"].median()),
            })
    return rows


def _sigma_h(lat: np.ndarray, lon: np.ndarray) -> float:
    pts = _enu_metres(lat, lon, float(np.mean(lat)), float(np.mean(lon)))
    return float(np.sqrt(pts[:, 0].var() + pts[:, 1].var()))


def berth_table(windows: pl.DataFrame) -> pl.DataFrame:
    out = []
    for port, sub in windows.group_by("port", maintain_order=True):
        port = port[0] if isinstance(port, tuple) else port
        rt_lat, rt_lon = sub["rt_lat"].to_numpy(), sub["rt_lon"].to_numpy()
        # Group on the real-time medians, exactly as in the published analysis.
        pts = _enu_metres(rt_lat, rt_lon, float(np.mean(rt_lat)), float(np.mean(rt_lon)))
        labels = _greedy_cluster(pts, EPS_M)
        for c in np.unique(labels):
            m = labels == c
            if int(m.sum()) < MIN_WINDOWS:
                continue
            sig_rt = _sigma_h(rt_lat[m], rt_lon[m])
            sig_ppp = _sigma_h(sub["ppp_lat"].to_numpy()[m], sub["ppp_lon"].to_numpy()[m])
            off = np.hypot(sub["rt_minus_ppp_e_m"].to_numpy()[m], sub["rt_minus_ppp_n_m"].to_numpy()[m])
            out.append({
                "port": port, "berth_lat": float(np.mean(rt_lat[m])), "berth_lon": float(np.mean(rt_lon[m])),
                "n_windows": int(m.sum()), "sigma_h_realtime_m": sig_rt, "sigma_h_ppp_m": sig_ppp,
                "rt_minus_ppp_horiz_median_m": float(np.median(off)),
                "anchorage_area": sig_rt > ANCHORAGE_SIGMA_M,
            })
    return pl.DataFrame(out).sort("sigma_h_ppp_m") if out else pl.DataFrame()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    rows = [r for d in list_days(cfg) for r in _windows_for_day(d, cfg)]
    if not rows:
        print("no stationary windows with PPP coverage; run ppp.compare first")
        return 1
    win = pl.DataFrame(rows)
    win.write_csv(tables_dir(cfg) / "T_ppp_berth_windows.csv")
    berths = berth_table(win)
    berths.write_csv(tables_dir(cfg) / "T_ppp_berth_repeatability.csv")
    with pl.Config(tbl_rows=40, tbl_cols=10, float_precision=3, tbl_width_chars=160):
        print(berths)
    keep = berths.filter(~pl.col("anchorage_area")) if berths.height else berths
    if keep.height:
        print(f"berths={keep.height}  windows={keep['n_windows'].sum()}  "
              f"median sigma_h: real-time {keep['sigma_h_realtime_m'].median():.3f} m, "
              f"PPP {keep['sigma_h_ppp_m'].median():.3f} m; "
              f"median real-time minus PPP offset {keep['rt_minus_ppp_horiz_median_m'].median():.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
