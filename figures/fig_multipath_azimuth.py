"""Azimuth-resolved code multipath (ESSD review, R1).

Bins the Estey & Meertens M1 residual (``derived/multipath``) by elevation
and azimuth and plots the RMS on a sky plot:

  (a) Earth-frame azimuth, all epochs;
  (b) ship-relative azimuth (satellite azimuth minus course over ground)
      for underway epochs (ground speed > UNDERWAY_M_S), where course over
      ground approximates the ship heading, so persistent sectors of high
      multipath map to the superstructure around the antenna;
  (c) Earth-frame azimuth, stationary epochs (ground speed < STATIONARY_M_S).

Sums of M1^2 and counts are accumulated per day so the cruise can be
processed in constant memory. The binned table is written alongside.

Usage:
    python -m figures.fig_multipath_azimuth [--days 20250930,20260415]
Outputs:
    figures/output/fig_multipath_azimuth.pdf
    tables/T_multipath_azimuth.csv       binned M1 RMS and sample counts
    tables/T_multipath_sectors.csv       ship-relative sectors (bow, starboard,
                                         stern, port; 60 deg wide), pooled M1 RMS
                                         below and above 30 deg elevation
    tables/T_multipath_az_coverage.csv   samples below 60 deg elevation per 10 deg
                                         azimuth bin, relative to the median bin,
                                         ship-relative (obstructed sectors) and in
                                         the Earth frame (sky coverage, for control)
    tables/T_multipath_az_sensitivity.csv the ship-relative coverage with the course
                                         over ground offset by +-5 and +-10 deg and
                                         with speed thresholds of 1 and 4 m/s, to
                                         test the course as a proxy for the heading
    tables/T_multipath_cno_az.csv        ship-relative L1 C/N0 of the dual-band
                                         signals minus the median of the same
                                         constellation and elevation bin (median per
                                         10 deg azimuth bin, below 30 deg and 30-60 deg)
"""
from __future__ import annotations

import logging

import click
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from analysis._common import derived_dir, figures_dir, list_days, load_config, read_parquet, staged_path, tables_dir
from figures._helpers import apply_style

log = logging.getLogger(__name__)

AZ_STEP = 10.0
EL_STEP = 5.0
UNDERWAY_M_S = 2.0
STATIONARY_M_S = 0.5
MIN_BIN_N = 200
AZ_EDGES = np.arange(0.0, 360.0 + AZ_STEP, AZ_STEP)
EL_EDGES = np.arange(0.0, 90.0 + EL_STEP, EL_STEP)
PANELS = ("earth_all", "ship_underway", "earth_stationary")
# Sensitivity of the ship-relative map to the course over ground as heading proxy.
HEADING_OFFSETS = (-10.0, -5.0, 5.0, 10.0)
SPEED_THRESHOLDS = (1.0, 4.0)
GNSS_INDEX = {0: 0, 2: 1, 3: 2}  # GPS, Galileo, BeiDou (u-blox gnssId)
CNO_MAX = 64                      # C/N0 is an integer in dB-Hz


def _empty() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    shape = (len(EL_EDGES) - 1, len(AZ_EDGES) - 1)
    return {p: (np.zeros(shape), np.zeros(shape, dtype=np.int64)) for p in PANELS}


def _add(acc, key, el, az, m1) -> None:
    s, n = acc[key]
    s += np.histogram2d(el, az, bins=[EL_EDGES, AZ_EDGES], weights=m1 * m1)[0]
    n += np.histogram2d(el, az, bins=[EL_EDGES, AZ_EDGES])[0].astype(np.int64)


def accumulate(days: list[str], cfg: dict):
    """M1 sums and counts per panel, counts of the heading-proxy variants and
    L1 C/N0 histograms (constellation x elevation bin x ship-relative azimuth bin)."""
    acc = _empty()
    shape = (len(EL_EDGES) - 1, len(AZ_EDGES) - 1)
    variants = {f"offset{o:+.0f}": np.zeros(shape, dtype=np.int64) for o in HEADING_OFFSETS}
    variants.update({f"speed>{v:g}": np.zeros(shape, dtype=np.int64) for v in SPEED_THRESHOLDS})
    cno_hist = np.zeros((len(GNSS_INDEX),) + shape + (CNO_MAX,), dtype=np.int64)
    for d in days:
        mp_path = derived_dir(cfg) / "multipath" / f"{d}.multipath.parquet"
        if not mp_path.exists():
            continue
        mp = read_parquet(mp_path)
        if "azim" not in mp.columns:
            log.warning("%s: multipath parquet has no azim column; rerun analysis.multipath", d)
            continue
        pvt = read_parquet(staged_path(d, "nav_pvt", cfg)).select(
            ["t_ns", pl.col("gSpeed_mm_s").cast(pl.Float64) / 1e3, pl.col("headMot_1e5").cast(pl.Float64) / 1e5]
        ).rename({"gSpeed_mm_s": "speed", "headMot_1e5": "cog"})
        df = mp.select(["t_ns", "gnssId", "M1", "elev", "azim", "cno1"]).drop_nulls().join(pvt, on="t_ns", how="inner")
        df = df.filter(pl.col("M1").is_finite() & (pl.col("elev") > 0))
        el = df["elev"].to_numpy().astype(float)
        az = df["azim"].to_numpy().astype(float) % 360.0
        m1 = df["M1"].to_numpy()
        sp = df["speed"].to_numpy()
        rel = (az - df["cog"].to_numpy()) % 360.0
        _add(acc, "earth_all", el, az, m1)
        mv = sp > UNDERWAY_M_S
        _add(acc, "ship_underway", el[mv], rel[mv], m1[mv])
        st = sp < STATIONARY_M_S
        _add(acc, "earth_stationary", el[st], az[st], m1[st])
        cog = df["cog"].to_numpy()
        for o in HEADING_OFFSETS:
            variants[f"offset{o:+.0f}"] += np.histogram2d(
                el[mv], (az[mv] - cog[mv] - o) % 360.0, bins=[EL_EDGES, AZ_EDGES])[0].astype(np.int64)
        for v in SPEED_THRESHOLDS:
            m = sp > v
            variants[f"speed>{v:g}"] += np.histogram2d(el[m], rel[m], bins=[EL_EDGES, AZ_EDGES])[0].astype(np.int64)
        gid = df["gnssId"].to_numpy()
        keep = mv & np.isin(gid, list(GNSS_INDEX))
        gi = np.vectorize(GNSS_INDEX.get)(gid[keep]) if keep.any() else np.zeros(0, dtype=int)
        eb = np.minimum((el[keep] // EL_STEP).astype(int), shape[0] - 1)
        ab = (rel[keep] // AZ_STEP).astype(int) % shape[1]
        cn = np.clip(df["cno1"].to_numpy()[keep].astype(int), 0, CNO_MAX - 1)
        np.add.at(cno_hist, (gi, eb, ab, cn), 1)
        log.info("%s: %d epochs (%d underway, %d stationary)", d, len(m1), mv.sum(), st.sum())
    return acc, variants, cno_hist


def _rms(acc, key):
    s, n = acc[key]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.sqrt(s / n)
    r[n < MIN_BIN_N] = np.nan
    return r, n


def write_table(acc, path) -> None:
    rows = []
    for key in PANELS:
        r, n = _rms(acc, key)
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                if n[i, j]:
                    rows.append({"frame": key, "el_lo": EL_EDGES[i], "az_lo": AZ_EDGES[j],
                                 "m1_rms_m": float(np.sqrt(acc[key][0][i, j] / n[i, j])), "n": int(n[i, j])})
    pl.DataFrame(rows).write_csv(path)


# Ship-relative sectors, 60 deg wide, centred on the bow, starboard, stern and port.
SECTORS = {"bow": (330.0, 30.0), "starboard": (60.0, 120.0), "stern": (150.0, 210.0), "port": (240.0, 300.0)}


def _in_sector(az_lo: pl.Expr, lo: float, hi: float) -> pl.Expr:
    return ((az_lo >= lo) & (az_lo < hi)) if lo < hi else ((az_lo >= lo) | (az_lo < hi))


def sector_table(table: pl.DataFrame) -> pl.DataFrame:
    """Pooled M1 RMS per ship-relative sector, below and above 30 deg elevation.

    Pooling uses all samples of the bins in the sector: sqrt(sum(n rms^2) / sum(n)).
    """
    ship = table.filter(pl.col("frame") == "ship_underway")
    rows = []
    for name, (lo, hi) in SECTORS.items():
        for band, cond in (("below_30", pl.col("el_lo") < 30.0), ("above_30", pl.col("el_lo") >= 30.0)):
            sub = ship.filter(_in_sector(pl.col("az_lo"), lo, hi) & cond)
            n = int(sub["n"].sum())
            ss = float((sub["n"] * sub["m1_rms_m"] ** 2).sum())
            rows.append({"sector": name, "az_from": lo, "az_to": hi, "elevation": band,
                         "n": n, "m1_rms_m": float(np.sqrt(ss / n)) if n else float("nan")})
    return pl.DataFrame(rows)


def az_coverage_table(table: pl.DataFrame, el_max: float = 60.0) -> pl.DataFrame:
    """Samples below ``el_max`` per azimuth bin, relative to the median bin.

    The ship-relative rows show obstructed sectors. The Earth-frame rows of all
    epochs are the control: a sector that is fixed to the ship is smeared out in
    the Earth frame, where only the sky coverage of the orbits remains.
    """
    out = []
    for frame in ("ship_underway", "earth_all"):
        sub = table.filter((pl.col("frame") == frame) & (pl.col("el_lo") < el_max))
        per_az = (pl.DataFrame({"az_lo": AZ_EDGES[:-1]})
                  .join(sub.group_by("az_lo").agg(pl.col("n").sum()), on="az_lo", how="left")
                  .with_columns(pl.col("n").fill_null(0)))
        med = float(per_az["n"].median())
        out.append(per_az.with_columns(pl.lit(frame).alias("frame"),
                                       (pl.col("n") / med).alias("share_of_median_bin")).sort("az_lo"))
    return pl.concat(out).select(["frame", "az_lo", "n", "share_of_median_bin"])


def sensitivity_table(acc, variants, el_max: float = 60.0) -> pl.DataFrame:
    """Ship-relative samples below ``el_max`` per azimuth bin for each heading-proxy variant."""
    below = EL_EDGES[:-1] < el_max
    rows = []
    for name, counts in [(f"speed>{UNDERWAY_M_S:g}", acc["ship_underway"][1])] + list(variants.items()):
        per_az = counts[below].sum(axis=0)
        med = float(np.median(per_az))
        rows += [{"variant": name, "az_lo": AZ_EDGES[j], "n": int(n),
                  "share_of_median_bin": n / med if med else float("nan")} for j, n in enumerate(per_az)]
    return pl.DataFrame(rows)


def _hist_median(h: np.ndarray) -> int:
    c = np.cumsum(h)
    return int(np.searchsorted(c, c[-1] / 2.0)) if c[-1] else -1


def cno_az_table(cno_hist: np.ndarray) -> pl.DataFrame:
    """Median L1 C/N0 residual per ship-relative azimuth bin and elevation band.

    The residual is C/N0 minus the median C/N0 of the same constellation and
    5 deg elevation bin over all azimuths, so the elevation dependence and the
    differences between constellations do not enter the azimuth structure.
    """
    n_g, n_el, n_az, _ = cno_hist.shape
    ref = np.array([[_hist_median(cno_hist[g, i].sum(axis=0)) for i in range(n_el)] for g in range(n_g)])
    offset = CNO_MAX - 1
    rows = []
    for band, (lo, hi) in {"below_30": (0.0, 30.0), "30_to_60": (30.0, 60.0)}.items():
        el_bins = [i for i in range(n_el) if lo <= EL_EDGES[i] < hi]
        for j in range(n_az):
            dh = np.zeros(2 * CNO_MAX - 1, dtype=np.int64)
            for g in range(n_g):
                for i in el_bins:
                    if ref[g, i] >= 0:
                        np.add.at(dh, np.arange(CNO_MAX) - ref[g, i] + offset, cno_hist[g, i, j])
            n = int(dh.sum())
            rows.append({"elevation": band, "az_lo": AZ_EDGES[j], "n": n,
                         "dcno1_median_db": (_hist_median(dh) - offset) if n else None})
    return pl.DataFrame(rows)


def build_figure(acc, out_path) -> None:
    apply_style()
    titles = {
        "earth_all": "(a) Earth frame, all epochs",
        "ship_underway": f"(b) Ship-relative, underway (> {UNDERWAY_M_S:g} m s$^{{-1}}$)",
        "earth_stationary": f"(c) Earth frame, stationary (< {STATIONARY_M_S:g} m s$^{{-1}}$)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), subplot_kw={"projection": "polar"}, constrained_layout=True)
    theta = np.deg2rad(AZ_EDGES)
    rad = 90.0 - EL_EDGES
    vmax = 1.5
    for ax, key in zip(axes, PANELS):
        r, n = _rms(acc, key)
        pc = ax.pcolormesh(theta, rad, r, cmap="viridis", vmin=0.0, vmax=vmax, shading="flat")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 90)
        ax.set_yticks([30, 60, 80])
        ax.set_yticklabels(["60°", "30°", "10°"], fontsize=7)
        ax.set_title(titles[key], fontsize=9, pad=14)
        ax.tick_params(axis="x", pad=1)
        if key == "ship_underway":
            ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
            ax.set_xticklabels(["bow", "stbd", "stern", "port"], fontsize=8)
        else:
            ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
            ax.set_xticklabels(["N", "E", "S", "W"], fontsize=8)
        n_tot = int(n.sum())
        ax.text(0.5, -0.16, f"n = {n_tot / 1e6:.1f} M", transform=ax.transAxes, ha="center", fontsize=7)
    cb = fig.colorbar(pc, ax=axes, shrink=0.75, pad=0.02, extend="max")
    cb.set_label("M1 RMS (m)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


@click.command()
@click.option("--days", default=None, help="Comma-separated YYYYMMDD; default all staged days.")
def main(days: str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    day_list = days.split(",") if days else list_days(cfg)
    acc, variants, cno_hist = accumulate(day_list, cfg)
    write_table(acc, tables_dir(cfg) / "T_multipath_azimuth.csv")
    table = pl.read_csv(tables_dir(cfg) / "T_multipath_azimuth.csv")
    sectors = sector_table(table)
    sectors.write_csv(tables_dir(cfg) / "T_multipath_sectors.csv")
    coverage = az_coverage_table(table)
    coverage.write_csv(tables_dir(cfg) / "T_multipath_az_coverage.csv")
    sens = sensitivity_table(acc, variants)
    sens.write_csv(tables_dir(cfg) / "T_multipath_az_sensitivity.csv")
    cno = cno_az_table(cno_hist)
    cno.write_csv(tables_dir(cfg) / "T_multipath_cno_az.csv")
    print(sectors)
    for frame, sub in coverage.group_by("frame", maintain_order=True):
        k = sub["share_of_median_bin"].arg_min()
        print(f"coverage {frame[0]:14s} lowest bin {sub['az_lo'][k]:.0f} deg: {100 * sub['share_of_median_bin'][k]:.1f} % of the median bin")
    for name, sub in sens.group_by("variant", maintain_order=True):
        print(f"variant {name[0]:10s} bins below 50 % of the median bin: {sub.filter(pl.col('share_of_median_bin') < 0.5)['az_lo'].to_list()}")
    print(cno.filter(pl.col("elevation") == "below_30").sort("dcno1_median_db").head(8))
    out = figures_dir(cfg) / "output" / "fig_multipath_azimuth.pdf"
    build_figure(acc, out)
    for key in PANELS:
        r, n = _rms(acc, key)
        hi = EL_EDGES[:-1] >= 30
        s, c = acc[key][0][hi].sum(), acc[key][1][hi].sum()
        print(f"{key:18s} n={int(n.sum()):>10d}  M1 RMS el>=30: {np.sqrt(s / c) if c else float('nan'):.3f} m")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
