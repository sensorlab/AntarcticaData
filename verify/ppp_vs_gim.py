"""Test of the ionospheric hypothesis for the real-time minus PPP differences (Sect. 5.3).

The daily median 3-D difference between the real-time UBX-NAV-HPPOSLLH position and
the kinematic PPP solution (tables/T_ppp_residuals.csv, written by ppp.compare) is
larger at low latitudes. The paper states the hypothesis that the ionospheric delay
causes this, and the observation that would refute it: no significant positive rank
correlation between the daily median difference and the daily mean vertical TEC of a
global ionosphere map (GIM) at the ship's position, tested with the effective number
of independent days as in verify.daily_inference.

For every UTC day with PPP, the CODE final 1-h GIM (the rapid one if the final one is
missing) is interpolated to the ship position at hh:30 of every hour with track data
(bilinear in latitude and longitude, linear in time; verify.tec_vs_gim.gim_interp) and
averaged over the day. Because the vertical TEC decreases with latitude, the partial
correlations with the TEC (controlling for the day-median absolute latitude) and with
the latitude (controlling for the TEC) are also given. All tests use the functions of
verify.daily_inference: Spearman rho or partial rho, N_eff from the lag-1
autocorrelations of the rank series, the two-sided p-value and the 95 % interval
from the Fisher transformation with N_eff - 3 - k degrees of freedom (quoted in the
paper); moving-block bootstrap intervals for the block lengths BLOCKS are kept as a check.

Inputs:  tables/T_ppp_residuals.csv, derived/track/track.all.parquet, CODE GIMs
         (downloaded once into derived/ionex_cache, shared with verify.tec_vs_gim).
Outputs: tables/T_ppp_vs_gim_daily.csv  (day, d3d_p50_m, vtec_mean_tecu, abs_lat, ...)
         tables/T_ppp_vs_gim_tests.csv

Usage:
    python -m verify.ppp_vs_gim [--n-boot 10000] [--ppp PATH] [--track PATH]
"""
from __future__ import annotations

import gzip
import logging
import os
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import polars as pl

from analysis._common import derived_dir, load_config, tables_dir
from verify.daily_inference import BLOCKS, SEED, _rho, block_bootstrap_ci, ci_from_neff, n_eff, p_from_neff
from verify.tec_vs_gim import fetch_ionex, gim_interp, parse_ionex

log = logging.getLogger(__name__)

# AIUB serves the CODE products over HTTPS as well (redirect to its object store);
# verify.tec_vs_gim.fetch_ionex tries the FTP server.
HTTPS_URL = "https://www.aiub.unibe.ch/download/CODE/{yyyy}/COD0OPS{kind}_{yyyy}{ddd}0000_01D_01H_GIM.INX.gz"
HOUR_NS = 3_600_000_000_000

TESTS = [
    ("d3d_vs_vtec", "d3d_p50_m", "vtec_mean_tecu", None),
    ("vtec_vs_abs_lat", "vtec_mean_tecu", "abs_lat", None),
    ("d3d_vs_abs_lat", "d3d_p50_m", "abs_lat", None),
    ("d3d_vs_vtec|abs_lat", "d3d_p50_m", "vtec_mean_tecu", "abs_lat"),
    ("d3d_vs_abs_lat|vtec", "d3d_p50_m", "abs_lat", "vtec_mean_tecu"),
]


def fetch_gim(day: str, cache_dir: Path) -> Path | None:
    """CODE final (else rapid) 1-h GIM for one day, cached under the name used by tec_vs_gim."""
    d = datetime.strptime(day, "%Y%m%d")
    doy, yy = d.timetuple().tm_yday, d.year % 100
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"igsg{doy:03d}0.{yy:02d}i"
    if path.exists() and path.stat().st_size > 1000:
        return path
    for kind in ("FIN", "RAP"):
        url = HTTPS_URL.format(yyyy=d.year, ddd=f"{doy:03d}", kind=kind)
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                text = gzip.decompress(r.read()).decode("ascii", "replace")
        except Exception as exc:  # noqa: BLE001
            log.info("%s: %s", url, str(exc)[:80])
            continue
        if "END OF FILE" in text:
            path.write_text(text)
            return path
    return fetch_ionex(day, cache_dir)


def hourly_positions(track: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Track sample nearest to hh:30 of every hour, and the day-median |latitude|."""
    lf = (pl.scan_parquet(track)
          .filter((pl.col("t_ns") > 1_700_000_000_000_000_000) & (pl.col("lat").abs() <= 90.0))
          .select(["t_ns", "lat", "lon"]))
    hourly = (lf.with_columns((pl.col("t_ns") // HOUR_NS).alias("hour"))
              .with_columns((pl.col("t_ns") - (pl.col("hour") * HOUR_NS + HOUR_NS // 2)).abs().alias("dmid"))
              .sort("dmid").group_by("hour").first().sort("hour")
              .with_columns(pl.from_epoch(pl.col("hour") * HOUR_NS + HOUR_NS // 2, time_unit="ns").alias("t_mid"))
              .with_columns(pl.col("t_mid").dt.strftime("%Y%m%d").alias("day"))
              .collect())
    geo = (lf.with_columns(pl.from_epoch(pl.col("t_ns"), time_unit="ns").dt.strftime("%Y%m%d").alias("day"))
           .group_by("day").agg(pl.col("lat").abs().median().alias("abs_lat"))
           .collect())
    return hourly, geo


def daily_table(ppp_csv: Path, track: Path, cache_dir: Path) -> pl.DataFrame:
    ppp = (pl.read_csv(ppp_csv, schema_overrides={"day": pl.Utf8})
           .filter(pl.col("n_matched") > 0).select(["day", "n_matched", "d3d_p50_m"]))
    hourly, geo = hourly_positions(track)
    rows = []
    for day in ppp["day"].to_list():
        path = fetch_gim(day, cache_dir)
        h = hourly.filter(pl.col("day") == day)
        if path is None or h.is_empty():
            log.warning("%s: no GIM or no track, skipped", day)
            continue
        gim = parse_ionex(path)
        vtec = [gim_interp(gim, t, la, lo)
                for t, la, lo in zip(h["t_mid"].to_list(), h["lat"].to_list(), h["lon"].to_list())]
        rows.append({"day": day, "n_hours": len(vtec), "vtec_mean_tecu": float(np.mean(vtec))})
    return (ppp.join(pl.DataFrame(rows), on="day", how="inner")
            .join(geo, on="day", how="left").sort("day"))


@click.command()
@click.option("--n-boot", default=10_000, show_default=True)
@click.option("--ppp", "ppp_csv", default=None, help="Default: tables/T_ppp_residuals.csv.")
@click.option("--track", default=None, help="Default: derived/track/track.all.parquet.")
@click.option("--gim-cache", default=None, help="Default: derived/ionex_cache.")
@click.option("--out-dir", default=None, help="Default: the tables directory.")
def main(n_boot: int, ppp_csv: str | None, track: str | None, gim_cache: str | None,
         out_dir: str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    # The GIM epochs and the track times are naive UTC datetimes; gim_interp compares
    # their .timestamp(), which uses the local time zone, so run in UTC.
    os.environ["TZ"] = "UTC"
    time.tzset()
    cfg = load_config()
    out = Path(out_dir) if out_dir else tables_dir(cfg)
    df = daily_table(Path(ppp_csv) if ppp_csv else tables_dir(cfg) / "T_ppp_residuals.csv",
                     Path(track) if track else derived_dir(cfg) / "track" / "track.all.parquet",
                     Path(gim_cache) if gim_cache else derived_dir(cfg) / "ionex_cache")
    df.write_csv(out / "T_ppp_vs_gim_daily.csv")
    rng = np.random.default_rng(SEED)
    rows = []
    for name, xc, yc, zc in TESTS:
        cols = [xc, yc] + ([zc] if zc else [])
        sub = df.select(cols).drop_nulls().filter(pl.all_horizontal(pl.col(c).is_finite() for c in cols))
        x, y = sub[xc].to_numpy().astype(float), sub[yc].to_numpy().astype(float)
        z = sub[zc].to_numpy().astype(float) if zc else None
        rho = _rho(x, y, z)
        ne = n_eff(x, y, z)
        row = {"test": name, "n_days": len(x), "rho": rho, "n_eff": ne,
               "p_neff": p_from_neff(rho, ne, 1 if zc else 0)}
        row["ci95_lo_neff"], row["ci95_hi_neff"] = ci_from_neff(rho, ne, 1 if zc else 0)
        for b in BLOCKS:
            lo, hi = block_bootstrap_ci(x, y, z, b, n_boot, rng)
            row[f"ci95_lo_L{b}"], row[f"ci95_hi_L{b}"] = lo, hi
        rows.append(row)
    res = pl.DataFrame(rows)
    res.write_csv(out / "T_ppp_vs_gim_tests.csv")
    log.info("%d days; daily mean vTEC %.1f-%.1f TECU", df.height,
             df["vtec_mean_tecu"].min(), df["vtec_mean_tecu"].max())
    with pl.Config(tbl_cols=20, float_precision=4, tbl_width_chars=220):
        print(res)


if __name__ == "__main__":
    main()
