"""Autocorrelation-aware inference on day-aggregated series (ESSD review, R1).

Daily values along a ship track are serially correlated (consecutive days
sample neighbouring latitudes, weather and operating modes), so nominal
p-values that assume n independent days are optimistic. For every test
reported in the paper this script gives:

  * the point estimate (Spearman rho, or partial Spearman rho);
  * the effective sample size N_eff = N (1 - r1 r2) / (1 + r1 r2) from the
    lag-1 autocorrelations r1, r2 of the two rank series (Bretherton et
    al. 1999), the corresponding two-sided p-value and the 95 % interval from
    the Fisher transformation with N_eff - 3 - k degrees of freedom (the values
    quoted in the paper);
  * 95 % moving-block-bootstrap intervals over days in cruise order (Kuensch
    1989) for the block lengths BLOCKS, kept as a check; the AR(1)
    calibration (--calibrate) shows that they are too narrow for series as
    persistent as these, so the paper does not use them.

Tests:
  slips_vs_abs_maglat   daily cycle-slip count vs day-median |geomagnetic lat|  (Sect. 5.5)
  sliprate_vs_abs_maglat  slip ratio per 1000 obs vs |geomagnetic lat|
  jamL5_vs_abs_lat      L5-block daily mean jamInd vs day-median |lat|        (Sect. 5.7)
  jamL5_vs_temp         ... vs daily mean receiver temperature
  lat_vs_temp           |lat| vs temperature (collinearity of the regressors)
  jamL5_vs_abs_lat|temp partial, controlling for temperature
  jamL5_vs_temp|abs_lat partial, controlling for |lat|

Inputs: tables/T4_daily_stats.parquet, derived/track/track.all.parquet and
staging/<day>/mon_rf.parquet + mon_sys.parquet (via analysis.jamming_confound).

Outputs:
    tables/T_daily_inference.csv
    tables/T_daily_series.csv  (the per-day table the tests run on)

Usage:
    python -m verify.daily_inference [--n-boot 10000]
    python -m verify.daily_inference --calibrate [--phi 0.9]
                                     # false-positive rates of the nominal test, the
                                     # N_eff test and the block bootstrap on pairs of
                                     # independent AR(1) series
"""
from __future__ import annotations

import logging

import click
import numpy as np
import polars as pl
from scipy import stats

from analysis._common import derived_dir, load_config, read_parquet, tables_dir
from analysis.jamming_confound import build_table as build_jam_table

log = logging.getLogger(__name__)

# Centred-dipole north geomagnetic pole, IGRF-14 epoch 2025.
POLE_LAT_DEG = 80.7
POLE_LON_DEG = -72.7
BLOCKS = (3, 7, 14)
BLOCK_MAIN = 7
SEED = 20261002


def geomag_lat(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    la, lo = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    pla, plo = np.deg2rad(POLE_LAT_DEG), np.deg2rad(POLE_LON_DEG)
    s = np.sin(la) * np.sin(pla) + np.cos(la) * np.cos(pla) * np.cos(lo - plo)
    return np.rad2deg(np.arcsin(np.clip(s, -1.0, 1.0)))


def daily_series(cfg: dict) -> pl.DataFrame:
    t4 = read_parquet(tables_dir(cfg) / "T4_daily_stats.parquet").select(
        ["day", "cycle_slip_count", "slip_ratio_per_1000"]
    )
    track = read_parquet(derived_dir(cfg) / "track" / "track.all.parquet").filter(
        (pl.col("t_ns") > 1_700_000_000_000_000_000) & (pl.col("lat").abs() <= 90.0)
    )
    track = track.with_columns(
        pl.Series("mlat", geomag_lat(track["lat"].to_numpy(), track["lon"].to_numpy())),
        pl.from_epoch(pl.col("t_ns"), time_unit="ns").dt.strftime("%Y%m%d").alias("day"),
    )
    geo = track.group_by("day").agg(
        pl.col("lat").abs().median().alias("abs_lat"),
        pl.col("mlat").abs().median().alias("abs_mlat"),
    )
    jam = build_jam_table(cfg).select(["day", "mean_jam_l5", "mean_temp_C"])
    return t4.join(geo, on="day", how="inner").join(jam, on="day", how="left").sort("day")


def _rank_partial(x, y, z=None):
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    if z is None:
        return rx, ry
    Z = np.column_stack([stats.rankdata(z), np.ones(len(z))])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return ex, ey


def _rho(x, y, z=None) -> float:
    a, b = _rank_partial(x, y, z)
    return float(np.corrcoef(a, b)[0, 1])


def _lag1(v: np.ndarray) -> float:
    v = v - v.mean()
    return float(np.sum(v[1:] * v[:-1]) / np.sum(v * v))


def n_eff(x, y, z=None) -> float:
    a, b = _rank_partial(x, y, z)
    r1, r2 = _lag1(a), _lag1(b)
    n = len(a)
    return float(np.clip(n * (1 - r1 * r2) / (1 + r1 * r2), 3.0, n))


def p_from_neff(rho: float, neff: float, k: int) -> float:
    dof = max(neff - 2 - k, 1.0)
    t = rho * np.sqrt(dof / max(1e-12, 1 - rho * rho))
    return float(2 * stats.t.sf(abs(t), dof))


def ci_from_neff(rho: float, neff: float, k: int, level: float = 0.95) -> tuple[float, float]:
    """Confidence interval for rho from the Fisher transformation with N_eff - 3 - k
    degrees of freedom (k control variables); consistent with the N_eff test."""
    se = 1.0 / np.sqrt(max(neff - 3 - k, 1.0))
    z = np.arctanh(np.clip(rho, -0.999999, 0.999999))
    q = stats.norm.ppf(0.5 + level / 2)
    return float(np.tanh(z - q * se)), float(np.tanh(z + q * se))


def block_bootstrap_ci(x, y, z, block: int, n_boot: int, rng) -> tuple[float, float]:
    n = len(x)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    out = np.empty(n_boot)
    for i in range(n_boot):
        j = idx[i]
        out[i] = _rho(x[j], y[j], None if z is None else z[j])
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


TESTS = [
    ("slips_vs_abs_maglat", "cycle_slip_count", "abs_mlat", None),
    ("sliprate_vs_abs_maglat", "slip_ratio_per_1000", "abs_mlat", None),
    ("jamL5_vs_abs_lat", "mean_jam_l5", "abs_lat", None),
    ("jamL5_vs_temp", "mean_jam_l5", "mean_temp_C", None),
    ("lat_vs_temp", "abs_lat", "mean_temp_C", None),
    ("jamL5_vs_abs_lat|temp", "mean_jam_l5", "abs_lat", "mean_temp_C"),
    ("jamL5_vs_temp|abs_lat", "mean_jam_l5", "mean_temp_C", "abs_lat"),
]


def calibration(n: int = 216, phi: float = 0.9, n_sim: int = 1000, seed: int = SEED,
                n_boot_cal: int = 2000) -> dict[str, float]:
    """Rejection rate at the 5 % level for pairs of independent AR(1) series.

    Both series have length ``n`` and lag-1 coefficient ``phi`` (0.9 gives a
    median N_eff of about 30, within the range 11-53 of the daily series of the
    cruise). Since the series
    are independent, every rejection is a false positive: a correct test
    rejects in about 5 % of the cases. Besides the nominal and the N_eff test,
    the moving-block bootstrap is counted as rejecting when its 95 % interval
    excludes zero, for each block length in BLOCKS (``n_boot_cal`` resamples;
    the resampling uses its own random stream, so the AR(1) series and the
    rates of the two t tests do not depend on it; ``n_boot_cal=0`` skips it).
    """
    rng = np.random.default_rng(seed)
    rng_boot = np.random.default_rng(seed + 1)

    def ar1() -> np.ndarray:
        e = rng.normal(size=n)
        x = np.empty(n)
        x[0] = e[0] / np.sqrt(1.0 - phi * phi)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + e[i]
        return x

    rej_nom = rej_eff = rej_eff_ci = 0
    rej_boot = {b: 0 for b in BLOCKS}
    ne_all = []
    for _ in range(n_sim):
        x, y = ar1(), ar1()
        r = _rho(x, y)
        ne = n_eff(x, y)
        ne_all.append(ne)
        rej_nom += p_from_neff(r, n, 0) < 0.05
        rej_eff += p_from_neff(r, ne, 0) < 0.05
        lo_e, hi_e = ci_from_neff(r, ne, 0)
        rej_eff_ci += lo_e > 0 or hi_e < 0
        if n_boot_cal:
            for b in BLOCKS:
                lo, hi = block_bootstrap_ci(x, y, None, b, n_boot_cal, rng_boot)
                rej_boot[b] += lo > 0 or hi < 0
    out = {"n": n, "phi": phi, "n_sim": n_sim, "reject_nominal": rej_nom / n_sim,
           "reject_neff": rej_eff / n_sim, "reject_neff_interval": rej_eff_ci / n_sim,
           "n_eff_median": float(np.median(ne_all))}
    if n_boot_cal:
        out.update({f"reject_bootstrap_L{b}": rej_boot[b] / n_sim for b in BLOCKS})
    return out


@click.command()
@click.option("--n-boot", default=10_000, show_default=True)
@click.option("--calibrate", is_flag=True, default=False,
              help="Only run the AR(1) calibration of the tests and print the rejection rates.")
@click.option("--phi", default=0.9, show_default=True, help="Lag-1 coefficient of the AR(1) series (--calibrate).")
def main(n_boot: int, calibrate: bool, phi: float) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    if calibrate:
        print(calibration(phi=phi))
        return
    cfg = load_config()
    df = daily_series(cfg)
    df.write_csv(tables_dir(cfg) / "T_daily_series.csv")
    rng = np.random.default_rng(SEED)
    rows = []
    for name, xc, yc, zc in TESTS:
        cols = [xc, yc] + ([zc] if zc else [])
        sub = df.select(cols).drop_nulls().filter(pl.all_horizontal(pl.col(c).is_finite() for c in cols))
        x, y = sub[xc].to_numpy().astype(float), sub[yc].to_numpy().astype(float)
        z = sub[zc].to_numpy().astype(float) if zc else None
        n = len(x)
        if n < 10:
            log.warning("%s: only %d days, skipping", name, n)
            continue
        rho = _rho(x, y, z)
        p_nom = p_from_neff(rho, n, 1 if zc else 0)
        ne = n_eff(x, y, z)
        row = {"test": name, "n_days": n, "rho": rho, "p_nominal": p_nom, "n_eff": ne,
               "p_neff": p_from_neff(rho, ne, 1 if zc else 0)}
        row["ci95_lo_neff"], row["ci95_hi_neff"] = ci_from_neff(rho, ne, 1 if zc else 0)
        for b in BLOCKS:
            lo, hi = block_bootstrap_ci(x, y, z, b, n_boot, rng)
            row[f"ci95_lo_L{b}"], row[f"ci95_hi_L{b}"] = lo, hi
        rows.append(row)
    out = pl.DataFrame(rows)
    out.write_csv(tables_dir(cfg) / "T_daily_inference.csv")
    with pl.Config(tbl_cols=20, float_precision=3, tbl_width_chars=200):
        print(out)


if __name__ == "__main__":
    main()
