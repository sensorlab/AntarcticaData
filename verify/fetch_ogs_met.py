"""Download R/V Laura Bassi surface meteorology for the cruise window (ESSD review, R2).

The ship's Vaisala WXT536 weather transmitter (air pressure, temperature,
relative humidity) is served at ~1 min cadence, with the ship position, by
the OGS dashboard history endpoint behind https://laurabassi.ogs.it/lb_plot.html.
These are the auxiliary observations a PWV retrieval needs: surface pressure
for the zenith hydrostatic delay and surface temperature for the weighted
mean temperature Tm.

Queries are issued one UTC day at a time and cached as JSON, then merged to

    derived/ogs_met/ogs_wxt536.parquet   (t_ns, lat, lon, press_hPa, temp_C, rh_pct)

The data belong to OGS; check their terms before redistributing.

Usage:
    python -m verify.fetch_ogs_met [--start 2025-09-26 --end 2026-04-29]
"""
from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone

import click
import polars as pl

from analysis._common import derived_dir, load_config, write_parquet

log = logging.getLogger(__name__)

HISTORY_URL = "https://laurabassi.ogs.it/lb_history.php"
VARIABLES = {"press_hPa": "wxt536,press", "temp_C": "wxt536,temp", "rh_pct": "wxt536,rh"}
MAX_AGE_S = 150  # same staleness limit the dashboard applies


def _fetch_day(var_path: str, day: date, cache) -> list[dict]:
    path = cache / f"{var_path.replace(',', '_')}_{day:%Y%m%d}.json"
    if not path.exists():
        q = urllib.parse.urlencode({
            "from": f"{day:%Y-%m-%d} 00:00:00",
            "to": f"{day + timedelta(days=1):%Y-%m-%d} 00:00:00",
            "var": var_path, "age": "wxt536,age", "max_age": MAX_AGE_S,
        })
        with urllib.request.urlopen(f"{HISTORY_URL}?{q}", timeout=120) as resp:
            path.write_bytes(resp.read())
        time.sleep(0.5)  # be gentle with the OGS server
    return json.loads(path.read_text()).get("result", [])


def _to_frame(rows: list[dict], col: str) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema={"t_ns": pl.Int64, "lat": pl.Float64, "lon": pl.Float64, col: pl.Float64})
    return pl.DataFrame({
        "t_ns": [int(datetime.strptime(r["time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1e9) for r in rows],
        "lat": [float(r["lat"]) for r in rows],
        "lon": [float(r["lon"]) for r in rows],
        col: [float(r["val"]) for r in rows],
    })


@click.command()
@click.option("--start", default="2025-09-26")
@click.option("--end", default="2026-04-29")
def main(start: str, end: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    out_dir = derived_dir(cfg) / "ogs_met"
    cache = out_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    merged = None
    for col, var_path in VARIABLES.items():
        parts = [_to_frame(_fetch_day(var_path, d, cache), col) for d in days]
        df = pl.concat(parts).unique(subset="t_ns").sort("t_ns")
        log.info("%s: %d samples", col, df.height)
        merged = df if merged is None else merged.join(df.drop(["lat", "lon"]), on="t_ns", how="full", coalesce=True)
    merged = merged.sort("t_ns")
    write_parquet(merged, out_dir / "ogs_wxt536.parquet")
    per_day = merged.with_columns(pl.from_epoch("t_ns", time_unit="ns").dt.date().alias("d")).group_by("d").len()
    print(f"samples={merged.height}  days with data={per_day.height}/{len(days)}  "
          f"median samples/day={per_day['len'].median()}")


if __name__ == "__main__":
    main()
