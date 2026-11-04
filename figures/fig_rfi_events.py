"""Figure: cruise-wide RFI events.

Single-panel map for the paper:

* Cruise track on a Robinson world projection with detected RFI events
  plotted as markers sized/coloured by L1 jamInd peak. The per-day
  jamInd peak time-series is tabulated in main.tex Table 8 and is
  therefore not duplicated in this figure.

Inputs:
* ``work/tables/T_rfi_events.csv`` — produced by
  ``verify.locate_rfi_events --scan --l1-threshold 40 --csv-out ...``
* ``work/derived/track/track.all.parquet`` — for the background track

Output:
* ``work/figures/output/fig13_rfi_events.pdf``

CLI:
    python -m figures.fig_rfi_events
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from analysis._common import figures_dir, load_config, read_parquet, resolve_path

log = logging.getLogger(__name__)

# Hand-curated location labels — geographic centroids you can edit.
KNOWN_PORTS = [
    {"name": "Trieste",          "lat": 45.61, "lon": 13.81, "radius_km": 30},
    {"name": "Adriatic",         "lat": 43.36, "lon": 14.83, "radius_km": 80},
    {"name": "Panama Canal",     "lat":  8.87, "lon": -79.54, "radius_km": 60},
    {"name": "Lyttelton (NZ)",   "lat": -43.61, "lon": 172.72, "radius_km": 60},
    {"name": "Mario Zucchelli",  "lat": -74.69, "lon": 164.12, "radius_km": 30},
]


def _label_for(lat: float, lon: float) -> str:
    """Return name of the closest known port within radius_km, else ``''``."""
    best = ""
    best_d = float("inf")
    for p in KNOWN_PORTS:
        # equirectangular small-circle approx is fine for naming
        dlat = (lat - p["lat"]) * 111
        dlon = (lon - p["lon"]) * 111 * np.cos(np.deg2rad((lat + p["lat"]) / 2))
        d = (dlat ** 2 + dlon ** 2) ** 0.5
        if d < p["radius_km"] and d < best_d:
            best = p["name"]
            best_d = d
    return best


def render(out_path: Path, cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    rfi_csv = resolve_path(cfg["paths"]["tables"]) / "T_rfi_events.csv"
    track_p = resolve_path(cfg["paths"]["derived"]) / "track" / "track.all.parquet"
    if not rfi_csv.exists():
        raise FileNotFoundError(
            f"{rfi_csv} not found — run "
            "`python -m verify.locate_rfi_events --scan --l1-threshold 40 "
            "--csv-out ../work/tables/T_rfi_events.csv` first."
        )
    if not track_p.exists():
        raise FileNotFoundError(
            f"{track_p} not found — run `python -m analysis.trajectory` first."
        )
    # The CSV is generated with a baseline L1 threshold of 40 so the
    # event catalogue captures weak-event provenance; for the paper
    # figure we restrict to the "strong" cruise-wide events at
    # L1_jam_max > 60, matching the threshold used in main.tex Table
    # tab:rfi-events and §5.7 narrative (nine of ten strong-RFI days).
    rfi_all = pl.read_csv(rfi_csv)
    rfi = rfi_all.filter(pl.col("L1_jam_max") > 60)
    track = read_parquet(track_p).filter(
        pl.col("lat").is_finite() & pl.col("lon").is_finite()
        & (pl.col("t_ns") > 0)
    )
    log.info(
        "Loaded %d/%d RFI events at L1_jam_max>60 and %d track rows",
        rfi.height, rfi_all.height, track.height,
    )

    fig = plt.figure(figsize=(10, 5.5))

    # ---- Map (try cartopy → plain) ----
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        ax_map = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=220))
        try:
            ax_map.add_feature(cfeature.LAND, facecolor="0.93", edgecolor="0.55", linewidth=0.3)
            ax_map.add_feature(cfeature.OCEAN, facecolor="#F4F8FB")
            ax_map.coastlines(linewidth=0.4, color="0.4")
        except Exception:  # noqa: BLE001
            pass
        ax_map.gridlines(linewidth=0.2, color="0.7")
        # Track drawn as a scatter dot cloud (not a polyline) — same
        # rationale as fig01: a polyline produces spurious straight
        # streaks every time the track crosses the projection's wrap
        # meridian or a data gap. ``rasterized=True`` collapses the
        # tens of thousands of dots into a single embedded raster image
        # inside the otherwise-vector PDF (drops file size from
        # ~130 MB to <2 MB); event markers, coastlines, gridlines and
        # text remain vector and crisp.
        ax_map.scatter(
            track["lon"].to_numpy(), track["lat"].to_numpy(),
            transform=ccrs.PlateCarree(), color="0.4", s=0.5, alpha=0.5,
            edgecolors="none", zorder=3, rasterized=True,
        )
        # Draw events
        sizes = np.clip(rfi["L1_jam_max"].cast(pl.Float64).to_numpy() ** 1.4, 30, 600)
        sc = ax_map.scatter(
            rfi["lon_med"].cast(pl.Float64).to_numpy(),
            rfi["lat_med"].cast(pl.Float64).to_numpy(),
            s=sizes,
            c=rfi["L1_jam_max"].cast(pl.Float64).to_numpy(),
            cmap="plasma",
            edgecolor="black",
            linewidth=0.5,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )
        ax_map.set_global()
        cb = fig.colorbar(sc, ax=ax_map, fraction=0.025, pad=0.02)
        cb.set_label("L1 jamInd peak")
    except Exception as exc:  # noqa: BLE001
        log.warning("cartopy unavailable (%s); plain map", exc)
        ax_map = fig.add_subplot(1, 1, 1)
        ax_map.scatter(track["lon"], track["lat"], color="0.4",
                       s=0.5, alpha=0.5, edgecolors="none", zorder=3,
                       rasterized=True)
        sizes = np.clip(rfi["L1_jam_max"].cast(pl.Float64).to_numpy() ** 1.4, 30, 600)
        sc = ax_map.scatter(
            rfi["lon_med"].cast(pl.Float64).to_numpy(),
            rfi["lat_med"].cast(pl.Float64).to_numpy(),
            s=sizes,
            c=rfi["L1_jam_max"].cast(pl.Float64).to_numpy(),
            cmap="plasma", edgecolor="black", linewidth=0.5, zorder=5,
        )
        ax_map.set_xlim(-180, 180)
        ax_map.set_ylim(-90, 90)
        ax_map.set_xlabel("Longitude (°E)")
        ax_map.set_ylabel("Latitude (°N)")
        ax_map.set_aspect("equal")
        cb = fig.colorbar(sc, ax=ax_map)
        cb.set_label("L1 jamInd peak")

    # Annotate clusters by majority-port label
    grouped: dict[str, dict] = {}
    for r in rfi.iter_rows(named=True):
        label = _label_for(float(r["lat_med"]), float(r["lon_med"]))
        if not label:
            continue
        g = grouped.setdefault(label, {"lat": [], "lon": [], "max": 0, "n": 0})
        g["lat"].append(float(r["lat_med"]))
        g["lon"].append(float(r["lon_med"]))
        g["max"] = max(g["max"], int(r["L1_jam_max"]))
        g["n"] += 1
    for label, g in grouped.items():
        try:
            ax_map.text(
                float(np.mean(g["lon"])), float(np.mean(g["lat"])) + 4.0,
                f"{label}\nn={g['n']}, peak={g['max']}",
                transform=(ccrs.PlateCarree() if "ccrs" in dir() else ax_map.transData),
                fontsize=7, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="0.3", alpha=0.85),
                zorder=10,
            )
        except Exception:  # noqa: BLE001
            ax_map.text(float(np.mean(g["lon"])), float(np.mean(g["lat"])) + 4.0,
                        f"{label}\nn={g['n']}, peak={g['max']}",
                        fontsize=7, ha="center",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor="0.3", alpha=0.85),
                        zorder=10)

    ax_map.set_title("Strong L1-band RFI events along the cruise  "
                     "(ZED-F9P-15B, full cruise; L1 jamInd peak per day)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)
    return out_path


@click.command()
@click.option("--out", default=None, type=click.Path())
def main(out: str | None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    out_path = Path(out) if out else figures_dir(cfg) / "output" / "fig13_rfi_events.pdf"
    render(out_path, cfg)


if __name__ == "__main__":
    main()
