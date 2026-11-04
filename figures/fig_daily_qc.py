"""Daily quality-control time series (Männel et al. 2021, Fig. 3 equivalent).

Two-panel stacked figure showing for every cruise day:

1. Carrier-phase validity losses per 1000 carrier-phase observations (losses of
   the carrier-phase-valid flag, a tracking-quality proxy for cycle slips),
2. Code-multipath RMS (M1 and M2, in metres).

Both are read from ``tables/T4_daily_stats.parquet`` (produced by
``analysis.qc_summary``). The NAV-PVT epoch count is approximately
constant across the cruise (the receiver held a continuous 2 Hz
recording) and is not plotted; the per-day epoch totals are tabulated
in ``T4_daily_stats``. The figure is sized for one ESSD column-pair
(14 cm wide) and is the natural companion to the Bosser 2021 Table 3
position-percentile table.

Usage:
    python -m figures.fig_daily_qc
Output:
    figures/output/fig_daily_qc.pdf
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from analysis._common import load_config, read_parquet, tables_dir
from figures._helpers import apply_style

log = logging.getLogger(__name__)


def _day_to_date(s: str) -> np.datetime64:
    return np.datetime64(f"{s[0:4]}-{s[4:6]}-{s[6:8]}")


def load_t4(cfg: dict | None = None) -> pl.DataFrame:
    cfg = cfg or load_config()
    p = tables_dir(cfg) / "T4_daily_stats.parquet"
    return read_parquet(p).sort("day")


def build_figure(t4: pl.DataFrame, out_path: Path) -> None:
    apply_style()
    dates = np.array([_day_to_date(d) for d in t4["day"].to_list()])
    fig, axes = plt.subplots(
        2, 1, figsize=(7, 4.6), sharex=True, constrained_layout=True
    )

    # Panel 1 — carrier-phase validity losses
    ax = axes[0]
    if "slip_ratio_per_1000" in t4.columns:
        y = t4["slip_ratio_per_1000"].to_numpy()
        ax.plot(dates, y, "-", lw=0.9, color="#d62728")
        ax.set_ylabel("Phase-validity losses\nper 1000 obs")
    else:
        y = t4["cycle_slip_count"].to_numpy()
        ax.plot(dates, y, "-", lw=0.9, color="#d62728")
        ax.set_ylabel("Phase-validity losses\nper day")
    ax.grid(alpha=0.3)

    # Panel 2 — multipath RMS (bottom panel: carries x-axis labels)
    ax = axes[1]
    if "mp1_rms_m" in t4.columns and "mp2_rms_m" in t4.columns:
        ax.plot(dates, t4["mp1_rms_m"].to_numpy(), "-", lw=0.9,
                color="#2ca02c", label="M1 (L1)")
        ax.plot(dates, t4["mp2_rms_m"].to_numpy(), "-", lw=0.9,
                color="#9467bd", label="M2 (L5/E5a/B2a)")
        ax.legend(loc="upper right", fontsize=7)
    ax.set_ylabel("Multipath\nRMS (m)")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlabel("UTC date")

    fig.suptitle(
        "Daily quality indicators for the 216-day R/V Laura Bassi cruise",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=200)
    log.info("Wrote %s", out_path)


def main_entry(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    t4 = load_t4(cfg)
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig_daily_qc.pdf"
    build_figure(t4, out_path)
    return out_path


@click.command()
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    main_entry()


if __name__ == "__main__":
    main()
