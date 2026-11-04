"""Check how PRIDE PPP-AR processed each day (ESSD revision).

For every day with PPP output (or one day with ``--day``) it reports, from the
kin file:

* the orbit and clock products and the ANTEX file named in the kin header;
* the GPS receiver-antenna height offsets PRIDE applied ("SITE ANTENNA PCO";
  about 54.7 mm on L1 and 57.9 mm on L5 for the calibrated TRSAX4E02, zero
  when PRIDE fell back to a zero antenna model);
* the mean number of satellites per system (PRIDE's ``Nsat/GREC2C3J``
  columns: all, GPS, GLONASS, Galileo, BDS-2, BDS-3, QZSS);
* the product warnings pdp3 wrote to ``run.log`` (none means the default WUM
  rapid products were used).

Output:
    tables/T_ppp_run_summary.csv   one row per day

With ``--require`` (used by run_strengthen.sh on its test day) the exit code
is 1 when no antenna offset was applied or the solution uses fewer than five
satellites per epoch on average (all systems together, as in the paper's
``nsat > 4`` epoch filter).

Usage:
    python -m verify.ppp_run_summary [--day YYYYMMDD] [--require]
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from analysis._common import list_days, load_config, resolve_path, tables_dir
from ppp.pride_runner import ANTENNA, _antex_has_antenna, antenna_applied, kin_header

SYSTEMS = ["all", "G", "R", "E", "C2", "C3", "J"]
MIN_MEAN_SATS = 5.0             # all systems together
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _kin_means(kin: Path) -> tuple[int, list[float], float]:
    """Epoch count, mean satellites per system and session length (h) of a PRIDE 3.x kin file."""
    sums = np.zeros(len(SYSTEMS))
    n = 0
    t0 = t1 = None
    in_data = False
    with kin.open(errors="replace") as fh:
        for raw in fh:
            if not in_data:
                in_data = "END OF HEADER" in raw
                continue
            parts = raw.split()
            if len(parts) not in (16, 17):   # 17 when the solution-flag column is present
                continue
            off = len(parts) - 16
            try:
                t = float(parts[0]) * 86400.0 + float(parts[1])
                sums += [float(v) for v in parts[8 + off:15 + off]]
            except ValueError:
                continue
            t0 = t if t0 is None else min(t0, t)
            t1 = t if t1 is None else max(t1, t)
            n += 1
    hours = (t1 - t0) / 3600.0 if n else 0.0
    return n, [float(s / n) if n else float("nan") for s in sums], hours


def _product_warnings(run_log: Path) -> str:
    if not run_log.exists():
        return ""
    msgs = []
    for line in ANSI.sub("", run_log.read_text(errors="replace")).splitlines():
        if "warning:" in line and "PrepareProducts" in line:
            msg = line.split("PrepareProducts:", 1)[1].strip()
            if msg not in msgs:
                msgs.append(msg)
    return " | ".join(msgs)


def summarise_day(day: str, cfg: dict) -> dict | None:
    out = resolve_path(cfg["paths"]["ppp"]) / day
    kins = sorted(k for k in out.glob("[0-9][0-9][0-9][0-9]/*/kin_*") if k.is_file())
    if not kins:
        return None
    kin = kins[0]
    n, means, hours = _kin_means(kin)
    hdr = kin_header(kin)
    applied = antenna_applied(kin)
    if applied is None:     # header without offsets: check the ANTEX file PRIDE linked
        atx = kin.parent / "abs_igs.atx"
        applied = _antex_has_antenna(atx) if atx.exists() else None
    row = {"day": day, "kin": kin.name, "n_epochs": n, "session_h": round(hours, 2)}
    row.update({f"mean_sats_{s}": round(m, 2) for s, m in zip(SYSTEMS, means)})
    log_text = ANSI.sub("", (out / "run.log").read_text(errors="replace")) if (out / "run.log").exists() else ""
    row["finished"] = bool(re.search(r"ProcessSingleSite .* done", log_text))
    row["setup"] = "2026-10" if (out / "pdp3_command.txt").exists() else "first run"
    row.update({
        "orbit": hdr["orbit"], "clock": hdr["clock"], "antex": hdr["antex"],
        "pco_up_gps_mm": " ".join(f"{k}={v:.2f}" for k, v in hdr["pco_gps"].items()),
        "antenna_applied": applied,
        "product_warnings": _product_warnings(out / "run.log"),
    })
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--day")
    ap.add_argument("--require", action="store_true")
    args = ap.parse_args()
    cfg = load_config()

    days = [args.day] if args.day else list_days(cfg)
    rows = [r for d in days if (r := summarise_day(d, cfg)) is not None]
    if not rows:
        print(f"no PRIDE kin output for {args.day or 'any day'}")
        return 1
    df = pl.DataFrame(rows, infer_schema_length=None)

    if not args.day:
        df.write_csv(tables_dir(cfg) / "T_ppp_run_summary.csv")
    with pl.Config(tbl_rows=10, tbl_cols=16, float_precision=1, tbl_width_chars=220, fmt_str_lengths=40):
        print(df.select("day", "n_epochs", "session_h", "mean_sats_G", "mean_sats_R", "mean_sats_E", "mean_sats_C3",
                        "mean_sats_J", "orbit", "pco_up_gps_mm", "antenna_applied"))

    print(f"\ndays with PPP: {df.height}; sessions shorter than 23 h: "
          f"{', '.join(df.filter(pl.col('session_h') < 23)['day']) or 'none'}")
    print("mean satellites per epoch (median over days): "
          + "  ".join(f"{s} {df[f'mean_sats_{s}'].median():.1f}" for s in SYSTEMS))
    print("orbit products:", dict(sorted(Counter(o[:10] for o in df["orbit"].to_list()).items())))
    print("ANTEX files:", dict(sorted(Counter(df["antex"].to_list()).items())))
    print("GPS antenna offsets applied (mm):", dict(sorted(Counter(df["pco_up_gps_mm"].to_list()).items())))
    no_ant = df.filter(~pl.col("antenna_applied").fill_null(False))
    print(f"days without {ANTENNA} antenna offsets: {no_ant.height}"
          + (f" -> {', '.join(no_ant['day'])}" if no_ant.height else ""))
    unfinished = df.filter(~pl.col("finished"))
    print(f"days whose run.log does not report the session as finished: {unfinished.height}"
          + (f" -> {', '.join(unfinished['day'])}" if unfinished.height else ""))
    print("days per processing run:", dict(sorted(Counter(df["setup"].to_list()).items())))
    warn = df.filter(pl.col("product_warnings") != "")
    print(f"days with product warnings (not plain WUM rapid): {warn.height}")
    for day, msg in warn.select("day", "product_warnings").iter_rows():
        print(f"  {day}: {msg[:200]}")

    if args.require:
        r = df.row(0, named=True)
        problems = []
        if not r["mean_sats_all"] >= MIN_MEAN_SATS:
            problems.append(f"only {r['mean_sats_all']:.1f} satellites per epoch on average")
        if r["antenna_applied"] is not True:
            problems.append(f"no {ANTENNA} offsets applied (GPS: {r['pco_up_gps_mm'] or 'none in header'})")
        if r["n_epochs"] < 1000:
            problems.append(f"only {r['n_epochs']} epochs in the kin file")
        for p in problems:
            print("✗", p)
        if problems:
            return 1
        print(f"✓ {r['day']}: antenna calibration applied; {r['mean_sats_all']:.1f} satellites per epoch "
              f"(GPS {r['mean_sats_G']:.1f}, Galileo {r['mean_sats_E']:.1f}, BeiDou {r['mean_sats_C2'] + r['mean_sats_C3']:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
