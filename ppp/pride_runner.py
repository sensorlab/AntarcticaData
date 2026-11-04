"""Wrap PRIDE PPP-AR (`pdp3`) for kinematic per-day processing.

Inputs (per day):
    rinex_obs   <paths.rinex>/<day>/obs.rnx
    products    downloaded by pdp3 (WUM rapid by default)

Outputs (pdp3 work folder, one per session):
    <paths.ppp>/<day>/<yyyy>/<doy>/kin_<yyyy><doy>_jsi1   (or <doy1>-<doy2>/)
    <paths.ppp>/<day>/run.log

Usage:
    python -m ppp.pride_runner --day 20250930

PPP runs in *kinematic* mode with troposphere ZTD as a random-walk
parameter (default σ = 5 mm/√h). Receiver position σ at each epoch is
``inf`` (fully kinematic). PPP-AR is enabled when integer ambiguity
resolution succeeds (typical for IGS final products with multi-GNSS).

The frequencies are passed explicitly. PRIDE's default (G12 R12 E15 C26
J12) needs GPS L2 and BeiDou B3I, which the L1/L5 receiver does not track,
and would leave Galileo as the only usable system.

The session start is passed explicitly as well (``-s``, the first epoch of
the file). Without it, pdp3 decimating 2 Hz data to 1 s looks for the first
epoch that lies within 1 ms after a whole second and starts the session
there. The F9P epochs usually sit 9-20 ms before the second, so pdp3 checks
every epoch (tens of minutes) and falls back to the first one. On days when
the receiver clock moves onto the second late in the file, it starts the
session minutes before the end and the day fails or is truncated.

The time window is widened from 0.01 s to 0.05 s (``-twnd``). PRIDE uses an
epoch only if it lies within the window of the 1 s processing grid, and the
receiver epochs drift by about 11 ms per day (and occasionally jump), so with
the default window the last hours of some days are dropped. The window only
selects epochs: PRIDE computes the geometry from each epoch's own time tag.
With 2 Hz data (epochs 0.5 s apart) a 0.05 s window cannot pick a wrong epoch.

pdp3 links the IGS ANTEX version named in the day's clock product as
``abs_igs.atx`` and downloads it when that version is not in PRIDE's table
folder. A downloaded file has no entry for the receiver antenna, and PRIDE
3.2.8 then applies a zero antenna model without any message. After each run
we therefore read the receiver-antenna offsets that PRIDE wrote into the kin
header ("SITE ANTENNA PCO"). If they are zero, the day fails and its kin
file is renamed so that it is neither used nor counted as done (fix with
``ppp/prepare_pride_tables.sh`` and rerun).

This module shells out to ``pdp3`` (the PRIDE PPP-AR CLI). It does not
embed PPP logic — PRIDE is a 70k-line Fortran package; building our own
would be misguided.
"""

from __future__ import annotations

import logging
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import click

from analysis._common import load_config, resolve_path

log = logging.getLogger(__name__)

# Ionosphere-free pairs available from the ZED-F9P-15B (L1/L5 firmware):
# GPS L1/L5, Galileo E1/E5a, BeiDou B1I/B2a, QZSS L1/L5.
FREQUENCIES = ["G15", "E15", "C25", "J15"]
ANTENNA = "TRSAX4E02"     # receiver antenna type in the RINEX header


def _have_pdp3() -> bool:
    return shutil.which("pdp3") is not None


def _first_epoch(obs: Path) -> list[str]:
    """pdp3 ``-s`` arguments for the first epoch of a RINEX 3 file (seconds rounded down to 0.01 s)."""
    with obs.open(errors="replace") as fh:
        for line in fh:
            if "END OF HEADER" in line:
                break
        for line in fh:
            if line.startswith("> "):
                y, mo, d, h, mi = (int(v) for v in line[2:].split()[:5])
                sec = math.floor(float(line[2:].split()[5]) * 100) / 100
                return [f"{y:04d}/{mo:02d}/{d:02d}", f"{h:02d}:{mi:02d}:{sec:05.2f}"]
    raise ValueError(f"no observation epoch in {obs}")


def _antex_has_antenna(atx: Path) -> bool:
    with atx.open(errors="replace") as fh:
        return any(line.startswith(ANTENNA + " ") and "TYPE / SERIAL NO" in line for line in fh)


def kin_header(kin: Path) -> dict[str, Any]:
    """Products, ANTEX file and GPS antenna offsets from a PRIDE 3.x kin header.

    ``pco_gps`` maps the GPS frequency name to the receiver-antenna height
    offset (mm) that PRIDE applied; zero means no calibration was found.
    """
    out: dict[str, Any] = {"orbit": "", "clock": "", "antex": "", "pco_gps": {}}
    labels = {"SAT ORBIT": "orbit", "SAT CLOCK": "clock", "TABLE ANTEX": "antex"}
    with kin.open(errors="replace") as fh:
        for line in fh:
            if "END OF HEADER" in line:
                break
            text = line.rstrip()
            for label, key in labels.items():
                if text.endswith(label):
                    out[key] = Path(text[: -len(label)].strip()).name
            if "SITE ANTENNA PCO" in text and text.startswith("G"):
                f = text.split("SITE ANTENNA PCO")[0].split()
                try:
                    out["pco_gps"][f[1]] = float(f[5])
                except (IndexError, ValueError):
                    pass
    return out


def antenna_applied(kin: Path) -> bool | None:
    """True if PRIDE applied a non-zero GPS antenna offset; None if the header has none."""
    pco = kin_header(kin)["pco_gps"]
    if not pco:
        return None
    return any(abs(h) > 1.0 for h in pco.values())


def _check_antenna(day: str, out_dir: Path) -> None:
    """Fail the day if PRIDE ran without the receiver-antenna calibration."""
    kins = sorted(k for k in out_dir.glob("[0-9][0-9][0-9][0-9]/*/kin_*") if k.is_file())
    if not kins:
        raise RuntimeError(f"{day}: pdp3 finished but wrote no kin file under {out_dir}")
    for kin in kins:
        ok = antenna_applied(kin)
        if ok is None:      # header without offsets: check the ANTEX file PRIDE linked
            atx = kin.parent / "abs_igs.atx"
            ok = atx.exists() and _antex_has_antenna(atx)
        if ok:
            continue
        for k in kins:
            k.rename(k.with_name("NOANT_" + k.name))
        msg = (f"{day}: PRIDE applied a zero antenna model ({ANTENNA} is not in "
               f"{(kin.parent / 'abs_igs.atx').resolve()}). Run ppp/prepare_pride_tables.sh "
               "and process the day again.")
        with (out_dir / "run.log").open("a") as fh:
            fh.write(f"error: {msg}\n")
        raise RuntimeError(msg)


def run_one_day(day: str, cfg: dict | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    if not _have_pdp3():
        raise RuntimeError(
            "pdp3 (PRIDE PPP-AR) not found in PATH. "
            "Install from https://github.com/PrideLab/PRIDE-PPPAR"
        )
    rinex_dir = resolve_path(cfg["paths"]["rinex"]) / day
    obs = rinex_dir / "obs.rnx"
    if not obs.exists():
        raise FileNotFoundError(
            f"Missing RINEX OBS for {day}: {obs}. Run Stage 5 (rinex) first."
        )
    out_dir = resolve_path(cfg["paths"]["ppp"]) / day
    out_dir.mkdir(parents=True, exist_ok=True)
    # PRIDE PPP-AR 3.x CLI. No products-dir flag — pdp3 downloads the
    # products into its work folder unless a -cfg file overrides.
    # Inter-system biases on by default for selected systems via -isb.
    # AR is on by default; -f disables it.
    cmd = [
        "pdp3",
        "-m", "K",            # kinematic
        "-sys", "GREC",       # GPS, GLONASS, Galileo, BeiDou
        "-isb", "REC",        # estimate ISB for non-GPS systems
        "-i", "1",            # 1 Hz output
        "-frq", *FREQUENCIES,
        "-s", *_first_epoch(obs),   # start at the first epoch (see module docstring)
        "-twnd", "0.05",            # epoch-matching window (see module docstring)
        str(obs),
    ]
    log.info("$ %s", " ".join(cmd))
    (out_dir / "pdp3_command.txt").write_text(" ".join(cmd) + "\n")
    log_file = out_dir / "run.log"
    with log_file.open("w") as fh:
        rc = subprocess.call(cmd, cwd=out_dir, stdout=fh, stderr=fh)
    if rc != 0:
        raise RuntimeError(f"pdp3 failed for {day} (rc={rc}). See {log_file}")
    # pdp3 can exit 0 after a failed session; intermediate kin files may exist then
    if not re.search(r"ProcessSingleSite .* done", log_file.read_text(errors="replace")):
        for kin in out_dir.glob("[0-9][0-9][0-9][0-9]/*/kin_*"):
            kin.rename(kin.with_name("INCOMPLETE_" + kin.name))
        raise RuntimeError(f"pdp3 did not finish the session for {day}. See {log_file}")
    _check_antenna(day, out_dir)
    return {
        "day": day,
        "kin": sorted(out_dir.glob("[0-9][0-9][0-9][0-9]/*/kin_*")),
        "log": log_file,
    }


@click.command()
@click.option("--day", required=True)
def main(day: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    res = run_one_day(day)
    for k, v in res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
