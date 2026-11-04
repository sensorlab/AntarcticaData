#!/usr/bin/env bash
# ESSD revision, second batch: kinematic PPP for every day that lacks a
# verified solution, the real-time vs PPP comparison, and the PPP
# within-berth repeatability. Independent of run_revision.sh; both can run at
# the same time.
#
# Once, before the first run (README, "External tools"):
#     bash ppp/prepare_pride_tables.sh ../ngs20.atx
#
# Run from code/ with the project environment active:
#     nohup bash run_strengthen.sh > ../work/logs/strengthen_$(date +%Y%m%d).log 2>&1 &
#
# Earlier PPP days are kept when their kin header shows that PRIDE applied the
# antenna calibration (non-zero GPS antenna offsets). Days of the first run
# (no pdp3_command.txt) often stop before the end of the file, because PRIDE's
# default 0.01 s epoch window loses the drifting receiver epochs; with
# REDO_SHORT=1 (default) those with a session under 23 h are processed again
# (about 3 h of computing per day). REDO_SHORT=0 keeps them, and moves back
# such days that an earlier run set aside. FULL=1 moves every earlier day
# aside. Running the script again continues where it stopped.
set -uo pipefail

J=${J:-8}                       # parallel days (pdp3 uses ~2 threads each)
PY=${PY:-python}
SMOKE_DAY=${SMOKE_DAY:-20260410}
FULL=${FULL:-0}
REDO_SHORT=${REDO_SHORT:-1}
NONCONV="20260101 20260302 20260303 20260306 20260328"   # failed in the first run; not used as test day
PARTIAL="20250926 20260429"                              # first and last file are shorter than a day
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p ../work/logs
die() { echo "ABORT: $*"; exit 1; }
has_ppp() { compgen -G "../work/ppp/$1/[0-9][0-9][0-9][0-9]/*/kin_*" >/dev/null; }

echo "== 0/8 pre-flight"
command -v pdp3 >/dev/null || die "pdp3 is not on PATH: export PATH=\$HOME/.PRIDE_PPPAR_BIN:\$PATH"
command -v bc >/dev/null || die "bc is not on PATH (pdp3 needs it)"
PB=$(dirname "$(command -v pdp3)")
if ldd "$PB/lsq" 2>/dev/null | grep -q "not found"; then
  die "PRIDE programs cannot find: $(ldd "$PB/lsq" | awk '/not found/{print $1}' | xargs). export LD_LIBRARY_PATH=\$HOME/.local/fortran/lib:\${LD_LIBRARY_PATH:-}"
fi
T=$(awk -F= '/^Table directory/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}' "$PB/config_template")
T=${T/#\~/$HOME}
for w in 2353 2385 2388 2401 2405 2408 2415; do
  grep -q "^TRSAX4E02 .*TYPE / SERIAL NO" "$T/igs20_$w.atx" 2>/dev/null \
    || die "no TRSAX4E02 entry in $T/igs20_$w.atx; run: bash ppp/prepare_pride_tables.sh ../ngs20.atx"
done
grep -q '"-frq"' ppp/pride_runner.py || die "ppp/pride_runner.py is an old version (no -frq); update the repository"
$PY -m py_compile ppp/pride_runner.py verify/ppp_run_summary.py verify/ppp_berth_repeatability.py \
  verify/export_ppp_tracks.py || die "python files do not compile"
echo "   $(pdp3 -V 2>&1 | grep -o 'version.*'); tables in $T; parallel jobs: $J"

echo "== 1/8 keep earlier PPP days that used the antenna calibration"
pco_ok() {  # kin file: success if PRIDE applied a non-zero GPS receiver-antenna offset
  awk '/SITE ANTENNA PCO/ && /^G/ { if ($6 > 1 || $6 < -1) ok = 1 } /END OF HEADER/ { exit } END { exit !ok }' "$1"
}
kin_hours() {  # kin file: hours between the first and the last epoch
  awk '/END OF HEADER/ { d = 1; next } d && (NF == 16 || NF == 17) { t = $1 * 86400 + $2; if (!n++) a = t; b = t }
       END { printf "%.1f", n ? (b - a) / 3600 : 0 }' "$1"
}
STAMP=$(date +%Y%m%d)
A=../work/ppp_set_aside_$STAMP
AD=../work/derived/ppp_set_aside_$STAMP
set_aside() {  # move a folder into the set-aside folder under a unique name
  local t="$2/$(basename "$1")" n=1
  while [ -e "$t" ]; do t="$2/$(basename "$1").$n"; n=$((n+1)); done
  mkdir -p "$2" && mv "$1" "$t"
}
done_ok() {  # day folder of this pipeline: pdp3 reported the session as finished
  [ ! -f "$1/pdp3_command.txt" ] || sed 's/\x1b\[[0-9;]*m//g' "$1/run.log" 2>/dev/null | grep -q "ProcessSingleSite .* done"
}
shopt -s nullglob
if [ "$REDO_SHORT" != 1 ] && [ "$FULL" != 1 ]; then
  # bring back first-run days that an earlier run set aside only for a short session
  back=0
  for dd in ../work/ppp_set_aside_*/2*/; do
    d=$(basename "$dd")
    [[ "$d" =~ ^[0-9]{8}$ ]] && [ ! -f "$dd/pdp3_command.txt" ] || continue
    kins=("$dd"[0-9][0-9][0-9][0-9]/*/kin_*)
    [ ${#kins[@]} -gt 0 ] || continue
    ok=1; for k in "${kins[@]}"; do pco_ok "$k" || ok=0; done
    [ $ok = 1 ] || continue
    if [ -d "../work/ppp/$d" ]; then
      compgen -G "../work/ppp/$d/[0-9][0-9][0-9][0-9]/*/kin_*" >/dev/null && done_ok "../work/ppp/$d" && continue
      set_aside "../work/ppp/$d" "$A"            # unfinished new run
    fi
    mv "${dd%/}" "../work/ppp/$d" && back=$((back+1))
  done
  [ $back -gt 0 ] && echo "   moved back $back first-run days (REDO_SHORT=0)"
fi
kept=0; reasons=""
for dd in ../work/ppp/2*/; do
  d=$(basename "$dd")
  [[ "$d" =~ ^[0-9]{8}$ ]] || continue
  kins=("$dd"[0-9][0-9][0-9][0-9]/*/kin_*)
  reason=""
  if [ "$FULL" = 1 ]; then reason="FULL=1"
  elif [ ${#kins[@]} -eq 0 ]; then reason="no kin file"
  elif ! done_ok "${dd%/}"; then reason="unfinished run"
  else
    for k in "${kins[@]}"; do pco_ok "$k" || { reason="zero antenna offsets"; break; }; done
    if [ -z "$reason" ] && [ "$REDO_SHORT" = 1 ] && [ ! -f "$dd/pdp3_command.txt" ] && [[ " $PARTIAL " != *" $d "* ]]; then
      h=$(kin_hours "${kins[0]}")
      awk -v h="$h" 'BEGIN { exit !(h < 23) }' && reason="session shorter than 23 h"
    fi
  fi
  if [ -z "$reason" ]; then kept=$((kept+1)); continue; fi
  set_aside "${dd%/}" "$A"
  reasons+="$reason"$'\n'
  [ "$reason" = "session shorter than 23 h" ] && echo "   $d: session ${h} h"
done
for dd in ../work/derived/ppp/2*/; do   # comparison output of days that will be redone
  [ -d "../work/ppp/$(basename "$dd")" ] || set_aside "${dd%/}" "$AD"
done
shopt -u nullglob
echo "   kept $kept earlier days"
[ -n "$reasons" ] && printf "%s" "$reasons" | sort | uniq -c | sed 's/^/   set aside: /'

echo "== 2/8 RINEX for the days that need PPP and have none"
need=()
for f in ../antarctica2026/*.ubx; do
  d=$(basename "$f" .ubx)
  has_ppp "$d" || [ -s "../work/derived/rinex/$d/obs.rnx" ] || need+=("$d")
done
echo "   ${#need[@]} days to convert"
if [ ${#need[@]} -gt 0 ]; then
  command -v convbin >/dev/null || die "convbin is not on PATH (README, External tools)"
  U=$($PY -c "from analysis._common import load_config, resolve_path; print(resolve_path(load_config()['paths']['ubx_dir']))")
  ls "$U"/*.ubx >/dev/null 2>&1 || die "paths.ubx_dir in config/pipeline.yaml ($U) has no .ubx files; convbin_runner reads from there"
  printf '%s\n' "${need[@]}" | xargs -P "$J" -I{} sh -c \
    '"$0" -m rinex.convbin_runner --day {} > /dev/null 2>&1 && echo "   OK-RINEX   {}" || echo "   FAIL-RINEX {}"' "$PY"
fi

echo "== 3/8 test the setup on one day that still needs PPP"
if has_ppp "$SMOKE_DAY" && [ -f "../work/ppp/$SMOKE_DAY/pdp3_command.txt" ]; then
  echo "   $SMOKE_DAY was already processed with this setup"
  $PY -m verify.ppp_run_summary --day "$SMOKE_DAY" --require || die "the test day $SMOKE_DAY failed the checks above"
  SMOKE_DAY=""; TESTED=1
fi
if [ -z "${TESTED:-}" ] && { [ ! -s "../work/derived/rinex/$SMOKE_DAY/obs.rnx" ] || has_ppp "$SMOKE_DAY"; }; then
  SMOKE_DAY=""
  for f in ../work/derived/rinex/*/obs.rnx; do
    d=$(basename "$(dirname "$f")")
    [[ " $NONCONV " == *" $d "* ]] || has_ppp "$d" || { SMOKE_DAY=$d; break; }
  done
fi
if [ -n "$SMOKE_DAY" ]; then
  echo "   PPP for $SMOKE_DAY ($(date +%H:%M))"
  $PY -m ppp.pride_runner --day "$SMOKE_DAY" > "../work/logs/ppp_test_$SMOKE_DAY.log" 2>&1 || {
    tail -5 "../work/logs/ppp_test_$SMOKE_DAY.log"; tail -25 "../work/ppp/$SMOKE_DAY/run.log"
    die "PPP failed for the test day $SMOKE_DAY"; }
  echo "   finished $(date +%H:%M)"
  $PY -m verify.ppp_run_summary --day "$SMOKE_DAY" --require || die "the test day $SMOKE_DAY failed the checks above"
elif [ -z "${TESTED:-}" ]; then
  echo "   every day with RINEX already has PPP output"
fi

echo "== 4/8 PPP for all days ($(date +%H:%M); finished days are skipped)"
bash run_ppp.sh --ppp-only --jobs "$J" --log ../work/logs/ppp_strengthen.log | grep -v SKIP-PPP
grep -E "OK-PPP|FAIL-PPP|NO-RINEX" ../work/logs/ppp_strengthen.log | awk '{print $2}' | sort | uniq -c
for d in $(awk '/FAIL-PPP|NO-RINEX/{print $3}' ../work/logs/ppp_strengthen.log); do
  echo "   $d: $(sed 's/\x1b\[[0-9;]*m//g' "../work/ppp/$d/run.log" 2>/dev/null | grep -iE 'error|fail' | tail -1)"
done

echo "== 5/8 per-day check: systems used, antenna file, products"
$PY -m verify.ppp_run_summary

echo "== 6/8 real-time vs PPP comparison for all days, percentile table"
$PY -m ppp.compare
$PY -m ppp.percentiles

echo "== 7/8 PPP within-berth repeatability"
$PY -m verify.ppp_berth_repeatability

echo "== 8/8 export the PPP trajectory (optional deposit file)"
$PY -m verify.export_ppp_tracks

echo "== done. Outputs: work/tables/T_pos_percentiles.{csv,tex}, T_ppp_residuals.csv,"
echo "   T_ppp_berth_repeatability.csv, T_ppp_berth_windows.csv and T_ppp_run_summary.csv"
