#!/usr/bin/env bash
# ESSD revision (Oct 2026): regenerate the cruise-wide tables, statistics and
# figures of the paper from the full 216-day staging.
#
# Run from code/ with the project environment active:
#     nohup bash run_revision.sh > ../work/logs/revision_$(date +%Y%m%d).log 2>&1 &
#
# Assumes work/staging/<day>/ already exists for all days (no re-parse) and
# that work/derived/track/track.all.parquet is current. A failing step does
# not stop the others; the failures are listed at the end.
set -uo pipefail

J=${J:-16}                      # parallel days (16 leaves room for a PPP run)
PY=${PY:-python}
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p ../work/logs
FAILED=()
step() {  # run one command, record it if it fails
  "$@" || { echo "   FAILED: $*"; FAILED+=("$*"); }
}
DAYS=$($PY -c "from analysis._common import list_days; print('\n'.join(list_days()))")
N=$(echo "$DAYS" | wc -l)
echo "== $N staged days, $J parallel ($(date +%H:%M))"

echo "== 1/9 multipath (new arc segmentation + azimuth column)"
echo "$DAYS" | xargs -P "$J" -I{} sh -c '"$0" -m analysis.multipath --day {} || echo "FAILED-DAY {}"' "$PY" \
  > ../work/logs/rev_multipath.log 2>&1
grep "FAILED-DAY" ../work/logs/rev_multipath.log | sed 's/^/   /'

echo "== 2/9 daily QC + T4 (multipath RMS feeds T4)"
echo "$DAYS" | xargs -P "$J" -I{} sh -c '"$0" -m analysis.qc_summary --day {} --no-aggregate || echo "FAILED-DAY {}"' "$PY" \
  > ../work/logs/rev_qc.log 2>&1
grep "FAILED-DAY" ../work/logs/rev_qc.log | sed 's/^/   /'
step $PY -c "from analysis.qc_summary import aggregate_t4; from analysis._common import load_config; aggregate_t4(load_config())"

echo "== 3/9 multipath figures and regime summary ($(date +%H:%M))"
step $PY -m figures.make_figures --only fig05
step $PY -m figures.fig_daily_qc
step $PY -m figures.fig_multipath_azimuth
step $PY -m verify.multipath_summary

echo "== 4/9 file inventory and gaps"
step $PY -m verify.file_inventory

echo "== 5/9 framing integrity ($(date +%H:%M))"
step $PY -m verify.framing_integrity --workers "$J"

echo "== 6/9 per-observable accuracy"
step $PY -m verify.obs_accuracy

echo "== 7/9 daily inference (block bootstrap, N_eff)"
step $PY -m analysis.jamming_confound
step $PY -m verify.daily_inference
step $PY -m verify.ppp_vs_gim          # needs tables/T_ppp_residuals.csv from ppp.compare

echo "== 8/9 OGS cross-check incl. campaign 9 (return leg)"
step $PY -m verify.ogs_cross_validation --refetch

echo "== 9/9 OGS ship meteorology (WXT536)"
step $PY -m verify.fetch_ogs_met

echo "== done ($(date +%H:%M)); failed steps: ${#FAILED[@]}"
for f in "${FAILED[@]}"; do echo "   $f"; done
echo "   Outputs: work/tables/T_{file_inventory,gaps,framing_integrity,obs_accuracy,nav_accuracy,"
echo "   multipath_daily,multipath_azimuth,daily_inference,daily_series,ogs_vs_f9p_summary,ogs_vs_f9p_daily,"
echo "   jam_confound,ppp_vs_gim_daily,ppp_vs_gim_tests}.csv, T4_daily_stats.csv, T_file_inventory.tex and"
echo "   work/figures/output/{fig05_*,fig_daily_qc,fig_multipath_azimuth,fig_ogs_vs_f9p}.pdf"
