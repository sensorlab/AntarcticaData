#!/usr/bin/env bash
# Add the receiver-antenna calibration to every IGS20 ANTEX file PRIDE PPP-AR
# can use for this cruise.
#
# pdp3 takes the ANTEX version named in each day's clock product
# ("SYS / PCVS APPLIED"). The WUM rapid clocks of 26 Sep 2025 - 29 Apr 2026
# name igs20_2353 to igs20_2415. When that version is missing from PRIDE's
# table folder, pdp3 downloads it from IGS, and the downloaded file has no
# entry for our antenna, so PRIDE applies a zero antenna model. This script
# downloads every IGS20 version issued in that window into the table folder
# and appends the TRSAX4E02 NONE entry (from the NGS file ngs20.atx) to each.
# It is safe to run again; files that already hold the entry are left alone.
#
# Usage (from code/, with pdp3 on PATH):
#     bash ppp/prepare_pride_tables.sh ../ngs20.atx
set -uo pipefail

NGS=${1:?usage: bash ppp/prepare_pride_tables.sh <ngs20.atx>}
ANT=TRSAX4E02
VERSIONS="2353 2354 2356 2360 2361 2370 2375 2385 2388 2401 2405 2408 2415 2417"
URL=https://files.igs.org/pub/station/general/pcv_archive

PDP3=$(command -v pdp3) || { echo "pdp3 is not on PATH" >&2; exit 1; }
CFG=$(dirname "$PDP3")/config_template
T=$(awk -F= '/^Table directory/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}' "$CFG")
T=${T/#\~/$HOME}
[ -d "$T" ] || { echo "PRIDE table folder not found: '$T' (Table directory in $CFG)" >&2; exit 1; }

ENTRY=$(mktemp)
trap 'rm -f "$ENTRY"' EXIT
awk -v ant="$ANT" '/START OF ANTENNA/{b=$0 ORS; f=1; next}
    f{b=b $0 ORS; if (/END OF ANTENNA/) {if (index(b, ant " ")) printf "%s", b; f=0}}' "$NGS" > "$ENTRY"
if [ "$(grep -c "TYPE / SERIAL NO" "$ENTRY")" -ne 1 ]; then
  echo "expected exactly one $ANT entry in $NGS" >&2; exit 1
fi

cd "$T" || exit 1
echo "PRIDE table folder: $T"
fail=0
for w in $VERSIONS; do
  f=igs20_$w.atx
  if ! grep -q "END OF HEADER" "$f" 2>/dev/null; then
    { curl -sfL "$URL/$f.gz" | gunzip -c > "$f"; } 2>/dev/null || curl -sfL "$URL/$f" -o "$f"
  fi
  if ! grep -q "END OF HEADER" "$f" 2>/dev/null; then
    echo "✗ $f could not be downloaded from $URL"; rm -f "$f"; fail=1; continue
  fi
  grep -q "^$ANT .*TYPE / SERIAL NO" "$f" || cat "$ENTRY" >> "$f"
done

for f in igs20*.atx; do
  n=$(grep -c "^$ANT .*TYPE / SERIAL NO" "$f")
  if [ "$n" -eq 1 ]; then echo "✓ $f"; else echo "✗ $f has $n $ANT entries"; fail=1; fi
done
exit $fail
