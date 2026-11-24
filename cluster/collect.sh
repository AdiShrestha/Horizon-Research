#!/usr/bin/env bash
# Pack cluster results for transfer back to the laptop (A18).
# Usage: bash cluster/collect.sh WORK_DIR [DEST_DIR]
# Writes results-core-<stamp>.tar (emulators, statistics without fields, checks, logs, status, graph)
# and results-fields-<stamp>.tar (seed-0 error and W3 fields), each with a SHA-256 manifest inside.
set -euo pipefail
WORK="$(cd "$1" && pwd)"; DEST="${2:-$WORK/..}"; STAMP="$(date +%Y%m%d-%H%M%S)"
cd "$WORK"
find emulators stats checks logs status graph.json slurm_ids.json -type f 2>/dev/null \
  | grep -v '/fields/' | sort > /tmp/hz_core_$$.txt
find stats -path '*/fields/*' -type f 2>/dev/null | sort > /tmp/hz_fields_$$.txt
xargs -a /tmp/hz_core_$$.txt sha256sum > RESULTS_MANIFEST_core.sha256
echo RESULTS_MANIFEST_core.sha256 >> /tmp/hz_core_$$.txt
tar -cf "$DEST/results-core-$STAMP.tar" -T /tmp/hz_core_$$.txt
if [ -s /tmp/hz_fields_$$.txt ]; then
  xargs -a /tmp/hz_fields_$$.txt sha256sum > RESULTS_MANIFEST_fields.sha256
  echo RESULTS_MANIFEST_fields.sha256 >> /tmp/hz_fields_$$.txt
  tar -cf "$DEST/results-fields-$STAMP.tar" -T /tmp/hz_fields_$$.txt
fi
rm -f /tmp/hz_core_$$.txt /tmp/hz_fields_$$.txt
(cd "$DEST" && sha256sum results-*-"$STAMP".tar | tee "results-$STAMP.sha256")
