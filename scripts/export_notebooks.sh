#!/usr/bin/env bash
# Export notebooks to self-contained, EXECUTED static HTML in exports/.
# Each export runs the notebook (bakes real plots/results), so training
# notebooks take minutes. Runs from notebooks/ so relative data paths resolve.
#
# On success, records the notebook's git blob hash in exports/.manifest, so the
# pre-push hook can tell "source changed since last export" without re-running
# (executed exports are not byte-deterministic, so output diffing would loop).
#
# Usage:
#   scripts/export_notebooks.sh                       # export all (minus skip)
#   scripts/export_notebooks.sh notebooks/Module_3_Practical_1_FCNN.py ...
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=exports
MANIFEST="$OUT/.manifest"
TIMEOUT="${EXPORT_TIMEOUT:-12000}"          # seconds per notebook
SKIP_FILE="scripts/export-skip.txt"

mkdir -p "$OUT"; touch "$MANIFEST"

# One export at a time: two runs share the log and the output files, and a run
# that fails deletes the HTML (see below), so it can remove another run's good export.
LOCK="$OUT/.export.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "Another export is running (lock: $LOCK, started by PID $(cat "$LOCK/pid" 2>/dev/null || echo '?'))."
  echo "If it is not, remove the lock with: rm -rf $LOCK"
  exit 1
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT

LOG="$OUT/_export.log"; : > "$LOG"

if   command -v gtimeout >/dev/null 2>&1; then TO=gtimeout
elif command -v timeout  >/dev/null 2>&1; then TO=timeout
else TO=""; fi

# skip-list (basenames), ignoring comments/blanks. Portable (no mapfile: macOS bash 3.2).
SKIPS=()
while IFS= read -r _line; do [ -n "$_line" ] && SKIPS+=("$_line"); done \
  < <(grep -vE '^[[:space:]]*#|^[[:space:]]*$' "$SKIP_FILE" 2>/dev/null || true)
is_skipped() { local b="$1" s; [ ${#SKIPS[@]} -eq 0 ] && return 1; for s in "${SKIPS[@]}"; do [ "$b" = "$s" ] && return 0; done; return 1; }

update_manifest() {  # base hash
  local tmp; tmp="$(mktemp)"
  grep -v "^$1 " "$MANIFEST" 2>/dev/null > "$tmp" || true
  echo "$1 $2" >> "$tmp"; sort "$tmp" -o "$tmp"; mv "$tmp" "$MANIFEST"
}

declare -a TARGETS=()
if [ "$#" -gt 0 ]; then
  for f in "$@"; do [[ "$f" == notebooks/*.py ]] && TARGETS+=("$(basename "$f")"); done
else
  for f in notebooks/*.py; do TARGETS+=("$(basename "$f")"); done
fi

declare -a OK=() FAIL=() SKIP=()
for py in "${TARGETS[@]}"; do
  base="${py%.py}"
  if is_skipped "$base"; then
    echo "SKIP  $base (skip-list)" | tee -a "$LOG"; SKIP+=("$base"); continue
  fi
  echo "=== exporting $base (timeout ${TIMEOUT}s) ===" | tee -a "$LOG"
  ( cd notebooks && ${TO:+$TO "$TIMEOUT"} uv run marimo export html "$py" -o "../$OUT/$base.html" --force >>"../$LOG" 2>&1 )
  rc=$?
  if [ $rc -eq 0 ] && [ -s "$OUT/$base.html" ]; then
    update_manifest "$base" "$(git hash-object "notebooks/$py")"
    echo "OK    $base" | tee -a "$LOG"; OK+=("$base")
  else
    echo "FAIL  $base (exit $rc)" | tee -a "$LOG"; FAIL+=("$base"); rm -f "$OUT/$base.html"
  fi
done

bash scripts/build_index.sh >/dev/null

echo "" | tee -a "$LOG"
echo "OK (${#OK[@]}): ${OK[*]:-}"       | tee -a "$LOG"
echo "SKIP (${#SKIP[@]}): ${SKIP[*]:-}" | tee -a "$LOG"
echo "FAIL (${#FAIL[@]}): ${FAIL[*]:-}" | tee -a "$LOG"
[ ${#FAIL[@]} -eq 0 ]
