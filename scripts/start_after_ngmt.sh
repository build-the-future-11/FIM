#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p results/current_component_ablations
exec > >(tee -a results/current_component_ablations/finalization.log) 2>&1

UPSTREAM_SESSION="ngmt-conference-full"
echo "WAITING for upstream tmux session: $UPSTREAM_SESSION"
while true; do
  if ! tmux has-session -t "$UPSTREAM_SESSION" 2>/dev/null; then
    echo "UPSTREAM SESSION DISAPPEARED WITHOUT RETAINED EXIT STATUS"
    exit 3
  fi
  dead="$(tmux display-message -p -t "$UPSTREAM_SESSION" '#{pane_dead}')"
  if [[ "$dead" == "1" ]]; then
    break
  fi
  sleep 30
done

status="$(tmux display-message -p -t "$UPSTREAM_SESSION" '#{pane_dead_status}')"
if [[ "$status" != "0" ]]; then
  echo "UPSTREAM NGMT FINALIZATION FAILED WITH STATUS $status; FIM NOT STARTED"
  exit 4
fi

echo "UPSTREAM NGMT FINALIZATION PASSED; STARTING FIM"
exec bash scripts/finalize_current_evidence.sh
