#!/usr/bin/env bash
# Launch the HIL dashboard, reachable over the LAN / Tailscale.
# Usage:
#   ./run_dashboard.sh              # bind 0.0.0.0:8501 (default)
#   ./run_dashboard.sh 8600         # custom port
#   PORT=8600 ./run_dashboard.sh    # custom port via env
#
# Reach it from another device at  http://<this-host-ip>:<port>
#   - Tailscale (any network):  http://<tailscale-ip>:<port>   (`tailscale ip -4`)
#   - Same LAN:                 http://<lan-ip>:<port>
# Ignore the "External URL" Streamlit prints (public IP, needs port-forwarding).
set -euo pipefail

# cd to this script's folder so `Home.py` is found regardless of where it's run.
cd "$(dirname "$0")"

# Activate the project venv if it's the usual one and not already active.
if [[ -z "${VIRTUAL_ENV:-}" && -f "$HOME/hil_env/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/hil_env/bin/activate"
fi

PORT="${1:-${PORT:-8501}}"

echo "Starting HIL dashboard on 0.0.0.0:${PORT} (Ctrl-C to stop)"
exec streamlit run Home.py \
  --server.address 0.0.0.0 \
  --server.port "${PORT}" \
  --server.headless true \
  --browser.gatherUsageStats false
