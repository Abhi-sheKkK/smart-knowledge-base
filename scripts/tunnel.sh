#!/usr/bin/env bash
# Expose the API running on this machine through ngrok, as the organisers' API contract expects.
# One-time setup (free ngrok account):
#   ngrok config add-authtoken <your token>          # from dashboard.ngrok.com
#   claim your free static dev domain on the dashboard (e.g. abc123.ngrok-free.dev)
# Then:
#   NGROK_DOMAIN=abc123.ngrok-free.dev ./scripts/tunnel.sh      start API + tunnel (detached)
#   ./scripts/tunnel.sh stop                                     stop both
# With the static domain the URL never changes, so the API can be restarted safely;
# do not stop the tunnel after submitting the URL. The Mac is kept awake while they run.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8000}"
mkdir -p data

if [ "${1:-}" = "stop" ]; then
  pkill -f "uvicorn kb.api:app --host 127.0.0.1 --port $PORT" || true
  pkill -f "ngrok http" || true
  echo "stopped"; exit 0
fi

command -v ngrok >/dev/null || { echo "install ngrok first: brew install ngrok"; exit 1; }
[ -n "${NGROK_DOMAIN:-}" ] || { echo "set NGROK_DOMAIN to your static ngrok domain (see header of this script)"; exit 1; }

if ! curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then
  # caffeinate -i: keep the machine from idle-sleeping while the API runs
  nohup caffeinate -i .venv/bin/uvicorn kb.api:app --host 127.0.0.1 --port "$PORT" --timeout-keep-alive 75 \
    > data/api.log 2>&1 &
  for _ in $(seq 1 30); do curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
fi
curl -fsS "http://localhost:$PORT/health" >/dev/null || { echo "API did not start; see data/api.log"; exit 1; }

if ! pgrep -f "ngrok http" >/dev/null; then
  nohup caffeinate -i ngrok http "$PORT" --url "https://$NGROK_DOMAIN" --log stdout > data/ngrok.log 2>&1 &
  sleep 5
fi
echo "https://$NGROK_DOMAIN" > data/tunnel_url.txt
curl -fsS -H "ngrok-skip-browser-warning: true" "https://$NGROK_DOMAIN/health" && echo "  <- public endpoint: https://$NGROK_DOMAIN"
