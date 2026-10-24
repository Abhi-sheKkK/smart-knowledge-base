#!/usr/bin/env bash
# Expose the API running on this machine through ngrok, as the organisers' API contract expects.
# One-time setup (free ngrok account):
#   ngrok config add-authtoken <your token>          # from dashboard.ngrok.com
#   free accounts get a static dev domain automatically (e.g. abc123.ngrok-free.dev)
# Then:
#   ./scripts/tunnel.sh                                          start API + tunnel (detached)
#   NGROK_DOMAIN=abc123.ngrok-free.dev ./scripts/tunnel.sh      same, pinning a specific domain
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

if ! curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then
  # caffeinate -i: keep the machine from idle-sleeping while the API runs
  nohup caffeinate -i .venv/bin/uvicorn kb.api:app --host 127.0.0.1 --port "$PORT" --timeout-keep-alive 75 \
    > data/api.log 2>&1 &
  for _ in $(seq 1 30); do curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
fi
curl -fsS "http://localhost:$PORT/health" >/dev/null || { echo "API did not start; see data/api.log"; exit 1; }

if ! pgrep -f "ngrok http" >/dev/null; then
  URL_ARG=()
  [ -n "${NGROK_DOMAIN:-}" ] && URL_ARG=(--url "https://$NGROK_DOMAIN")
  nohup caffeinate -i ngrok http "$PORT" ${URL_ARG[@]+"${URL_ARG[@]}"} --log stdout > data/ngrok.log 2>&1 &
fi
URL=""
for _ in $(seq 1 30); do  # the agent's local API reports the public URL
  URL=$(curl -fsS http://127.0.0.1:4040/api/tunnels 2>/dev/null | grep -oE '"public_url":"https://[^"]+' | head -1 | cut -d'"' -f4 || true)
  [ -n "$URL" ] && break; sleep 1
done
[ -n "$URL" ] || { echo "ngrok did not come up; see data/ngrok.log"; exit 1; }
echo "$URL" > data/tunnel_url.txt
curl -fsS -H "ngrok-skip-browser-warning: true" "$URL/health" && echo "  <- public endpoint: $URL"
