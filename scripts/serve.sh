#!/usr/bin/env bash
# Run the API without Docker (Linux server). Requires Python 3.12 and tesseract-ocr.
#   sudo apt-get install -y tesseract-ocr        # Debian / Ubuntu
#   cp .env.example .env && edit .env            # OPENCODE_API_KEY, LANGFUSE_* keys
#   ./scripts/serve.sh
set -euo pipefail
cd "$(dirname "$0")/.."

command -v tesseract >/dev/null || { echo "tesseract-ocr is not installed"; exit 1; }
[ -f .env ] || { echo ".env is missing (copy .env.example)"; exit 1; }

if [ ! -d .venv ]; then
  python3.12 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi

exec .venv/bin/uvicorn kb.api:app --host 0.0.0.0 --port "${PORT:-8000}" --timeout-keep-alive 3600
