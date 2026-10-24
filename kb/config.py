"""Runtime settings, read from the environment (and .env if present)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# organiser-provided OpenCode key (repo is private); OPENCODE_API_KEY in the environment / .env overrides it
OPENCODE_API_KEY = os.getenv("OPENCODE_API_KEY") or "oc_sk_4aa146c4984d_oylFr89_Yv4iUQ3or7UyYgq0lOar33MX"
OPENCODE_BASE_URL = os.getenv("OPENCODE_BASE_URL", "https://opencode.ai/inference/openai/v1/chat/completions")

ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://opencode.ai/inference/anthropic")

# one model for routing / filtering, optionally a different one for the final answer
FAST_MODEL = os.getenv("FAST_MODEL", "claude-haiku-5-5")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", FAST_MODEL)
# sent as reasoning_effort when set; empty = provider default (the model always reasons a little)
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "")

# USD per 1M tokens (input, output), used for the cost we report in traces and Langfuse
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),  # prompts <= 100K tokens
    "claude-sonnet-5-5": (2.00, 10.00),
    "deepseek-v4.1-flash": (0.30, 1.20),  # OpenCode Zen list price (conservative)
}


def price(model: str) -> tuple[float, float]:
    """PRICE_IN_PER_M / PRICE_OUT_PER_M override the table for every model."""
    if os.getenv("PRICE_IN_PER_M") and os.getenv("PRICE_OUT_PER_M"):
        return float(os.getenv("PRICE_IN_PER_M")), float(os.getenv("PRICE_OUT_PER_M"))
    return PRICES.get(model, (0.0, 0.0))

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")  # quantised bge-small runs ~11 texts/s on CPU
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data" / "collections"))
WORKERS = int(os.getenv("WORKERS", str(max(2, (os.cpu_count() or 4) - 2))))
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "8"))

LANGFUSE_ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
