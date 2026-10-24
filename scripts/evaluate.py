"""Score saved run outputs with the organisers' evaluation code, loaded verbatim
from Train/Evaluation/Evaluation.ipynb (cells are executed unmodified; only the
API key is supplied to the client constructor).

usage: python scripts/evaluate.py <run.jsonl> [<scores.json>] [--dev-judge]

--dev-judge: the notebook's configured model (deepseek-v4.1-flash, a reasoning model)
exhausts its 10000-token budget on hidden reasoning for long answers and returns
empty / truncated JSON; the organisers never said which judge model they use. With
this flag the notebook code still runs unmodified (same prompts, same parsing, same
formulas), but its HTTP requests are answered by Claude Haiku 5.5 with thinking
disabled, via OpenCode's Anthropic Messages endpoint. Scores are labelled "dev_judge".
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

from kb import config  # noqa: E402

load_dotenv(ROOT / ".env")
NB = ROOT / "Train" / "Evaluation" / "Evaluation.ipynb"
BENCH = ROOT / "Train" / "Benchmarks" / "benchmark_100.jsonl"


DEV_JUDGE_MODEL = "claude-haiku-5-5"


class _JudgeResponse:
    """Minimal requests.Response look-alike in the OpenAI chat-completions shape."""

    def __init__(self, content: str, finish: str):
        self._data = {"choices": [{"message": {"content": content}, "finish_reason": finish}]}

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class _HaikuJudgeRequests:
    """Stands in for the `requests` module inside the notebook namespace (dev judge only):
    translates the notebook's OpenAI-style request into an Anthropic Messages call."""

    def __init__(self, real):
        import anthropic

        self._real = real
        self.RequestException = real.RequestException
        self._client = anthropic.Anthropic(api_key=(os.environ.get("OPENCODE_API_KEY") or config.OPENCODE_API_KEY),
                                           base_url=os.getenv("ANTHROPIC_BASE_URL", "https://opencode.ai/inference/anthropic"),
                                           timeout=300.0, max_retries=3)

    def post(self, url, json=None, headers=None, timeout=None, **kw):
        import anthropic

        msgs = json["messages"]
        system = "\n".join(m["content"] for m in msgs if m["role"] == "system")
        user = [{"role": m["role"], "content": m["content"]} for m in msgs if m["role"] != "system"]
        try:
            r = self._client.messages.create(model=DEV_JUDGE_MODEL, max_tokens=json.get("max_tokens", 10000),
                                             thinking={"type": "disabled"}, system=system, messages=user)
        except anthropic.APIError as e:
            raise self._real.RequestException(str(e)) from e
        text = "".join(b.text for b in r.content if b.type == "text")
        return _JudgeResponse(text, "length" if r.stop_reason == "max_tokens" else "stop")

    def __getattr__(self, name):
        return getattr(self._real, name)


def load_official(dev_judge: bool = False):
    ns: dict = {}
    for cell in json.loads(NB.read_text())["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] != "code" or "sample_dataset" in src:
            continue  # skip the example-usage cell only
        exec(src, ns)
    if dev_judge:
        ns["requests"] = _HaikuJudgeRequests(ns["requests"])
    return ns


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dev_judge = "--dev-judge" in sys.argv
    run_path = Path(args[0])
    suffix = ".devjudge.scores.json" if dev_judge else ".scores.json"
    out_path = Path(args[1]) if len(args) > 1 else run_path.with_suffix(suffix)
    ns = load_official(dev_judge)
    client = ns["OpenCodeClient"](api_key=(os.environ.get("OPENCODE_API_KEY") or config.OPENCODE_API_KEY))
    evaluator = ns["RAGMetricsEvaluator"](client)

    gold = {json.loads(l)["id"]: json.loads(l) for l in BENCH.open()}
    runs = [json.loads(l) for l in run_path.open()]

    def score(r):
        g = gold[r["id"]]
        item = {
            "query": g["question"],
            "ai_answer": r["answer"],
            "benchmark_answer": g["benchmark_answer"],
            "retrieved_chunks": [c["text"] for c in r["chunks"]],
            "benchmark_chunks": [c["content"] for v in g["benchmark_chunks_w_metadata"].values() for c in v],
        }
        try:
            s = ns["evaluate_dataset"](evaluator, [item], verbose=False)["per_question"][0]
        except Exception as e:
            print(f"{r['id']} eval failed: {e}", flush=True)
            return None
        s["id"] = r["id"]
        print(f"{r['id']} AnsP={s['answer_precision']:.2f} AnsR={s['answer_recall']:.2f} "
              f"CtxP={s['context_precision']:.2f} CtxR={s['context_recall']:.2f}", flush=True)
        return s

    with ThreadPoolExecutor(max_workers=6) as ex:
        per_q = [s for s in ex.map(score, runs) if s]

    def avg(k):
        return sum(min(s[k], 1.0) if k == "context_recall_capped" else s[k] for s in per_q) / max(1, len(per_q))

    summary = {
        "judge": f"dev_judge ({DEV_JUDGE_MODEL}, thinking disabled)" if dev_judge else "official (unmodified)",
        "num_questions": len(per_q),
        "num_failed": len(runs) - len(per_q),
        "avg_answer_precision": avg("answer_precision"),
        "avg_answer_recall": avg("answer_recall"),
        "avg_context_precision": avg("context_precision"),
        "avg_context_recall": avg("context_recall"),
        # reported for our own honesty: recall with each question capped at 1.0
        "avg_context_recall_capped": sum(min(s["context_recall"], 1.0) for s in per_q) / max(1, len(per_q)),
        "avg_latency_seconds": sum(r["latency_seconds"] for r in runs) / max(1, len(runs)),
        "avg_prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in runs) / max(1, len(runs)),
        "avg_completion_tokens": sum(r["usage"]["completion_tokens"] for r in runs) / max(1, len(runs)),
    }
    for fam in "ABC":
        fs = [s for s in per_q if s["id"].startswith(fam)]
        if fs:
            summary[f"family_{fam}"] = {k: round(sum(s[k] for s in fs) / len(fs), 3) for k in
                                        ("answer_precision", "answer_recall", "context_precision", "context_recall")}
    out_path.write_text(json.dumps({"summary": summary, "per_question": per_q}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
