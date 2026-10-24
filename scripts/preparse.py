"""Parse every PDF in the given folders into the parse cache (no LLM needed)."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kb import config
from kb.index import _parse_cached
from kb.parse import PagePool


def main():
    pdfs = sorted(p for f in sys.argv[1:] for p in Path(f).glob("*.pdf"))
    t0 = time.time()
    with PagePool(config.WORKERS) as pool:
        for i, p in enumerate(pdfs):
            t = time.time()
            d = _parse_cached(p, f"d{i:03d}", pool)
            print(f"{time.time()-t:6.1f}s {d['n_pages']:5d}p {len(d['units']):6d}u {len(d['sections']):5d}s ocr={len(d['ocr_pages'])} {p.name}", flush=True)
    print(f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
