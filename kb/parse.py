"""PDF -> page-bounded structural units + section tree.

Every unit lives on exactly one page and belongs to exactly one section, so
retrieval can expand a hit to its whole section and every citation carries a
page number.  Structure comes from the PDF outline when present and from
font-size / boldness heading detection otherwise.  Pages with no text layer
are OCR'd with Tesseract.
"""
from __future__ import annotations

import io
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, asdict

import pymupdf

PARSER_VERSION = 2  # bump whenever parse output changes; part of the parse-cache key
MIN_UNIT_CHARS = 250
TARGET_UNIT_CHARS = 1200
MAX_UNIT_CHARS = 2200
OCR_DPI = 220


@dataclass
class Section:
    id: str
    doc_id: str
    title: str
    level: int
    parent: str | None
    path: list[str]
    page_start: int
    page_end: int = 0
    unit_ids: list[str] = field(default_factory=list)


@dataclass
class Unit:
    id: str
    doc_id: str
    page: int  # 1-based
    section_id: str
    kind: str  # text | table | ocr
    text: str


@dataclass
class ParsedDoc:
    doc_id: str
    filename: str
    n_pages: int
    sections: list[Section]
    units: list[Unit]
    ocr_pages: list[int]

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "n_pages": self.n_pages,
            "ocr_pages": self.ocr_pages,
            "sections": [asdict(s) for s in self.sections],
            "units": [asdict(u) for u in self.units],
        }


# ---------------------------------------------------------------- helpers

_WS = re.compile(r"[ \t ]+")
_DIGITS = re.compile(r"\d+")


def normalize_text(text: str) -> str:
    """Unicode compatibility folding (ligatures like \ufb01 -> fi, full-width digits) and
    re-joining thousands separators split by PDF spacing ("60, 000" -> "60,000")."""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"(?<=\d), (?=\d{3}\b)", ",", text)


def _clean(text: str) -> str:
    text = normalize_text(text).replace("\u00ad", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # de-hyphenate line breaks
    text = _WS.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _norm_key(text: str) -> str:
    return _DIGITS.sub("#", text.strip().lower())[:80]


def _is_bold(span: dict) -> bool:
    return bool(span["flags"] & 16) or "bold" in span["font"].lower()


def _ocr_page(page: pymupdf.Page) -> str:
    import pytesseract
    from PIL import Image

    pix = page.get_pixmap(dpi=OCR_DPI)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)


def _split_long(text: str, limit: int = MAX_UNIT_CHARS) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, buf = [], ""
    for sent in re.split(r"(?<=[.;:])\s+(?=[A-Z(\d])", text):
        if buf and len(buf) + len(sent) > TARGET_UNIT_CHARS:
            parts.append(buf.strip())
            buf = ""
        buf += sent + " "
    if buf.strip():
        parts.append(buf.strip())
    # sentences longer than the limit get hard-split
    out = []
    for p in parts:
        while len(p) > limit:
            out.append(p[:TARGET_UNIT_CHARS])
            p = p[TARGET_UNIT_CHARS:]
        out.append(p)
    return out


_OCR_NUM_HEAD = re.compile(r"^(chapter|section|part|annex|appendix)\b|^(\d{1,2}(\.\d{1,2}){0,3}\.?|[A-H]\.|[IVX]{1,4}\.)\s+[A-Z]", re.I)


def _ocr_heading_level(line: str) -> int | None:
    """Heading guess for OCR text, which has no font info: numbered or ALL-CAPS short lines."""
    t = line.strip()
    if not (4 <= len(t) <= 110) or t.endswith((".", ",", ";")) and not re.match(r"^\d+\.\s", t):
        return None
    if not re.search(r"[A-Za-z]{3}", t):
        return None
    m = re.match(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+[A-Z]", t)
    if m and len(t.split()) <= 12:
        return 1 + m.group(1).count(".")
    if _OCR_NUM_HEAD.match(t) and len(t.split()) <= 12:
        return 2
    letters = [c for c in t if c.isalpha()]
    if len(letters) >= 6 and sum(c.isupper() for c in letters) / len(letters) > 0.85 and len(t.split()) <= 10:
        return 1
    return None


# ---------------------------------------------------------------- page model

@dataclass
class _Line:
    text: str
    size: float
    bold: bool
    y0: float
    y1: float


@dataclass
class _Block:
    kind: str  # text | table | ocr
    y0: float
    lines: list[_Line] = field(default_factory=list)
    text: str = ""


def _page_blocks(page: pymupdf.Page, with_tables: bool) -> tuple[list[_Block], bool]:
    """Return blocks in reading order, and whether the page needed OCR."""
    raw = page.get_text("text").strip()
    if len(raw) < 30 and page.get_images():
        txt = _clean(_ocr_page(page))
        return ([_Block("ocr", 0, text=txt)] if txt else []), True

    blocks: list[_Block] = []
    table_rects = []
    # find_tables (lines strategy) only finds ruled tables; skip the costly call on pages without vector lines
    if with_tables and len(page.get_drawings()) >= 4:
        try:
            for tab in page.find_tables().tables:
                if tab.row_count < 2 or tab.col_count < 2:
                    continue
                md = tab.to_markdown(clean=False).strip()
                if len(md) < 40:
                    continue
                table_rects.append(pymupdf.Rect(tab.bbox))
                blocks.append(_Block("table", tab.bbox[1], text=md))
        except Exception:
            pass

    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        r = pymupdf.Rect(b["bbox"])
        if any(r.intersect(t).get_area() > 0.6 * max(r.get_area(), 1) for t in table_rects):
            continue
        blk = _Block("text", b["bbox"][1])
        for ln in b["lines"]:
            spans = [s for s in ln["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = normalize_text("".join(s["text"] for s in spans)).strip()
            by_size: Counter = Counter()
            for s in spans:
                by_size[s["size"]] += len(s["text"])
            size = by_size.most_common(1)[0][0]  # dominant size, not max (math/superscripts)
            bold = sum(len(s["text"]) for s in spans if _is_bold(s)) >= 0.6 * len(text)
            blk.lines.append(_Line(text, round(size * 2) / 2, bold, ln["bbox"][1], ln["bbox"][3]))
        if blk.lines:
            blocks.append(blk)
    if table_rects:
        blocks.sort(key=lambda b: b.y0)
    return blocks, False


# ---------------------------------------------------------------- parser

class _HeadingModel:
    """Decides whether a line is a heading, and its level, from doc-wide font stats."""

    def __init__(self, size_counts: Counter, heading_sizes: Counter):
        self.body = size_counts.most_common(1)[0][0] if size_counts else 10.0
        big = sorted((s for s, n in heading_sizes.items() if s >= self.body + 1.5 and n >= 2), reverse=True)
        self.levels = {s: i + 1 for i, s in enumerate(big[:3])}

    def level(self, line: _Line, block_lines: int) -> int | None:
        t = line.text
        if not (3 <= len(t) <= 160) or not re.search(r"[A-Za-z]{3}", t):
            return None
        if re.fullmatch(r"[\d\s.,%()₹$€\-–/]+", t):
            return None
        if line.size >= self.body + 1.5 and block_lines > 4:
            return None  # big font inside a long block is a pull-quote / formula, not a heading
        if line.size in self.levels:
            return self.levels[line.size]
        if line.size >= self.body + 1.5:
            return min(len(self.levels) + 1, 3)
        if line.bold and line.size >= self.body - 0.5 and block_lines <= 3 and len(t) <= 110 and not t.endswith((",", ";")):
            return 4
        return None


def _extract_range(args: tuple[str, int, int, bool]) -> list[tuple[list[_Block], bool]]:
    path, start, end, with_tables = args
    doc = pymupdf.open(path)
    return [_page_blocks(doc[i], with_tables) for i in range(start, end)]


class PagePool:
    """Process pool for page extraction that survives native crashes (MuPDF can
    segfault on odd pages): a dead worker breaks the pool, so the pool is rebuilt
    and the unfinished ranges are retried page by page, then without tables."""

    def __init__(self, workers: int, page_timeout: float = 120.0):
        self.workers = workers
        self.page_timeout = page_timeout
        self._ex = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._shutdown()

    def _executor(self):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        if self._ex is None:
            self._ex = ProcessPoolExecutor(self.workers, mp_context=mp.get_context("spawn"))
        return self._ex

    def _shutdown(self):
        if self._ex is not None:
            procs = list((getattr(self._ex, "_processes", None) or {}).values())
            self._ex.shutdown(wait=False, cancel_futures=True)
            for p in procs:
                p.kill()  # a hung native call never returns on its own
            self._ex = None

    def run(self, jobs: list[tuple]) -> dict[tuple, list | None]:
        """Run _extract_range jobs; returns job -> result, or None for jobs that crashed or timed out."""
        from concurrent.futures import TimeoutError as FutTimeout
        from concurrent.futures.process import BrokenProcessPool

        out: dict[tuple, list | None] = {}
        futs = {self._executor().submit(_extract_range, j): j for j in jobs}
        for f, j in futs.items():
            try:
                out[j] = f.result(timeout=self.page_timeout * max(1, j[2] - j[1]))
            except (BrokenProcessPool, FutTimeout, OSError):
                out[j] = None
            except Exception:
                out[j] = None
        if any(v is None for v in out.values()):
            self._shutdown()  # pool may be broken; rebuild lazily
        return out


def extract_pages(path: str, with_tables: bool = True, pool: PagePool | None = None, batch: int = 25) -> list[tuple[list[_Block], bool]]:
    """Per-page blocks for every page. With a PagePool, ranges run in parallel and
    crashing pages fall back to single-page retries, then to text-only extraction."""
    n = len(pymupdf.open(path))
    if pool is None:
        return [page for s in range(0, n, batch) for page in _extract_range((path, s, min(s + batch, n), with_tables))]

    pages: dict[int, tuple[list[_Block], bool]] = {}
    jobs = [(path, s, min(s + batch, n), with_tables) for s in range(0, n, batch)]
    failed = []
    for j, res in pool.run(jobs).items():
        if res is None:
            failed.extend(range(j[1], j[2]))
        else:
            pages.update({j[1] + k: r for k, r in enumerate(res)})
    for attempt_tables in ([with_tables, False] if with_tables else [False]):
        if not failed:
            break
        retry = pool.run([(path, i, i + 1, attempt_tables) for i in failed])
        failed = []
        for j, res in retry.items():
            if res is None:
                failed.append(j[1])
            else:
                pages[j[1]] = res[0]
    for i in failed:  # still crashing even without tables: keep the page, empty
        pages[i] = ([], False)
    return [pages[i] for i in range(n)]


def parse_pdf(path: str, doc_id: str, filename: str, with_tables: bool = True, pool=None) -> ParsedDoc:
    doc = pymupdf.open(path)
    n = len(doc)
    pages: list[list[_Block]] = []
    ocr_pages: list[int] = []
    for i, (blocks, ocr) in enumerate(extract_pages(path, with_tables, pool)):
        pages.append(blocks)
        if ocr:
            ocr_pages.append(i + 1)

    # repeated running headers / footers: same normalised line on many pages
    line_pages: Counter = Counter()
    for blocks in pages:
        seen = set()
        for b in blocks:
            for ln in b.lines:
                seen.add(_norm_key(ln.text))
        line_pages.update(seen)
    for blocks in pages:  # OCR'd pages have no line objects; count their raw lines too
        for b in blocks:
            if b.kind == "ocr":
                line_pages.update({_norm_key(l) for l in b.text.split("\n") if l.strip()})
    thresh = max(3, int(0.25 * n))
    boiler = {k for k, c in line_pages.items() if c >= thresh and len(k) < 120}

    size_counts: Counter = Counter()
    heading_sizes: Counter = Counter()
    for blocks in pages:
        for b in blocks:
            for ln in b.lines:
                size_counts[ln.size] += len(ln.text)
                if len(ln.text) < 160:
                    heading_sizes[ln.size] += 1
    hm = _HeadingModel(size_counts, heading_sizes)

    # outline (bookmarks): title -> (level, page); used as primary skeleton when rich enough
    toc = [(lvl, title.strip(), pg) for lvl, title, pg in doc.get_toc() if title.strip() and 1 <= pg <= n]
    use_toc = len(toc) >= 5
    toc_by_page: dict[int, list[tuple[int, str]]] = {}
    for lvl, title, pg in toc if use_toc else []:
        toc_by_page.setdefault(pg, []).append((min(lvl, 4), title))
    toc_depth = max((l for l, _, _ in toc), default=0) if use_toc else 0

    sections: list[Section] = []
    units: list[Unit] = []
    stack: list[Section] = []

    def open_section(title: str, level: int, page: int):
        while stack and stack[-1].level >= level:
            stack.pop()
        parent = stack[-1] if stack else None
        sec = Section(
            id=f"{doc_id}:s{len(sections)}",
            doc_id=doc_id,
            title=title[:200],
            level=level,
            parent=parent.id if parent else None,
            path=(parent.path if parent else []) + [title[:120]],
            page_start=page,
            page_end=page,
        )
        sections.append(sec)
        stack.append(sec)

    open_section(filename.rsplit(".", 1)[0], 0, 1)  # root

    buf: list[str] = []
    buf_kind = "text"

    def flush(page: int, force: bool = True):
        nonlocal buf, buf_kind
        text = _clean("\n".join(buf))
        buf = []
        if not text:
            return
        sec = stack[-1]
        # glue a tiny trailing piece onto the previous unit of the same section+page
        if len(text) < MIN_UNIT_CHARS and units and units[-1].section_id == sec.id and units[-1].page == page \
                and units[-1].kind == "text" and len(units[-1].text) + len(text) <= MAX_UNIT_CHARS:
            units[-1].text += "\n" + text
            return
        for piece in _split_long(text):
            u = Unit(f"{doc_id}:u{len(units)}", doc_id, page, sec.id, buf_kind, piece)
            units.append(u)
            sec.unit_ids.append(u.id)
            sec.page_end = max(sec.page_end, page)

    for pi, blocks in enumerate(pages):
        pno = pi + 1
        pending_toc = list(toc_by_page.get(pno, []))
        # outline entries for this page open before any content on it
        for lvl, title in pending_toc:
            flush(pno)
            open_section(title, lvl, pno)
        toc_titles = {_norm_key(t) for _, t in pending_toc}

        for b in blocks:
            if b.kind in ("table", "ocr"):
                flush(pno)
                if b.kind == "table":
                    buf_kind = "table"
                    buf = [b.text]
                    flush(pno)
                    buf_kind = "text"
                else:
                    buf_kind = "ocr"
                    for para in re.split(r"\n\s*\n", b.text):
                        lines = [l for l in para.split("\n") if _norm_key(l) not in boiler]
                        if not lines:
                            continue
                        lvl = _ocr_heading_level(lines[0]) if len(lines) <= 3 else None
                        if lvl is not None:
                            flush(pno)
                            open_section(" ".join(l.strip() for l in lines)[:200], lvl + toc_depth if use_toc else lvl, pno)
                            continue
                        buf.append(" ".join(l.strip() for l in lines))
                        if sum(len(x) for x in buf) >= TARGET_UNIT_CHARS:
                            flush(pno)
                    flush(pno)
                    buf_kind = "text"
                continue
            para: list[str] = []
            for ln in b.lines:
                key = _norm_key(ln.text)
                if key in boiler:
                    continue
                if key in toc_titles:
                    continue  # already opened from the outline
                lvl = hm.level(ln, len(b.lines))
                if lvl is not None:
                    lvl = lvl + toc_depth if use_toc else lvl
                    if para:
                        buf.append(" ".join(para))
                        para = []
                    flush(pno)
                    cur = stack[-1]
                    if cur.level == lvl and not cur.unit_ids and cur.page_start == pno and len(cur.title) < 150:
                        cur.title = (cur.title + " " + ln.text)[:200]  # wrapped multi-line heading
                        cur.path[-1] = cur.title[:120]
                    else:
                        open_section(ln.text, lvl, pno)
                    continue
                para.append(ln.text)
            if para:
                buf.append(" ".join(para))
            if sum(len(x) for x in buf) >= TARGET_UNIT_CHARS:
                flush(pno)
        flush(pno)  # units never cross pages

    # keep sections that hold units or have descendants that do; roll page ranges up
    by_id = {s.id: s for s in sections}
    keep = {sections[0].id}
    for s in sections:
        if s.unit_ids:
            cur = s
            while cur and cur.id not in keep:
                keep.add(cur.id)
                cur = by_id.get(cur.parent) if cur.parent else None
    sections = [s for s in sections if s.id in keep]
    for s in sections:
        p = by_id.get(s.parent) if s.parent else None
        while p:
            p.page_end = max(p.page_end, s.page_end)
            p = by_id.get(p.parent) if p.parent else None
    return ParsedDoc(doc_id, filename, n, sections, units, ocr_pages)
