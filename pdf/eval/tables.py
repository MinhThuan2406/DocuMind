from __future__ import annotations

import argparse
import html
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pdf.backends import get as get_backend, reset_cache  # noqa: E402
from pdf.backends.rss import peak_rss_mb  # noqa: E402
from pdf.backends.tatr_onnx import table_to_html  # noqa: E402
from pdf.eval.metrics import strip_markup  # noqa: E402


@dataclass
class _HtmlNode:
    tag: str
    colspan: int = 1
    rowspan: int = 1
    text: str = ""
    children: list[_HtmlNode] = field(default_factory=list)


class _TableTreeParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root: _HtmlNode | None = None
        self._stack: list[_HtmlNode] = []
        self._in_cell = False
        self._cell_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t == "table":
            node = _HtmlNode("table")
            if self.root is None:
                self.root = node
            if self._stack:
                self._stack[-1].children.append(node)
            self._stack.append(node)
        elif t == "tr" and self._stack:
            node = _HtmlNode("tr")
            self._stack[-1].children.append(node)
            self._stack.append(node)
        elif t in ("td", "th") and self._stack:
            attr_map = {k.lower(): (v or "") for k, v in attrs}
            try:
                cspan = max(1, int(attr_map.get("colspan", "1") or "1"))
            except ValueError:
                cspan = 1
            try:
                rspan = max(1, int(attr_map.get("rowspan", "1") or "1"))
            except ValueError:
                rspan = 1
            node = _HtmlNode("td", colspan=cspan, rowspan=rspan)
            self._stack[-1].children.append(node)
            self._stack.append(node)
            self._in_cell = True
            self._cell_parts = []

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        norm = "td" if t in ("td", "th") else t
        if norm in ("table", "tr", "td") and self._stack and self._stack[-1].tag == norm:
            node = self._stack.pop()
            if norm == "td":
                self._in_cell = False
                raw = html.unescape("".join(self._cell_parts))
                node.text = " ".join(strip_markup(raw).split())
                self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_parts.append(data)


def parse_table_html(markup: str) -> _HtmlNode:
    parser = _TableTreeParser()
    parser.feed(markup)
    parser.close()
    return parser.root or _HtmlNode("table")


def _norm_levenshtein(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb] / max(la, lb)


def _node_size(node: _HtmlNode) -> int:
    return 1 + sum(_node_size(ch) for ch in node.children)


def _postorder(node: _HtmlNode) -> tuple[list[_HtmlNode], list[int], list[int]]:
    # Standard Zhang-Shasha postorder indexing with leftmost-leaf descendants and keyroots.
    nodes: list[_HtmlNode] = []
    lmds: list[int] = []

    def _walk(cur: _HtmlNode) -> int:
        first_lmd = -1
        for ch in cur.children:
            ch_lmd = _walk(ch)
            if first_lmd == -1:
                first_lmd = ch_lmd
        idx = len(nodes)
        nodes.append(cur)
        lmd = idx if first_lmd == -1 else first_lmd
        lmds.append(lmd)
        return lmd

    _walk(node)
    last_by_lmd: dict[int, int] = {}
    for idx, lmd in enumerate(lmds):
        last_by_lmd[lmd] = idx
    keyroots = sorted(last_by_lmd.values())
    return nodes, lmds, keyroots


def _rename_cost(
    a: _HtmlNode,
    b: _HtmlNode,
    *,
    structure_only: bool,
    lev_cache: dict[tuple[str, str], float] | None = None,
) -> float:
    if a.tag != b.tag or a.colspan != b.colspan or a.rowspan != b.rowspan:
        return 1.0
    if structure_only or a.tag != "td":
        return 0.0
    if a.text == b.text:
        return 0.0
    if lev_cache is not None:
        key = (a.text, b.text) if a.text <= b.text else (b.text, a.text)
        cached = lev_cache.get(key)
        if cached is not None:
            return cached
        val = _norm_levenshtein(a.text, b.text)
        lev_cache[key] = val
        return val
    return _norm_levenshtein(a.text, b.text)


def teds_score(pred_html: str, gt_html: str, *, structure_only: bool = False) -> float:
    t1 = parse_table_html(pred_html)
    t2 = parse_table_html(gt_html)
    n1, n2 = _node_size(t1), _node_size(t2)
    if max(n1, n2) == 0:
        return 1.0

    nodes1, lmds1, kr1 = _postorder(t1)
    nodes2, lmds2, kr2 = _postorder(t2)
    tdist = [[0.0] * (n2 + 1) for _ in range(n1 + 1)]
    lev_cache: dict[tuple[str, str], float] = {}

    for i in kr1:
        for j in kr2:
            li, lj = lmds1[i], lmds2[j]
            rows = (i - li + 1) + 1
            cols = (j - lj + 1) + 1
            fd = [[0.0] * cols for _ in range(rows)]
            for x in range(1, rows):
                fd[x][0] = fd[x - 1][0] + 1.0
            for y in range(1, cols):
                fd[0][y] = fd[0][y - 1] + 1.0
            for x in range(1, rows):
                pi = li + x - 1
                for y in range(1, cols):
                    pj = lj + y - 1
                    if lmds1[pi] == li and lmds2[pj] == lj:
                        cost = _rename_cost(nodes1[pi], nodes2[pj], structure_only=structure_only, lev_cache=lev_cache)
                        val = min(fd[x - 1][y] + 1.0, fd[x][y - 1] + 1.0, fd[x - 1][y - 1] + cost)
                        fd[x][y] = val
                        tdist[pi + 1][pj + 1] = val
                    else:
                        sx = lmds1[pi] - li
                        sy = lmds2[pj] - lj
                        fd[x][y] = min(
                            fd[x - 1][y] + 1.0,
                            fd[x][y - 1] + 1.0,
                            fd[sx][sy] + tdist[pi + 1][pj + 1],
                        )

    return round(max(0.0, 1.0 - tdist[n1][n2] / max(n1, n2)), 4)


def load_omnidoc_tables(
    *,
    language: str = "english",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    gt_file = ROOT / "datasets" / "OmniDocBench" / "OmniDocBench.json"
    img_dir = ROOT / "datasets" / "OmniDocBench" / "images"
    raw = json.loads(gt_file.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for doc in raw:
        info = doc.get("page_info", {})
        if language and info.get("page_attribute", {}).get("language") != language:
            continue
        img_path = img_dir / info.get("image_path", "")
        if not img_path.is_file():
            continue
        for det in doc.get("layout_dets", ()):
            if det.get("category_type") != "table" or not det.get("html"):
                continue
            poly = det.get("poly") or ()
            if len(poly) < 8:
                continue
            xs, ys = poly[0::2], poly[1::2]
            bbox = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))
            gt_html = str(det["html"])
            out.append({
                "image_path": img_path,
                "bbox": bbox,
                "html": gt_html,
                "has_merged": ("colspan=" in gt_html or "rowspan=" in gt_html),
            })
            if limit is not None and len(out) >= limit:
                return out
    return out


def bench_omnidoc(
    *,
    device: str = "auto",
    ocr_backend: str = "rapidocr-onnx",
    limit: int | None = None,
    with_ocr: bool = True,
) -> dict[str, Any]:
    from PIL import Image

    samples = load_omnidoc_tables(language="english", limit=limit)
    if not samples:
        raise FileNotFoundError("OmniDocBench table samples not found")

    extractor = get_backend("table", "tatr-onnx", device=device)
    t_load0 = time.perf_counter()
    extractor.ensure_loaded()
    load_sec = time.perf_counter() - t_load0

    ocr_rec = get_backend("ocr", ocr_backend, device=device) if with_ocr else None
    if ocr_rec is not None and hasattr(ocr_rec, "ensure_loaded"):
        ocr_rec.ensure_loaded()

    teds_struct_all: list[float] = []
    teds_struct_merged: list[float] = []
    teds_full_all: list[float] = []
    parsed_ok = 0
    infer_sec = 0.0

    by_img: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        by_img[s["image_path"]].append(s)

    for img_path, items in by_img.items():
        with Image.open(img_path) as im:
            rgb = im.convert("RGB")
            for item in items:
                x0, y0, x1, y1 = item["bbox"]
                crop = rgb.crop((max(0, int(x0) - 4), max(0, int(y0) - 4), min(rgb.width, int(x1) + 4), min(rgb.height, int(y1) + 4)))
                tokens: list[tuple[str, tuple[float, float, float, float]]] = []
                if ocr_rec is not None:
                    tokens = [(ln.text, ln.bbox) for ln in ocr_rec.recognize(crop)]
                t0 = time.perf_counter()
                tbl = extractor.extract(crop, (0.0, 0.0, float(crop.width), float(crop.height)), tokens)
                infer_sec += time.perf_counter() - t0
                if tbl.num_rows > 0 and tbl.num_cols > 0:
                    parsed_ok += 1
                pred_html = table_to_html(tbl)
                ts = teds_score(pred_html, item["html"], structure_only=True)
                teds_struct_all.append(ts)
                if item["has_merged"]:
                    teds_struct_merged.append(ts)
                if with_ocr:
                    teds_full_all.append(teds_score(pred_html, item["html"], structure_only=False))

    n = len(samples)
    return {
        "dataset": "OmniDocBench_en_tables",
        "backend": "tatr-onnx",
        "device": device,
        "tables": n,
        "merged_tables": len(teds_struct_merged),
        "parsed_nonempty": parsed_ok,
        "teds_struct": round(sum(teds_struct_all) / n, 4),
        "teds_struct_merged": round(sum(teds_struct_merged) / max(1, len(teds_struct_merged)), 4),
        "teds_full": round(sum(teds_full_all) / len(teds_full_all), 4) if teds_full_all else None,
        "load_sec": round(load_sec, 4),
        "tatr_ms_per_table": round(infer_sec * 1000.0 / n, 2),
        "ram_mb": peak_rss_mb(),
    }


def bench_autosar(*, device: str = "auto", max_tables: int = 40) -> list[dict[str, Any]]:
    import pdfplumber

    from pdf.figures import detect_figures
    from pdf.tables import extract_tables_on_page
    from pdf.text import extract_words

    src_pages = [
        ("AUTOSAR_CP_EXP_VFB.pdf", [17, 35, 50, 74, 100]),
        ("AUTOSAR_CP_SWS_OS.pdf", [1, 131, 209, 240, 302]),
    ]
    cases: list[tuple[Path, int, list[tuple[float, float, float, float]]]] = []
    for fname, pages in src_pages:
        pdf_path = ROOT / "datasets" / "AUTOSAR" / fname
        if not pdf_path.is_file():
            continue
        figs = detect_figures(pdf_path, pages, layout="heron", device=device)
        for pno in pages:
            tboxes = [b.bbox for b in figs.get(pno, ()) if b.kind == "table"]
            if tboxes:
                cases.append((pdf_path, pno, tboxes))

    rows: list[dict[str, Any]] = []
    for backend_name in ("pdfplumber", "tatr-onnx", "auto"):
        reset_cache()
        extractor = get_backend("table", backend_name, device=device)
        t_load0 = time.perf_counter()
        if hasattr(extractor, "ensure_loaded"):
            extractor.ensure_loaded()
        load_sec = time.perf_counter() - t_load0

        n_tables = 0
        nonempty = 0
        merged_found = 0
        total_cells = 0
        nonempty_text_cells = 0
        header_cells = 0
        t_infer = 0.0

        for pdf_path, pno, tboxes in cases:
            with pdfplumber.open(str(pdf_path)) as doc:
                page = doc.pages[pno - 1]
                words = extract_words(page)
                use_boxes = tboxes[: max(0, max_tables - n_tables)]
                if not use_boxes:
                    break
                t0 = time.perf_counter()
                parsed = extract_tables_on_page(page, use_boxes, words=words, extractor=extractor, device=device)
                t_infer += time.perf_counter() - t0
                for tbl in parsed:
                    n_tables += 1
                    if tbl.num_rows > 0 and tbl.num_cols > 0:
                        nonempty += 1
                    if any(c.rowspan > 1 or c.colspan > 1 for c in tbl.cells):
                        merged_found += 1
                    total_cells += len(tbl.cells)
                    nonempty_text_cells += sum(1 for c in tbl.cells if c.text)
                    header_cells += sum(1 for c in tbl.cells if c.is_header)
            if n_tables >= max_tables:
                break

        rows.append({
            "dataset": "AUTOSAR_native_sample",
            "backend": backend_name,
            "tables": n_tables,
            "parsed_nonempty": nonempty,
            "tables_with_merged": merged_found,
            "total_cells": total_cells,
            "text_cells": nonempty_text_cells,
            "header_cells": header_cells,
            "load_sec": round(load_sec, 4),
            "ms_per_table": round(t_infer * 1000.0 / max(1, n_tables), 2),
            "ram_mb": peak_rss_mb(),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark PDF table structure backends on OmniDocBench and AUTOSAR.")
    ap.add_argument("--task", default="all", choices=["omnidoc", "autosar", "all"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--limit", type=int, default=50, help="Cap OmniDocBench tables (None via 0)")
    ap.add_argument("--no-ocr", action="store_true", help="Measure pure TATR structure TEDS-Struct without RapidOCR")
    args = ap.parse_args(argv)

    cap = None if args.limit == 0 else args.limit
    if args.task in ("omnidoc", "all"):
        res = bench_omnidoc(device=args.device, limit=cap, with_ocr=not args.no_ocr)
        print(json.dumps(res, ensure_ascii=False))
    if args.task in ("autosar", "all"):
        for row in bench_autosar(device=args.device):
            print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
