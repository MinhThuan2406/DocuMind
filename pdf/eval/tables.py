from __future__ import annotations

import argparse
import html
import json
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pdf.eval.gt import PT_PER_PX_150DPI, append_run_log, omnidoc_image_split  # noqa: E402
from pdf.eval.metrics import iou, prf, strip_markup  # noqa: E402


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


def _html_merged_spans(gt_html: str) -> set[tuple[int, int, int, int]]:
    root = parse_table_html(gt_html)
    occupied: set[tuple[int, int]] = set()
    spans: set[tuple[int, int, int, int]] = set()
    row_idx = 0
    for tr in root.children:
        if tr.tag != "tr":
            continue
        col_idx = 0
        for cell in tr.children:
            if cell.tag != "td":
                continue
            while (row_idx, col_idx) in occupied:
                col_idx += 1
            rspan = max(1, cell.rowspan)
            cspan = max(1, cell.colspan)
            for rr in range(row_idx, row_idx + rspan):
                for cc in range(col_idx, col_idx + cspan):
                    occupied.add((rr, cc))
            if rspan > 1 or cspan > 1:
                spans.add((row_idx, col_idx, rspan, cspan))
            col_idx += cspan
        row_idx += 1
    return spans


def _pred_merged_spans(pred_cells: Any) -> set[tuple[int, int, int, int]]:
    items = getattr(pred_cells, "cells", pred_cells) or ()
    spans: set[tuple[int, int, int, int]] = set()
    for cell in items:
        if isinstance(cell, dict):
            row = int(cell["row"])
            col = int(cell["col"])
            rspan = int(cell.get("rowspan", 1))
            cspan = int(cell.get("colspan", 1))
        else:
            row = int(cell.row)
            col = int(cell.col)
            rspan = int(cell.rowspan)
            cspan = int(cell.colspan)
        if rspan > 1 or cspan > 1:
            spans.add((row, col, rspan, cspan))
    return spans


def merged_cell_span_f1(pred_cells: Sequence[Any], gt_html: str) -> float:
    pred_spans = _pred_merged_spans(pred_cells)
    gt_spans = _html_merged_spans(gt_html)
    if not pred_spans and not gt_spans:
        return 1.0
    if not pred_spans or not gt_spans:
        return 0.0
    tp = len(pred_spans & gt_spans)
    _, _, f1 = prf(tp, len(pred_spans) - tp, len(gt_spans) - tp)
    return round(f1, 4)


def load_omnidoc_tables(
    *,
    language: str = "english",
    split: str | None = "dev",
    limit: int | None = None,
    gt_file: Path | None = None,
    img_dir: Path | None = None,
) -> list[dict[str, Any]]:
    if split is not None and split not in ("dev", "test"):
        raise ValueError(f"unknown split: {split!r}")
    resolved_gt = gt_file if gt_file is not None else (ROOT / "datasets" / "OmniDocBench" / "OmniDocBench.json")
    resolved_imgs = img_dir if img_dir is not None else (ROOT / "datasets" / "OmniDocBench" / "images")
    raw = json.loads(resolved_gt.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for doc in raw:
        info = doc.get("page_info", {})
        if language and info.get("page_attribute", {}).get("language") != language:
            continue
        img_name = str(info.get("image_path", ""))
        if not img_name:
            continue
        if split is not None and omnidoc_image_split(img_name) != split:
            continue
        img_path = resolved_imgs / img_name
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
                "image_name": img_name,
                "image_path": img_path,
                "bbox": bbox,
                "html": gt_html,
                "has_merged": ("colspan=" in gt_html or "rowspan=" in gt_html),
            })
            if limit is not None and len(out) >= limit:
                return out
    return out


def _match_tables_by_iou(
    preds: Sequence[dict[str, Any]],
    gts: Sequence[dict[str, Any]],
    *,
    min_iou: float = 0.5,
) -> dict[int, int]:
    candidates = sorted(
        (
            (-iou(p["bbox"], [v * PT_PER_PX_150DPI for v in g["bbox"]]), pi, gi)
            for pi, p in enumerate(preds)
            for gi, g in enumerate(gts)
            if iou(p["bbox"], [v * PT_PER_PX_150DPI for v in g["bbox"]]) >= min_iou
        ),
        key=lambda item: item[0],
    )
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches: dict[int, int] = {}
    for _, pi, gi in candidates:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches[gi] = pi
    return matches


def bench_omnidoc(
    *,
    mode: str = "oracle",
    split: str = "dev",
    device: str = "auto",
    ocr_backend: str = "rapidocr-onnx",
    limit: int | None = None,
    with_ocr: bool = True,
) -> dict[str, Any]:
    if mode not in ("oracle", "e2e"):
        raise ValueError(f"unknown mode: {mode!r}")

    from PIL import Image

    from pdf.backends import get as get_backend
    from pdf.backends.rss import peak_rss_mb
    from pdf.backends.tatr_onnx import table_to_html

    samples = load_omnidoc_tables(language="english", split=split, limit=limit)
    if not samples:
        raise FileNotFoundError(f"OmniDocBench table samples not found for split={split!r}")

    teds_struct_all: list[float] = []
    teds_struct_merged: list[float] = []
    teds_full_all: list[float] = []
    span_tp = 0
    span_fp = 0
    span_fn = 0

    by_img: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        by_img[s["image_path"]].append(s)

    n = len(samples)

    if mode == "oracle":
        extractor = get_backend("table", "tatr-onnx", device=device)
        t_load0 = time.perf_counter()
        extractor.ensure_loaded()
        load_sec = time.perf_counter() - t_load0

        ocr_rec = get_backend("ocr", ocr_backend, device=device) if with_ocr else None
        if ocr_rec is not None and hasattr(ocr_rec, "ensure_loaded"):
            ocr_rec.ensure_loaded()

        parsed_ok = 0
        infer_sec = 0.0
        for img_path, items in by_img.items():
            with Image.open(img_path) as im:
                rgb = im.convert("RGB")
                for item in items:
                    x0, y0, x1, y1 = item["bbox"]
                    crop = rgb.crop(
                        (max(0, int(x0) - 4), max(0, int(y0) - 4), min(rgb.width, int(x1) + 4), min(rgb.height, int(y1) + 4))
                    )
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
                    ps = _pred_merged_spans(tbl.cells)
                    gs = _html_merged_spans(item["html"])
                    hit = len(ps & gs)
                    span_tp += hit
                    span_fp += len(ps) - hit
                    span_fn += len(gs) - hit

        span_f1 = round(prf(span_tp, span_fp, span_fn)[2], 4) if (span_tp + span_fp + span_fn) > 0 else 1.0
        return {
            "dataset": "OmniDocBench_en_tables",
            "mode": "oracle",
            "split": split,
            "backend": "tatr-onnx",
            "device": device,
            "tables": n,
            "merged_tables": len(teds_struct_merged),
            "parsed_nonempty": parsed_ok,
            "teds_struct": round(sum(teds_struct_all) / n, 4),
            "teds_struct_merged": round(sum(teds_struct_merged) / max(1, len(teds_struct_merged)), 4),
            "teds_full": round(sum(teds_full_all) / len(teds_full_all), 4) if teds_full_all else None,
            "merged_cell_span_f1": span_f1,
            "load_sec": round(load_sec, 4),
            "ms_per_table": round(infer_sec * 1000.0 / n, 2),
            "ram_mb": peak_rss_mb(),
        }

    from pdf import extract_ir
    from pdf.eval.gt import build_raster_pdf

    img_paths = list(by_img.keys())
    preds_by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pdf_path = tmp_path / "omnidoc_tables_e2e.pdf"
        build_raster_pdf(img_paths, pdf_path)

        t0 = time.perf_counter()
        if with_ocr:
            ir = extract_ir(
                pdf_path,
                figures=True,
                ocr=True,
                tables=True,
                table_backend="auto",
                ocr_backend=ocr_backend,
                device=device,
                structure=False,
            )
            for tbl_dict in ir["tables"]:
                preds_by_page[int(tbl_dict["page"])].append(tbl_dict)
        else:
            import pdfplumber

            from pdf.figures import detect_figures
            from pdf.tables import extract_tables_on_page, table_to_ir_dict

            page_nums = list(range(1, len(img_paths) + 1))
            figs = detect_figures(
                pdf_path,
                page_nums,
                layout="heron-onnx",
                device=device,
                cache_dir=tmp_path / "layout_cache",
                include_tables=True,
            )
            with pdfplumber.open(str(pdf_path)) as doc:
                for pno in page_nums:
                    tboxes = [b.bbox for b in figs.get(pno, ()) if b.kind == "table"]
                    if not tboxes:
                        continue
                    parsed = extract_tables_on_page(
                        doc.pages[pno - 1],
                        tboxes,
                        words=[],
                        is_raster=False,
                        extractor="auto",
                        device=device,
                    )
                    for idx, tbl in enumerate(parsed, start=1):
                        if tbl.num_rows > 0 and tbl.num_cols > 0:
                            preds_by_page[pno].append(
                                table_to_ir_dict(f"tbl_{pno}_{idx}", f"elem_{pno}_{idx}", pno, pdf_path.name, tbl)
                            )
        wall_sec = time.perf_counter() - t0

    det_tp = 0
    det_fp = 0
    det_fn = 0
    teds_struct_matched: list[float] = []

    for pno, img_path in enumerate(img_paths, start=1):
        gt_items = by_img[img_path]
        pred_items = preds_by_page.get(pno, [])
        matches = _match_tables_by_iou(pred_items, gt_items, min_iou=0.5)
        det_tp += len(matches)
        det_fp += len(pred_items) - len(matches)
        det_fn += len(gt_items) - len(matches)

        for gi, item in enumerate(gt_items):
            gs = _html_merged_spans(item["html"])
            if gi in matches:
                pred = pred_items[matches[gi]]
                ts = teds_score(pred["html"], item["html"], structure_only=True)
                teds_struct_all.append(ts)
                teds_struct_matched.append(ts)
                if item["has_merged"]:
                    teds_struct_merged.append(ts)
                if with_ocr:
                    teds_full_all.append(teds_score(pred["html"], item["html"], structure_only=False))
                ps = _pred_merged_spans(pred["cells"])
                hit = len(ps & gs)
                span_tp += hit
                span_fp += len(ps) - hit
                span_fn += len(gs) - hit
            else:
                teds_struct_all.append(0.0)
                if item["has_merged"]:
                    teds_struct_merged.append(0.0)
                if with_ocr:
                    teds_full_all.append(0.0)
                span_fn += len(gs)

    dp, dr, df1 = prf(det_tp, det_fp, det_fn)
    span_f1 = round(prf(span_tp, span_fp, span_fn)[2], 4) if (span_tp + span_fp + span_fn) > 0 else 1.0
    return {
        "dataset": "OmniDocBench_en_tables",
        "mode": "e2e",
        "split": split,
        "backend": "auto",
        "device": device,
        "pages": len(img_paths),
        "tables": n,
        "pred_tables": det_tp + det_fp,
        "matched_tables": det_tp,
        "det_precision": round(dp, 4),
        "det_recall": round(dr, 4),
        "det_f1": round(df1, 4),
        "teds_struct": round(sum(teds_struct_all) / n, 4),
        "teds_struct_matched": round(sum(teds_struct_matched) / max(1, len(teds_struct_matched)), 4),
        "teds_struct_merged": round(sum(teds_struct_merged) / max(1, len(teds_struct_merged)), 4),
        "teds_full": round(sum(teds_full_all) / len(teds_full_all), 4) if teds_full_all else None,
        "merged_cell_span_f1": span_f1,
        "wall_sec": round(wall_sec, 4),
        "ms_per_table": round(wall_sec * 1000.0 / n, 2),
        "ram_mb": peak_rss_mb(),
    }


def bench_autosar(*, split: str = "dev", device: str = "auto", max_tables: int = 40) -> list[dict[str, Any]]:
    if split not in ("dev", "test"):
        raise ValueError(f"unknown split: {split!r}")

    import pdfplumber

    from pdf.backends import get as get_backend, reset_cache
    from pdf.backends.rss import peak_rss_mb
    from pdf.figures import detect_figures
    from pdf.tables import extract_tables_on_page
    from pdf.text import extract_words

    split_pages = {
        "dev": [("AUTOSAR_CP_EXP_VFB.pdf", [17, 74, 100, 117, 131])],
        "test": [("AUTOSAR_CP_SWS_OS.pdf", [1, 131, 209, 240, 302])],
    }
    cases: list[tuple[Path, int, list[tuple[float, float, float, float]]]] = []
    for fname, pages in split_pages[split]:
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
            "dataset": f"AUTOSAR_native_{split}",
            "split": split,
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
    ap.add_argument("--mode", default="oracle", choices=["oracle", "e2e"], help="oracle=GT crop+TATR; e2e=extract_ir/layout+AutoTableExtractor")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--final", action="store_true", help="Required when --split test")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--limit", type=int, default=50, help="Cap OmniDocBench tables (None via 0)")
    ap.add_argument("--no-ocr", action="store_true", help="Measure structure TEDS-Struct without RapidOCR")
    ap.add_argument("--log-file", default=str(ROOT / "outputs" / "eval_runs.jsonl"))
    args = ap.parse_args(argv)

    if args.split == "test" and not args.final:
        print("REFUSED: --split test requires --final to prevent accidental test-split evaluation.", file=sys.stderr)
        return 2

    cap = None if args.limit == 0 else args.limit
    logged_results: list[dict[str, Any]] = []
    if args.task in ("omnidoc", "all"):
        res = bench_omnidoc(
            mode=args.mode,
            split=args.split,
            device=args.device,
            limit=cap,
            with_ocr=not args.no_ocr,
        )
        print(json.dumps(res, ensure_ascii=False))
        logged_results.append(res)
    if args.task in ("autosar", "all"):
        rows = bench_autosar(split=args.split, device=args.device)
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))
        logged_results.extend(rows)

    if args.task in ("omnidoc", "all") and logged_results:
        append_run_log(
            Path(args.log_file),
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task": args.task,
                "mode": args.mode,
                "split": args.split,
                "final": args.final,
                "results": logged_results,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
