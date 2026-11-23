from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from PIL import Image

from .._pdf_io import bbox_iou, clip_box, page_bbox, round_bbox
from ..models import BBox, ExtractedTable, TableCell
from ..text import extract_horiz_rules, extract_words
from .base import RunStats
from .rss import peak_rss_mb

_CROP_EXPAND_PT = 2.0
_STRICT_SETTINGS = {"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict"}


def _best_iou_table(tables: Sequence[Any], clip_bbox: BBox) -> Any | None:
    best = None
    best_iou = 0.0
    for tbl in tables:
        tb = tuple(float(v) for v in tbl.bbox)
        iou = bbox_iou((tb[0], tb[1], tb[2], tb[3]), clip_bbox)
        if iou > best_iou:
            best_iou = iou
            best = tbl
    return best if best_iou >= 0.3 else None


def _find_best_pdfplumber_table(page: Any, clip_bbox: BBox) -> Any | None:
    bounds = page_bbox(page)
    padded = (
        clip_bbox[0] - _CROP_EXPAND_PT,
        clip_bbox[1] - _CROP_EXPAND_PT,
        clip_bbox[2] + _CROP_EXPAND_PT,
        clip_bbox[3] + _CROP_EXPAND_PT,
    )
    clamped = clip_box(padded, bounds)
    target = page.crop(clamped) if clamped is not None else page
    # Try stroked ruling lines first so shaded background rects do not bisect merged cells.
    for settings in (_STRICT_SETTINGS, None):
        found = (target.find_tables() if settings is None else target.find_tables(table_settings=settings)) or ()
        best = _best_iou_table(found, clip_bbox) if found else None
        if best is not None:
            return best
    if clamped is not None:
        for settings in (_STRICT_SETTINGS, None):
            found = (page.find_tables() if settings is None else page.find_tables(table_settings=settings)) or ()
            best = _best_iou_table(found, clip_bbox) if found else None
            if best is not None:
                return best
    return None


def _track_bounds(starts: Sequence[float], end: float, count: int, span_start: float, span_end: float) -> list[float]:
    if len(starts) == count and count > 0:
        return [*starts, end]
    if count <= 0:
        return []
    step = (span_end - span_start) / count
    return [span_start + i * step for i in range(count + 1)]


def _has_horiz_rule_at(
    horiz_rules: Sequence[BBox],
    x0: float,
    x1: float,
    y: float,
    y_tol: float = 2.5,
) -> bool:
    cw = max(1.0, x1 - x0)
    for rx0, ry0, rx1, ry1 in horiz_rules:
        if ry0 <= y + y_tol and ry1 >= y - y_tol:
            if min(rx1, x1) - max(rx0, x0) >= 0.5 * cw:
                return True
    return False


def _word_items(words: Sequence[Any]) -> list[tuple[str, BBox]]:
    out: list[tuple[str, BBox]] = []
    for w in words:
        if hasattr(w, "text") and hasattr(w, "bbox"):
            txt = str(w.text).strip()
            if txt:
                out.append((txt, tuple(float(v) for v in w.bbox)))  # type: ignore[arg-type]
        elif isinstance(w, dict):
            txt = str(w.get("text", "")).strip()
            if txt:
                out.append((txt, (float(w["x0"]), float(w["top"]), float(w["x1"]), float(w["bottom"]))))
    return out


def _words_in_rect(word_items: Sequence[tuple[str, BBox]], rect: BBox) -> str:
    hits: list[tuple[float, float, str]] = []
    for txt, wb in word_items:
        cx = 0.5 * (wb[0] + wb[2])
        cy = 0.5 * (wb[1] + wb[3])
        if rect[0] <= cx <= rect[2] and rect[1] <= cy <= rect[3]:
            hits.append((round(cy, 1), wb[0], txt))
    hits.sort(key=lambda item: (item[0], item[1]))
    return " ".join(txt for _, _, txt in hits)


def _convert_pdfplumber_table(
    tbl: Any,
    fallback_bbox: BBox,
    *,
    page: Any = None,
    words: Sequence[Any] | None = None,
) -> ExtractedTable:
    raw_rows = tbl.extract(x_tolerance=1.5)
    num_rows = len(raw_rows)
    num_cols = max((len(r) for r in raw_rows), default=0)
    if num_rows == 0 or num_cols == 0:
        return ExtractedTable(bbox=fallback_bbox, num_rows=0, num_cols=0, cells=(), source="pdfplumber", confidence=0.0)

    tb = tuple(float(v) for v in tbl.bbox)
    col_objs = getattr(tbl, "columns", ())
    row_objs = getattr(tbl, "rows", ())
    col_bounds = _track_bounds(
        [float(c.bbox[0]) for c in col_objs] if col_objs else (),
        max((float(c.bbox[2]) for c in col_objs), default=tb[2]),
        num_cols,
        tb[0],
        tb[2],
    )
    row_bounds = _track_bounds(
        [float(r.bbox[1]) for r in row_objs] if row_objs else (),
        max((float(r.bbox[3]) for r in row_objs), default=tb[3]),
        num_rows,
        tb[1],
        tb[3],
    )
    # col.bbox[2] / row.bbox[3] expands across merged cells; use consecutive track bounds for midpoints.
    col_mids = [(col_bounds[i] + col_bounds[i + 1]) * 0.5 for i in range(num_cols)]
    row_mids = [(row_bounds[i] + row_bounds[i + 1]) * 0.5 for i in range(num_rows)]

    cells: list[TableCell] = []
    occupied: set[tuple[int, int]] = set()
    cell_at: dict[tuple[int, int], int] = {}
    for r_idx, row_vals in enumerate(raw_rows):
        row_cells = getattr(row_objs[r_idx], "cells", ()) if r_idx < len(row_objs) else ()
        for c_idx, val in enumerate(row_vals):
            rect_val = row_cells[c_idx] if c_idx < len(row_cells) else None
            # pdfplumber sets covered slots of a merged cell to None.
            if val is None and rect_val is None:
                continue
            if (r_idx, c_idx) in occupied:
                continue
            text = " ".join(str(val or "").split())
            if rect_val is not None:
                cx0, cy0, cx1, cy1 = (float(v) for v in rect_val)
                cell_box: BBox | None = (cx0, cy0, cx1, cy1)
                cspan = max(1, sum(1 for xm in col_mids[c_idx:] if cx0 - 0.5 <= xm <= cx1 + 0.5))
                rspan = max(1, sum(1 for ym in row_mids[r_idx:] if cy0 - 0.5 <= ym <= cy1 + 0.5))
                cspan = min(cspan, num_cols - c_idx)
                rspan = min(rspan, num_rows - r_idx)
            else:
                cell_box, rspan, cspan = None, 1, 1
            while cspan > 1 and any((r_idx, c_idx + dc) in occupied for dc in range(cspan)):
                cspan -= 1
            while rspan > 1 and any(
                (r_idx + dr, c_idx + dc) in occupied for dr in range(rspan) for dc in range(cspan)
            ):
                rspan -= 1
            cell_idx = len(cells)
            for dr in range(rspan):
                for dc in range(cspan):
                    occupied.add((r_idx + dr, c_idx + dc))
                    cell_at[(r_idx + dr, c_idx + dc)] = cell_idx
            cells.append(
                TableCell(
                    row=r_idx,
                    col=c_idx,
                    rowspan=rspan,
                    colspan=cspan,
                    text=text,
                    bbox=cell_box,
                    is_header=(r_idx == 0),
                    confidence=1.0,
                )
            )

    if len(occupied) < num_rows * num_cols:
        src_page = page if page is not None else getattr(tbl, "page", None)
        horiz_rules = extract_horiz_rules(src_page, page_bbox(src_page)) if src_page is not None else ()
        if words is not None:
            w_items = _word_items(words)
        elif src_page is not None:
            w_items = _word_items(extract_words(src_page))
        else:
            w_items = []

        for r_idx in range(num_rows):
            for c_idx in range(num_cols):
                if (r_idx, c_idx) in occupied:
                    continue
                # Check if open slot continues the cell immediately above without a separating horizontal rule.
                above_idx = cell_at.get((r_idx - 1, c_idx)) if r_idx > 0 else None
                if above_idx is not None:
                    above = cells[above_idx]
                    span_cols = range(above.col, above.col + above.colspan)
                    if (
                        above.row + above.rowspan == r_idx
                        and all((r_idx, cc) not in occupied for cc in span_cols)
                    ):
                        ax0 = above.bbox[0] if above.bbox is not None else col_bounds[above.col]
                        ax1 = above.bbox[2] if above.bbox is not None else col_bounds[above.col + above.colspan]
                        if not _has_horiz_rule_at(horiz_rules, ax0, ax1, row_bounds[r_idx]):
                            ext_r = 1
                            while (
                                r_idx + ext_r < num_rows
                                and all((r_idx + ext_r, cc) not in occupied for cc in span_cols)
                                and not _has_horiz_rule_at(horiz_rules, ax0, ax1, row_bounds[r_idx + ext_r])
                            ):
                                ext_r += 1
                            new_y1 = row_bounds[r_idx + ext_r]
                            new_bbox: BBox = (
                                ax0,
                                above.bbox[1] if above.bbox is not None else row_bounds[above.row],
                                ax1,
                                new_y1,
                            )
                            ext_text = _words_in_rect(w_items, (ax0, row_bounds[r_idx], ax1, new_y1)) if w_items else ""
                            merged_text = " ".join(part for part in (above.text, ext_text) if part)
                            cells[above_idx] = TableCell(
                                row=above.row,
                                col=above.col,
                                rowspan=above.rowspan + ext_r,
                                colspan=above.colspan,
                                text=merged_text,
                                bbox=new_bbox,
                                is_header=above.is_header,
                                confidence=above.confidence,
                            )
                            for dr in range(ext_r):
                                for cc in span_cols:
                                    occupied.add((r_idx + dr, cc))
                                    cell_at[(r_idx + dr, cc)] = above_idx
                            continue

                # Otherwise merge consecutive unseparated open slots downward into one cell and backfill text.
                cx0, cx1 = col_bounds[c_idx], col_bounds[c_idx + 1]
                rspan = 1
                while (
                    r_idx + rspan < num_rows
                    and (r_idx + rspan, c_idx) not in occupied
                    and not _has_horiz_rule_at(horiz_rules, cx0, cx1, row_bounds[r_idx + rspan])
                ):
                    rspan += 1
                interp_box: BBox = (cx0, row_bounds[r_idx], cx1, row_bounds[r_idx + rspan])
                fill_text = _words_in_rect(w_items, interp_box) if w_items else ""
                new_idx = len(cells)
                for dr in range(rspan):
                    occupied.add((r_idx + dr, c_idx))
                    cell_at[(r_idx + dr, c_idx)] = new_idx
                cells.append(
                    TableCell(
                        row=r_idx,
                        col=c_idx,
                        rowspan=rspan,
                        colspan=1,
                        text=fill_text,
                        bbox=interp_box,
                        is_header=(r_idx == 0),
                        confidence=1.0,
                    )
                )

    return ExtractedTable(
        bbox=round_bbox(tb),
        num_rows=num_rows,
        num_cols=num_cols,
        cells=tuple(cells),
        source="pdfplumber",
        confidence=1.0,
    )


class PdfPlumberTableExtractor:
    name = "pdfplumber"

    def __init__(self) -> None:
        self.last_stats: RunStats | None = None

    def extract_from_page(
        self,
        page: Any,
        table_boxes_pt: Sequence[BBox],
        *,
        words: Sequence[Any] | None = None,
        is_raster: bool = False,
        ocr_backend: str = "rapidocr-onnx",
        ocr_lang: str | None = None,
        device: str | None = None,
        dpi: int = 200,
    ) -> list[ExtractedTable]:
        del is_raster, ocr_backend, ocr_lang, device, dpi
        t0 = time.perf_counter()
        out: list[ExtractedTable] = []
        for box in table_boxes_pt:
            tbl = _find_best_pdfplumber_table(page, box)
            if tbl is None:
                out.append(ExtractedTable(bbox=box, num_rows=0, num_cols=0, cells=(), source="pdfplumber", confidence=0.0))
            else:
                out.append(_convert_pdfplumber_table(tbl, box, page=page, words=words))
        self.last_stats = RunStats(
            load_sec=0.0,
            infer_sec=round(time.perf_counter() - t0, 4),
            pages=1,
            peak_rss_mb=peak_rss_mb(),
            in_process=True,
        )
        return out

    def extract(
        self,
        image: Image.Image,
        table_bbox_pt: BBox,
        tokens: Sequence[tuple[str, BBox]] = (),
    ) -> ExtractedTable:
        del image, tokens
        return ExtractedTable(bbox=table_bbox_pt, num_rows=0, num_cols=0, cells=(), source="pdfplumber", confidence=0.0)
