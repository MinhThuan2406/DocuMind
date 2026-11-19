from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from PIL import Image

from .._pdf_io import bbox_iou, clip_box, page_bbox, round_bbox
from ..models import BBox, ExtractedTable, TableCell
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


def _track_mids(starts: Sequence[float], end: float) -> list[float]:
    if not starts:
        return []
    bounds = [*starts, end]
    return [(bounds[i] + bounds[i + 1]) * 0.5 for i in range(len(starts))]


def _convert_pdfplumber_table(tbl: Any, fallback_bbox: BBox) -> ExtractedTable:
    raw_rows = tbl.extract(x_tolerance=1.5)
    num_rows = len(raw_rows)
    num_cols = max((len(r) for r in raw_rows), default=0)
    if num_rows == 0 or num_cols == 0:
        return ExtractedTable(bbox=fallback_bbox, num_rows=0, num_cols=0, cells=(), source="pdfplumber", confidence=0.0)

    col_objs = getattr(tbl, "columns", ())
    row_objs = getattr(tbl, "rows", ())
    # col.bbox[2] / row.bbox[3] expands across merged cells; use consecutive left/top edges for track midpoints.
    col_mids = (
        _track_mids([float(c.bbox[0]) for c in col_objs], max(float(c.bbox[2]) for c in col_objs))
        if col_objs
        else []
    )
    row_mids = (
        _track_mids([float(r.bbox[1]) for r in row_objs], max(float(r.bbox[3]) for r in row_objs))
        if row_objs
        else []
    )

    cells: list[TableCell] = []
    occupied: set[tuple[int, int]] = set()
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
                cspan = max(1, sum(1 for xm in col_mids[c_idx:] if cx0 - 0.5 <= xm <= cx1 + 0.5)) if col_mids else 1
                rspan = max(1, sum(1 for ym in row_mids[r_idx:] if cy0 - 0.5 <= ym <= cy1 + 0.5)) if row_mids else 1
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
            for dr in range(rspan):
                for dc in range(cspan):
                    occupied.add((r_idx + dr, c_idx + dc))
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

    tb = tbl.bbox
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
        device: str | None = None,
        dpi: int = 200,
    ) -> list[ExtractedTable]:
        del words, is_raster, ocr_backend, device, dpi
        t0 = time.perf_counter()
        out: list[ExtractedTable] = []
        for box in table_boxes_pt:
            tbl = _find_best_pdfplumber_table(page, box)
            if tbl is None:
                out.append(ExtractedTable(bbox=box, num_rows=0, num_cols=0, cells=(), source="pdfplumber", confidence=0.0))
            else:
                out.append(_convert_pdfplumber_table(tbl, box))
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
