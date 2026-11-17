from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pymupdf

from .backends import CachedLayoutBackend, FigureGrouper, LayoutBackend, get as get_backend
from .backends.layout_image import render_bgr
from .models import FIGURE_FOOTER_FRAC, FIGURE_MIN_AREA, BBox, FigureBox, FigureSource

DEFAULT_DPI = 150
LAYOUT_BOX_CONFIDENCE = 0.95

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _to_pt_box(box_px: Sequence[float], img_size: tuple[int, int], page_rect: pymupdf.Rect) -> BBox:
    iw, ih = img_size
    sx = float(page_rect.width) / iw
    sy = float(page_rect.height) / ih
    return (
        round(box_px[0] * sx, 3),
        round(box_px[1] * sy, 3),
        round(box_px[2] * sx, 3),
        round(box_px[3] * sy, 3),
    )


def _table_boxes_px(
    items: list[dict[str, Any]],
    img_size: tuple[int, int],
    page_size: tuple[float, float],
    footer_frac: float,
) -> list[list[float]]:
    iw, ih = img_size
    dw, dh = page_size
    sx = iw / dw if dw > 0 else 1.0
    sy = ih / dh if dh > 0 else 1.0
    tables: list[list[float]] = []
    for it in items:
        if it.get("label") != "table" or not it.get("bbox") or len(it["bbox"]) != 4:
            continue
        b = [it["bbox"][0] * sx, it["bbox"][1] * sy, it["bbox"][2] * sx, it["bbox"][3] * sy]
        if b[2] > b[0] and b[3] > b[1] and b[1] < footer_frac * ih:
            tables.append(b)
    return tables


def detect_figures(
    pdf_path: str | Path,
    pages: Sequence[int] | None = None,
    *,
    layout: str | LayoutBackend = "heron-onnx",
    grouper: str | FigureGrouper = "v1_anchor",
    dpi: int = DEFAULT_DPI,
    device: str = "auto",
    min_area: float = FIGURE_MIN_AREA,
    footer_frac: float = FIGURE_FOOTER_FRAC,
    cache_dir: str | Path | None = None,
    include_tables: bool = True,
) -> dict[int, list[FigureBox]]:
    path = Path(pdf_path)
    cache_root = (
        Path(cache_dir)
        if cache_dir is not None
        else _REPO_ROOT / "outputs" / "08_pdf_figure_crops" / path.stem / "_layout_cache"
    )
    backend = get_backend("layout", layout, device=device, dpi=dpi) if isinstance(layout, str) else layout
    fig_grouper = get_backend("grouper", grouper) if isinstance(grouper, str) else grouper
    fig_source: FigureSource = getattr(fig_grouper, "source", "v1_group")

    doc = pymupdf.open(str(path))
    results: dict[int, list[FigureBox]] = {}
    try:
        target_pages = list(range(1, len(doc) + 1)) if pages is None else sorted(set(pages))
        for pno in target_pages:
            if not (1 <= pno <= len(doc)):
                raise ValueError(f"page {pno} out of range 1..{len(doc)}")
        if not target_pages:
            return results

        layouts = CachedLayoutBackend(backend, cache_root).analyze(path, target_pages)

        for pno in target_pages:
            page = doc[pno - 1]
            bgr, img_size = render_bgr(page, dpi)
            page_layout = layouts[pno]

            iw, ih = img_size
            raw_figs = fig_grouper.group(bgr, page_layout, min_area)
            kept_figs = [
                b for b in raw_figs if (b[2] - b[0]) * (b[3] - b[1]) >= min_area * iw * ih and b[1] < footer_frac * ih
            ]
            kept_figs.sort(key=lambda b: (b[1], b[0]))

            page_boxes: list[FigureBox] = [
                FigureBox(
                    bbox=_to_pt_box(b, img_size, page.rect),
                    kind="figure",
                    source=fig_source,
                    confidence=LAYOUT_BOX_CONFIDENCE,
                )
                for b in kept_figs
            ]
            if include_tables:
                raw_tables = _table_boxes_px(page_layout.items, img_size, page_layout.frame, footer_frac)
                raw_tables.sort(key=lambda b: (b[1], b[0]))
                page_boxes.extend(
                    FigureBox(
                        bbox=_to_pt_box(b, img_size, page.rect),
                        kind="table",
                        source="layout_table",
                        confidence=LAYOUT_BOX_CONFIDENCE,
                    )
                    for b in raw_tables
                )
            results[pno] = page_boxes
    finally:
        doc.close()

    return results
