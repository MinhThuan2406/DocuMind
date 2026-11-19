from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium

from .backends import CachedLayoutBackend, FigureGrouper, LayoutBackend, get as get_backend
from .backends.base import RunStats
from .backends.layout_cache import file_sha256
from .backends.layout_image import render_bgr
from .backends.rss import peak_rss_mb
from .models import FIGURE_FOOTER_FRAC, FIGURE_MIN_AREA, BBox, FigureBox

DEFAULT_DPI = 150
DEFAULT_RENDER_BATCH_SIZE = 16
LAYOUT_BOX_CONFIDENCE = 0.95

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _to_pt_box(box_px: Sequence[float], img_size: tuple[int, int], page_size_pt: tuple[float, float]) -> BBox:
    iw, ih = img_size
    pw, ph = page_size_pt
    sx = float(pw) / iw
    sy = float(ph) / ih
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
) -> list[tuple[BBox, float]]:
    iw, ih = img_size
    dw, dh = page_size
    sx = iw / dw if dw > 0 else 1.0
    sy = ih / dh if dh > 0 else 1.0
    tables: list[tuple[BBox, float]] = []
    for it in items:
        if it.get("label") != "table" or not it.get("bbox") or len(it["bbox"]) != 4:
            continue
        b = (it["bbox"][0] * sx, it["bbox"][1] * sy, it["bbox"][2] * sx, it["bbox"][3] * sy)
        if b[2] > b[0] and b[3] > b[1] and b[1] < footer_frac * ih:
            raw_score = it.get("score")
            score = round(float(raw_score), 4) if raw_score is not None else LAYOUT_BOX_CONFIDENCE
            tables.append((b, score))
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
    pdfium_doc: Any = None,
    pdf_hash: str | None = None,
    warnings_out: list[dict[str, Any]] | None = None,
) -> dict[int, list[FigureBox]]:
    path = Path(pdf_path)
    cache_root = (
        Path(cache_dir)
        if cache_dir is not None
        else _REPO_ROOT / "outputs" / "08_pdf_figure_crops" / path.stem / "_layout_cache"
    )
    backend = get_backend("layout", layout, device=device, dpi=dpi) if isinstance(layout, str) else layout
    fig_grouper = get_backend("grouper", grouper) if isinstance(grouper, str) else grouper

    owns_doc = pdfium_doc is None
    doc = pdfium.PdfDocument(str(path)) if owns_doc else pdfium_doc
    results: dict[int, list[FigureBox]] = {}
    rendered_cache = getattr(backend, "last_rendered", None)
    try:
        target_pages = list(range(1, len(doc) + 1)) if pages is None else sorted(set(pages))
        for pno in target_pages:
            if not (1 <= pno <= len(doc)):
                raise ValueError(f"page {pno} out of range 1..{len(doc)}")
        if not target_pages:
            return results

        if isinstance(rendered_cache, dict):
            rendered_cache.clear()
        cached_backend = CachedLayoutBackend(backend, cache_root)
        resolved_hash = pdf_hash or file_sha256(path)
        can_reuse_bgr = getattr(backend, "cache_dpi", None) == dpi and isinstance(rendered_cache, dict)
        batch_size = DEFAULT_RENDER_BATCH_SIZE if isinstance(rendered_cache, dict) else len(target_pages)
        batches = [target_pages[i : i + batch_size] for i in range(0, len(target_pages), batch_size)]

        use_doc = getattr(backend, "_use_pdfium_doc", None)
        doc_ctx = use_doc(path, doc) if callable(use_doc) else nullcontext()
        batch_stats: list[Any] = []

        with doc_ctx:
            for batch in batches:
                prev_stats = getattr(backend, "last_stats", None)
                layouts = cached_backend.analyze(path, batch, pdf_hash=resolved_hash)
                cur_stats = getattr(backend, "last_stats", None)
                if cur_stats is not None and cur_stats is not prev_stats:
                    batch_stats.append(cur_stats)
                    if warnings_out is not None:
                        for w in getattr(cur_stats, "warnings", ()):
                            warnings_out.append(dict(w))

                for pno in batch:
                    page = doc[pno - 1]
                    page_size_pt = tuple(float(v) for v in page.get_size())
                    bgr = rendered_cache.pop(pno, None) if can_reuse_bgr else None
                    if bgr is not None:
                        ih, iw = bgr.shape[:2]
                        img_size = (iw, ih)
                    else:
                        bgr, img_size = render_bgr(page, dpi)
                        iw, ih = img_size
                    page_layout = layouts[pno]

                    raw_figs = fig_grouper.group(bgr, page_layout, min_area)
                    kept_figs = [
                        g
                        for g in raw_figs
                        if (g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1]) >= min_area * iw * ih
                        and g.bbox[1] < footer_frac * ih
                    ]
                    kept_figs.sort(key=lambda g: (g.bbox[1], g.bbox[0]))

                    page_boxes: list[FigureBox] = [
                        FigureBox(
                            bbox=_to_pt_box(g.bbox, img_size, page_size_pt),
                            kind="figure",
                            source=g.source,
                            confidence=round(float(g.score), 4),
                        )
                        for g in kept_figs
                    ]
                    if include_tables:
                        raw_tables = _table_boxes_px(page_layout.items, img_size, page_layout.frame, footer_frac)
                        raw_tables.sort(key=lambda item: (item[0][1], item[0][0]))
                        page_boxes.extend(
                            FigureBox(
                                bbox=_to_pt_box(box, img_size, page_size_pt),
                                kind="table",
                                source="layout_table",
                                confidence=score,
                            )
                            for box, score in raw_tables
                        )
                    results[pno] = page_boxes

        if hasattr(backend, "last_stats"):
            if len(batch_stats) == 0:
                backend.last_stats = RunStats(
                    load_sec=0.0,
                    infer_sec=0.0,
                    pages=0,
                    peak_rss_mb=peak_rss_mb(),
                    in_process=True,
                )
            elif len(batch_stats) > 1:
                backend.last_stats = RunStats(
                    load_sec=round(max(s.load_sec for s in batch_stats), 4),
                    infer_sec=round(sum(s.infer_sec for s in batch_stats), 4),
                    pages=sum(s.pages for s in batch_stats),
                    peak_rss_mb=max(s.peak_rss_mb for s in batch_stats),
                    in_process=all(s.in_process for s in batch_stats),
                    warnings=tuple(w for s in batch_stats for w in getattr(s, "warnings", ())),
                )
    finally:
        if isinstance(rendered_cache, dict):
            rendered_cache.clear()
        if owns_doc:
            doc.close()

    return results
