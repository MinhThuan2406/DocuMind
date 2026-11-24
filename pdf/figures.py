from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium

from ._pdf_io import pdfium_box_to_page_box, pdfium_page_box, round_bbox
from .backends import CachedLayoutBackend, FigureGrouper, LayoutBackend, get as get_backend
from .backends.base import RunStats
from .backends.layout_cache import file_sha256
from .backends.layout_image import render_bgr
from .backends.rss import peak_rss_mb
from .models import FIGURE_FOOTER_FRAC, FIGURE_MIN_AREA, BBox, FigureBox, SemanticRegion

DEFAULT_DPI = 150
DEFAULT_RENDER_BATCH_SIZE = 16
LAYOUT_BOX_CONFIDENCE = 0.95
_RAW_PICTURE_FALLBACK_MIN_SCORE = 0.70

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SEMANTIC_LABELS = frozenset(
    {"title", "section_header", "caption", "footnote", "list_item", "formula", "code", "page_header", "page_footer"}
)


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


def _to_page_pt_box(box_px: Sequence[float], img_size: tuple[int, int], page_box: BBox) -> BBox:
    page_size_pt = (max(0.0, page_box[2] - page_box[0]), max(0.0, page_box[3] - page_box[1]))
    return round_bbox(pdfium_box_to_page_box(_to_pt_box(box_px, img_size, page_size_pt), page_box))


def _layout_boxes_px(
    items: list[dict[str, Any]],
    labels: Collection[str],
    img_size: tuple[int, int],
    page_size: tuple[float, float],
    footer_frac: float,
    *,
    min_area: float = 0.0,
    min_score: float = 0.0,
    clamp: bool = False,
) -> list[tuple[BBox, float]]:
    iw, ih = img_size
    dw, dh = page_size
    sx = iw / dw if dw > 0 else 1.0
    sy = ih / dh if dh > 0 else 1.0
    area_thresh = min_area * iw * ih
    boxes: list[tuple[BBox, float]] = []
    for it in items:
        label = str(it.get("label", "")).lower()
        if label not in labels or not it.get("bbox") or len(it["bbox"]) != 4:
            continue
        x0, y0, x1, y1 = (
            float(it["bbox"][0]) * sx,
            float(it["bbox"][1]) * sy,
            float(it["bbox"][2]) * sx,
            float(it["bbox"][3]) * sy,
        )
        if clamp:
            x0, x1 = max(0.0, min(float(iw), x0)), max(0.0, min(float(iw), x1))
            y0, y1 = max(0.0, min(float(ih), y0)), max(0.0, min(float(ih), y1))
        if x1 <= x0 or y1 <= y0 or y0 >= footer_frac * ih:
            continue
        if area_thresh > 0.0 and (x1 - x0) * (y1 - y0) < area_thresh:
            continue
        raw_score = it.get("score")
        score = round(float(raw_score), 4) if raw_score is not None else LAYOUT_BOX_CONFIDENCE
        if min_score > 0.0 and score < min_score:
            continue
        boxes.append(((x0, y0, x1, y1), score))
    return boxes


def _semantic_regions_pt(
    items: list[dict[str, Any]],
    img_size: tuple[int, int],
    page_size: tuple[float, float],
    page_box: BBox,
) -> list[SemanticRegion]:
    iw, ih = img_size
    dw, dh = page_size
    sx = iw / dw if dw > 0 else 1.0
    sy = ih / dh if dh > 0 else 1.0
    regions: list[SemanticRegion] = []
    for it in items:
        label = str(it.get("label", "")).lower()
        bbox = it.get("bbox")
        if label not in _SEMANTIC_LABELS or not bbox or len(bbox) != 4:
            continue
        bx0 = max(0.0, min(float(iw), float(bbox[0]) * sx))
        by0 = max(0.0, min(float(ih), float(bbox[1]) * sy))
        bx1 = max(0.0, min(float(iw), float(bbox[2]) * sx))
        by1 = max(0.0, min(float(ih), float(bbox[3]) * sy))
        if bx1 <= bx0 or by1 <= by0:
            continue
        pt_box = _to_page_pt_box((bx0, by0, bx1, by1), img_size, page_box)
        if pt_box[2] <= pt_box[0] or pt_box[3] <= pt_box[1]:
            continue
        raw_score = it.get("score")
        score = round(float(raw_score), 4) if raw_score is not None else LAYOUT_BOX_CONFIDENCE
        regions.append(SemanticRegion(bbox=pt_box, label=label, confidence=score))
    regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
    return regions


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
    semantic_regions_out: dict[int, list[SemanticRegion]] | None = None,
    page_boxes: Mapping[int, BBox] | None = None,
    figure_pages: Collection[int] | None = None,
    raw_picture_pages: Collection[int] | None = None,
    table_pages: Collection[int] | None = None,
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
                    page_box = (
                        page_boxes[pno]
                        if (page_boxes is not None and pno in page_boxes)
                        else pdfium_page_box(page)
                    )
                    run_grouper = figure_pages is None or pno in figure_pages
                    bgr = rendered_cache.pop(pno, None) if can_reuse_bgr else None
                    if bgr is not None:
                        ih, iw = bgr.shape[:2]
                        img_size = (iw, ih)
                    elif run_grouper:
                        bgr, img_size = render_bgr(page, dpi)
                        iw, ih = img_size
                    else:
                        iw = max(1, round(max(0.0, page_box[2] - page_box[0]) * dpi / 72.0))
                        ih = max(1, round(max(0.0, page_box[3] - page_box[1]) * dpi / 72.0))
                        img_size = (iw, ih)
                    page_layout = layouts[pno]
                    if semantic_regions_out is not None:
                        semantic_regions_out[pno] = _semantic_regions_pt(
                            page_layout.items, img_size, page_layout.frame, page_box
                        )

                    detected_boxes: list[FigureBox] = []
                    if run_grouper:
                        raw_figs = fig_grouper.group(bgr, page_layout, min_area)
                        kept_figs = [
                            g
                            for g in raw_figs
                            if (g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1]) >= min_area * iw * ih
                            and g.bbox[1] < footer_frac * ih
                        ]
                        kept_figs.sort(key=lambda g: (g.bbox[1], g.bbox[0]))
                        detected_boxes.extend(
                            FigureBox(
                                bbox=_to_page_pt_box(g.bbox, img_size, page_box),
                                kind="figure",
                                source=g.source,
                                confidence=round(float(g.score), 4),
                            )
                            for g in kept_figs
                        )
                    elif raw_picture_pages is not None and pno in raw_picture_pages:
                        raw_pics = _layout_boxes_px(
                            page_layout.items,
                            ("picture", "figure"),
                            img_size,
                            page_layout.frame,
                            footer_frac,
                            min_area=min_area,
                            min_score=_RAW_PICTURE_FALLBACK_MIN_SCORE,
                            clamp=True,
                        )
                        raw_pics.sort(key=lambda item: (item[0][1], item[0][0]))
                        detected_boxes.extend(
                            FigureBox(
                                bbox=_to_page_pt_box(box, img_size, page_box),
                                kind="figure",
                                source="layout_picture",
                                confidence=score,
                            )
                            for box, score in raw_pics
                        )

                    if include_tables and (table_pages is None or pno in table_pages):
                        raw_tables = _layout_boxes_px(
                            page_layout.items, ("table",), img_size, page_layout.frame, footer_frac
                        )
                        raw_tables.sort(key=lambda item: (item[0][1], item[0][0]))
                        detected_boxes.extend(
                            FigureBox(
                                bbox=_to_page_pt_box(box, img_size, page_box),
                                kind="table",
                                source="layout_table",
                                confidence=score,
                            )
                            for box, score in raw_tables
                        )
                    results[pno] = detected_boxes

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
