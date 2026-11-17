from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image

from .layout import Cluster, get_layout_detector, postprocess_layout
from .models import BBox, FigureBox, RasterPage, TextBlock
from .ocr import OcrLine
from .text import xy_cut_order

DEFAULT_RASTER_DPI = 150
LAYOUT_BOX_CONFIDENCE = 0.95

_OCR_CACHE: dict[str, Any] = {}


def _get_recognizer(backend: str = "rapidocr") -> Any:
    if backend not in _OCR_CACHE:
        if backend == "rapidocr":
            from .ocr.rapidocr import RapidOcrRecognizer

            _OCR_CACHE[backend] = RapidOcrRecognizer()
        else:
            raise ValueError(f"unknown ocr backend: {backend!r}")
    return _OCR_CACHE[backend]


def _area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter_area(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def _scale_box(box: BBox, sx: float, sy: float) -> BBox:
    return (
        round(box[0] * sx, 3),
        round(box[1] * sy, 3),
        round(box[2] * sx, 3),
        round(box[3] * sy, 3),
    )


def _join_ocr_lines(lines: list[OcrLine]) -> str:
    ordered = sorted(lines, key=lambda ln: (round((ln.bbox[1] + ln.bbox[3]) * 0.5 / 8.0), ln.bbox[0]))
    return " ".join(ln.text for ln in ordered if ln.text).strip()


def _assign_lines_to_clusters(
    lines: list[OcrLine],
    clusters: list[Cluster],
) -> tuple[dict[int, list[OcrLine]], list[OcrLine]]:
    picture_boxes = [c.bbox for c in clusters if c.label == "picture"]
    text_targets = [c for c in clusters if c.label != "picture"]
    by_cluster: dict[int, list[OcrLine]] = defaultdict(list)
    orphans: list[OcrLine] = []

    for line in lines:
        l_area = _area(line.bbox)
        best_cid: int | None = None
        best_frac = 0.2
        if l_area > 0:
            for c in text_targets:
                frac = _inter_area(line.bbox, c.bbox) / l_area
                if frac > best_frac:
                    best_frac = frac
                    best_cid = c.id
        if best_cid is not None:
            by_cluster[best_cid].append(line)
            continue

        cx = (line.bbox[0] + line.bbox[2]) * 0.5
        cy = (line.bbox[1] + line.bbox[3]) * 0.5
        in_picture = any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in picture_boxes)
        if not in_picture:
            orphans.append(line)

    return by_cluster, orphans


def read_raster_page(
    pdf_path: str | Path,
    page: int,
    *,
    raster_backend: str = "default",
    layout: str = "heron",
    ocr_backend: str = "rapidocr",
    device: str = "auto",
    dpi: int = DEFAULT_RASTER_DPI,
) -> RasterPage:
    if raster_backend == "mineru-pipeline":
        return read_raster_pages(
            pdf_path,
            [page],
            raster_backend=raster_backend,
            layout=layout,
            ocr_backend=ocr_backend,
            device=device,
            dpi=dpi,
        )[page]
    if raster_backend not in ("default", "heron_rapidocr"):
        raise ValueError(f"unknown raster_backend: {raster_backend!r}")

    doc_pdf = pymupdf.open(str(pdf_path))
    try:
        if not (1 <= page <= len(doc_pdf)):
            raise ValueError(f"page {page} out of range 1..{len(doc_pdf)}")
        pdf_page = doc_pdf[page - 1]
        page_w_pt = float(pdf_page.rect.width)
        page_h_pt = float(pdf_page.rect.height)
        zoom = dpi / 72.0
        pix = pdf_page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc_pdf.close()

    iw, ih = image.size
    sx = page_w_pt / iw
    sy = page_h_pt / ih

    detector, _ = get_layout_detector(layout, device)
    raw_items = detector.detect(image)
    clusters = postprocess_layout(raw_items, (float(iw), float(ih)), keep_empty_clusters=True)

    recognizer = _get_recognizer(ocr_backend)
    ocr_lines = recognizer.recognize(image)

    by_cluster, orphans = _assign_lines_to_clusters(ocr_lines, clusters)
    blocks: list[TextBlock] = []
    figures: list[FigureBox] = []

    for c in clusters:
        if c.label == "picture":
            figures.append(
                FigureBox(
                    bbox=_scale_box(c.bbox, sx, sy),
                    kind="figure",
                    source="layout_picture",
                    confidence=LAYOUT_BOX_CONFIDENCE,
                )
            )
            continue
        if c.label == "table":
            figures.append(
                FigureBox(
                    bbox=_scale_box(c.bbox, sx, sy),
                    kind="table",
                    source="layout_table",
                    confidence=LAYOUT_BOX_CONFIDENCE,
                )
            )
        cluster_lines = by_cluster.get(c.id)
        if not cluster_lines:
            continue
        text = _join_ocr_lines(cluster_lines)
        if not text:
            continue
        mean_conf = round(sum(ln.confidence for ln in cluster_lines) / len(cluster_lines), 4)
        ux0 = min(ln.bbox[0] for ln in cluster_lines)
        uy0 = min(ln.bbox[1] for ln in cluster_lines)
        ux1 = max(ln.bbox[2] for ln in cluster_lines)
        uy1 = max(ln.bbox[3] for ln in cluster_lines)
        blocks.append(
            TextBlock(
                text=text,
                bbox=_scale_box((ux0, uy0, ux1, uy1), sx, sy),
                font_size=0.0,
                bold=False,
                source="ocr",
                confidence=mean_conf,
                words=(),
                label=c.label,
            )
        )

    for ln in orphans:
        blocks.append(
            TextBlock(
                text=ln.text,
                bbox=_scale_box(ln.bbox, sx, sy),
                font_size=0.0,
                bold=False,
                source="ocr",
                confidence=ln.confidence,
                words=(),
                label="text",
            )
        )

    ordered_blocks = tuple(xy_cut_order(blocks, lambda b: b.bbox))
    return RasterPage(
        blocks=ordered_blocks,
        figures=tuple(figures),
        reading_order=tuple(range(len(ordered_blocks))),
    )


def read_raster_pages(
    pdf_path: str | Path,
    pages: Sequence[int],
    *,
    raster_backend: str = "default",
    layout: str = "heron",
    ocr_backend: str = "rapidocr",
    device: str = "auto",
    dpi: int = DEFAULT_RASTER_DPI,
) -> dict[int, RasterPage]:
    page_list = list(pages)
    if not page_list:
        return {}
    if raster_backend == "mineru-pipeline":
        from .layout.mineru_pipeline import mineru_middle_json_to_raster_pages, run_mineru_pdf

        doc_pdf = pymupdf.open(str(pdf_path))
        try:
            page_sizes = {
                idx: (float(p.rect.width), float(p.rect.height))
                for idx, p in enumerate(doc_pdf, start=1)
            }
        finally:
            doc_pdf.close()
        for pno in page_list:
            if pno not in page_sizes:
                raise ValueError(f"page {pno} out of range 1..{len(page_sizes)}")
        middle_json, _ = run_mineru_pdf(pdf_path, device=device, strip_content=False, ocr_mode="auto")
        return mineru_middle_json_to_raster_pages(middle_json, page_sizes, target_pages=page_list)
    if raster_backend not in ("default", "heron_rapidocr"):
        raise ValueError(f"unknown raster_backend: {raster_backend!r}")
    return {
        pno: read_raster_page(
            pdf_path,
            pno,
            raster_backend=raster_backend,
            layout=layout,
            ocr_backend=ocr_backend,
            device=device,
            dpi=dpi,
        )
        for pno in page_list
    }
