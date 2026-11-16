from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import pymupdf

from .models import BBox, FigureBox, RasterPage, TextBlock

DEFAULT_RASTER_DPI = 150
DEFAULT_OCR_CONFIDENCE = 0.85

_CONVERTER_CACHE: dict[tuple[int, str], Any] = {}


def _build_ocr_converter(device: str) -> Any:
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, ImageFormatOption

    opts = ImageFormatOption().pipeline_options
    opts.do_ocr = True
    opts.accelerator_options = AcceleratorOptions(device=device)
    return DocumentConverter(format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts)})


def _get_ocr_converter(device: str) -> Any:
    key = (id(_build_ocr_converter), device)
    if key not in _CONVERTER_CACHE:
        _CONVERTER_CACHE[key] = _build_ocr_converter(device)
    return _CONVERTER_CACHE[key]


def _extract_ocr_confidence(result: Any) -> float:
    conf = getattr(result, "confidence", None)
    if conf is None:
        return DEFAULT_OCR_CONFIDENCE
    pages = getattr(conf, "pages", None)
    if isinstance(pages, dict) and pages:
        first = pages.get(1) or next(iter(pages.values()))
        score = getattr(first, "ocr_score", None)
        if isinstance(score, (int, float)) and not math.isnan(score) and 0.0 < float(score) <= 1.0:
            return round(float(score), 4)
    for attr in ("mean_ocr_score", "ocr_score"):
        score = getattr(conf, attr, None)
        if isinstance(score, (int, float)) and not math.isnan(score) and 0.0 < float(score) <= 1.0:
            return round(float(score), 4)
    return DEFAULT_OCR_CONFIDENCE


def _to_pt_bbox(raw_bbox: Any, page_h_docling: float, sx: float, sy: float) -> BBox:
    b = raw_bbox.to_top_left_origin(page_h_docling)
    return (
        round(float(b.l) * sx, 3),
        round(float(b.t) * sy, 3),
        round(float(b.r) * sx, 3),
        round(float(b.b) * sy, 3),
    )


def read_raster_page(
    pdf_path: str | Path,
    page: int,
    *,
    device: str = "auto",
    dpi: int = DEFAULT_RASTER_DPI,
) -> RasterPage:
    doc_pdf = pymupdf.open(str(pdf_path))
    try:
        if not (1 <= page <= len(doc_pdf)):
            raise ValueError(f"page {page} out of range 1..{len(doc_pdf)}")
        pdf_page = doc_pdf[page - 1]
        page_w_pt = float(pdf_page.rect.width)
        page_h_pt = float(pdf_page.rect.height)
        zoom = dpi / 72.0
        pix = pdf_page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_png = Path(tmp_dir) / "page.png"
            pix.save(str(tmp_png))
            converter = _get_ocr_converter(device)
            result = converter.convert(str(tmp_png))
    finally:
        doc_pdf.close()

    docling_doc = result.document
    dl_page = docling_doc.pages[1]
    dl_w = float(dl_page.size.width) or 1.0
    dl_h = float(dl_page.size.height) or 1.0
    sx = page_w_pt / dl_w
    sy = page_h_pt / dl_h
    ocr_conf = _extract_ocr_confidence(result)

    blocks: list[TextBlock] = []
    figures: list[FigureBox] = []

    for it, _lvl in docling_doc.iterate_items():
        raw_label = getattr(it, "label", type(it).__name__)
        label = (raw_label.value if hasattr(raw_label, "value") else str(raw_label)).lower()
        prov = getattr(it, "prov", None)
        if not prov or not hasattr(prov[0], "bbox"):
            continue
        bbox = _to_pt_bbox(prov[0].bbox, dl_h, sx, sy)
        if label in ("picture", "figure"):
            figures.append(FigureBox(bbox=bbox, kind="figure", source="docling_picture", confidence=ocr_conf))
        elif label == "table":
            figures.append(FigureBox(bbox=bbox, kind="table", source="docling_table", confidence=ocr_conf))
        else:
            text = str(getattr(it, "text", "") or "").strip()
            if text:
                blocks.append(
                    TextBlock(
                        text=text,
                        bbox=bbox,
                        font_size=0.0,
                        bold=False,
                        source="ocr",
                        confidence=ocr_conf,
                        words=(),
                        label=label,
                    )
                )

    return RasterPage(
        blocks=tuple(blocks),
        figures=tuple(figures),
        reading_order=tuple(range(len(blocks))),
    )
