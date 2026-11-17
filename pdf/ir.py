from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf

from .assemble import AssembledItem, assemble_page
from .models import BBox, FigureBox, PageProfile, RasterPage
from .router import classify_page
from .text import extract_words


def _round_bbox(bbox: BBox) -> list[float]:
    return [round(v, 3) + 0.0 for v in bbox]


def _page_rect(page: pymupdf.Page) -> BBox:
    r = page.rect
    return (float(r.x0), float(r.y0), float(r.x1), float(r.y1))


def _element_dict(
    elem_id: str,
    page_no: int,
    reading_order: int,
    source_file: str,
    item: AssembledItem,
    parent_id: str | None,
) -> dict[str, Any]:
    bbox = _round_bbox(item.bbox)
    meta: dict[str, Any] = {}
    if item.font_size is not None:
        meta["font_size"] = item.font_size
    if item.bold is not None:
        meta["bold"] = item.bold
    if item.label is not None:
        meta["layout_label"] = item.label
    return {
        "id": elem_id,
        "type": item.kind,
        "page": page_no,
        "bbox": bbox,
        "reading_order": reading_order,
        "level": item.level,
        "parent_id": parent_id,
        "text": item.text,
        "source": item.source,
        "confidence": item.confidence,
        "source_ref": {"file": source_file, "page": page_no, "bbox": bbox},
        "metadata": meta,
    }


def extract_ir(
    path: str | Path,
    *,
    figures: bool = True,
    ocr: bool = True,
    layout: str = "heron",
    ocr_backend: str = "rapidocr",
    raster_backend: str = "default",
    device: str = "auto",
    min_font_size: float = 0.0,
) -> dict[str, Any]:
    pdf_path = Path(path)
    source_file = pdf_path.name
    doc = pymupdf.open(str(pdf_path))

    try:
        profiles: list[PageProfile] = []
        page_sizes: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        unreadable_pages: list[dict[str, Any]] = []

        for idx, page in enumerate(doc, start=1):
            prof = classify_page(page, page_number=idx)
            profiles.append(prof)
            page_sizes.append({
                "number": idx,
                "kind": prof.kind,
                "width_pt": round(float(page.rect.width), 3),
                "height_pt": round(float(page.rect.height), 3),
                "suspect_ocr_layer": prof.suspect_ocr_layer,
                "figure_candidate": prof.figure_candidate,
            })
            if prof.kind == "hybrid" and prof.suspect_ocr_layer:
                warnings.append({"page": idx, "reason": "suspect_ocr_layer"})
            if prof.kind == "raster" and not ocr:
                unreadable_pages.append({"page": idx, "reason": "raster_page_ocr_disabled"})

        figures_by_page: dict[int, list[FigureBox]] = {}
        candidate_pages = [p.page for p in profiles if p.kind != "raster" and p.figure_candidate]
        if figures and candidate_pages:
            from .figures import detect_figures

            figures_by_page = detect_figures(pdf_path, candidate_pages, layout=layout, device=device)

        raster_by_page: dict[int, RasterPage] = {}
        if ocr:
            raster_pages = [p.page for p in profiles if p.kind == "raster"]
            if raster_pages:
                from .raster import read_raster_pages

                raster_by_page = read_raster_pages(
                    pdf_path,
                    raster_pages,
                    raster_backend=raster_backend,
                    layout=layout,
                    ocr_backend=ocr_backend,
                    device=device,
                )

        elements: list[dict[str, Any]] = []
        elem_counter = 0

        for prof in profiles:
            page = doc[prof.page - 1]
            words = [] if prof.kind == "raster" else extract_words(page, min_font_size=min_font_size)
            assembled = assemble_page(
                prof,
                words,
                _page_rect(page),
                figures_by_page.get(prof.page, ()),
                raster_by_page.get(prof.page),
            )
            page_elem_ids: list[str] = []
            for ro, item in enumerate(assembled):
                elem_counter += 1
                elem_id = f"elem_{elem_counter:04d}"
                page_elem_ids.append(elem_id)
                parent_id = None if item.parent_index is None else page_elem_ids[item.parent_index]
                elements.append(_element_dict(elem_id, prof.page, ro, source_file, item, parent_id))
    finally:
        doc.close()

    return {
        "doc_id": pdf_path.stem,
        "source_file": source_file,
        "doc_type": "pdf",
        "metadata": {
            "coord_unit": "pt",
            "parser": f"pdf (pymupdf {pymupdf.__version__})",
            "reading_order_status": "xy_cut",
            "pages": page_sizes,
            "unreadable_pages": unreadable_pages,
            "warnings": warnings,
        },
        "elements": elements,
        "tables": [],
        "diagrams": [],
    }
