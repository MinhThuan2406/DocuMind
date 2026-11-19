from __future__ import annotations

from pathlib import Path
from typing import Any

import pdfplumber

from ._pdf_io import extract_toc, extract_words_raw, page_bbox
from .assemble import AssembledItem, assemble_page, tag_running_headers_footers
from .backends import FigureGrouper, LayoutBackend, PageReader, TableExtractor
from .hierarchy import apply_structure
from .models import BBox, FigureBox, PageProfile, RasterPage
from .router import classify_page
from .text import extract_words


def _round_bbox(bbox: BBox) -> list[float]:
    return [round(v, 3) + 0.0 for v in bbox]


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


def _uses_ocr_fallback(prof: PageProfile, ocr: bool) -> bool:
    return ocr and (prof.kind == "raster" or (prof.kind == "hybrid" and prof.suspect_ocr_layer))


def extract_ir(
    path: str | Path,
    *,
    figures: bool = True,
    ocr: bool = True,
    tables: bool = False,
    layout: str | LayoutBackend = "heron-onnx",
    grouper: str | FigureGrouper = "v1_anchor",
    reader: str | PageReader = "composed",
    ocr_backend: str = "rapidocr-onnx",
    table_backend: str | TableExtractor = "auto",
    device: str = "auto",
    min_font_size: float = 0.0,
    structure: bool = True,
) -> dict[str, Any]:
    pdf_path = Path(path)
    source_file = pdf_path.name
    toc = extract_toc(pdf_path) if structure else ()

    with pdfplumber.open(str(pdf_path)) as doc:
        profiles: list[PageProfile] = []
        raw_words_by_page: dict[int, list[dict[str, Any]]] = {}
        page_sizes: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        unreadable_pages: list[dict[str, Any]] = []

        for idx, page in enumerate(doc.pages, start=1):
            raw_words = extract_words_raw(page, with_style=True)
            raw_words_by_page[idx] = raw_words
            prof = classify_page(page, page_number=idx, raw_words=raw_words)
            profiles.append(prof)
            page_sizes.append({
                "number": idx,
                "kind": prof.kind,
                "width_pt": round(float(page.width), 3),
                "height_pt": round(float(page.height), 3),
                "suspect_ocr_layer": prof.suspect_ocr_layer,
                "figure_candidate": prof.figure_candidate,
            })
            if prof.kind == "hybrid" and prof.suspect_ocr_layer:
                warn: dict[str, Any] = {"page": idx, "reason": "suspect_ocr_layer"}
                if ocr:
                    warn["fallback"] = "ocr"
                warnings.append(warn)
            if prof.kind == "raster" and not ocr:
                unreadable_pages.append({"page": idx, "reason": "raster_page_ocr_disabled"})

        figures_by_page: dict[int, list[FigureBox]] = {}
        candidate_pages = [
            p.page for p in profiles if not _uses_ocr_fallback(p, ocr) and p.kind != "raster" and p.figure_candidate
        ]
        if figures and candidate_pages:
            from .figures import detect_figures

            figures_by_page = detect_figures(pdf_path, candidate_pages, layout=layout, grouper=grouper, device=device)

        raster_by_page: dict[int, RasterPage] = {}
        if ocr:
            raster_pages = [p.page for p in profiles if _uses_ocr_fallback(p, ocr)]
            if raster_pages:
                from .raster import read_raster_pages

                raster_by_page = read_raster_pages(
                    pdf_path,
                    raster_pages,
                    reader=reader,
                    layout=layout if isinstance(layout, str) else "heron-onnx",
                    ocr_backend=ocr_backend,
                    device=device,
                )

        elements: list[dict[str, Any]] = []
        extracted_tables: list[dict[str, Any]] = []
        elem_counter = 0
        tbl_counter = 0

        for prof in profiles:
            page = doc.pages[prof.page - 1]
            use_ocr = _uses_ocr_fallback(prof, ocr) or prof.kind == "raster"
            words = (
                []
                if use_ocr
                else extract_words(page, min_font_size=min_font_size, raw_words=raw_words_by_page[prof.page])
            )
            assembled = assemble_page(
                prof,
                words,
                page_bbox(page),
                figures_by_page.get(prof.page, ()),
                raster_by_page.get(prof.page),
            )
            page_elems: list[dict[str, Any]] = []
            page_elem_ids: list[str] = []
            for ro, item in enumerate(assembled):
                elem_counter += 1
                elem_id = f"elem_{elem_counter:04d}"
                page_elem_ids.append(elem_id)
                parent_id = None if item.parent_index is None else page_elem_ids[item.parent_index]
                page_elems.append(_element_dict(elem_id, prof.page, ro, source_file, item, parent_id))

            if tables:
                tbl_elems = [e for e in page_elems if e["type"] == "table" and e["level"] == 1]
                if tbl_elems:
                    from .tables import extract_tables_on_page, table_to_ir_dict

                    parsed_tables = extract_tables_on_page(
                        page,
                        [tuple(e["bbox"]) for e in tbl_elems],
                        words=words,
                        is_raster=use_ocr,
                        extractor=table_backend,
                        ocr_backend=ocr_backend,
                        device=device,
                    )
                    for tbl_elem, parsed in zip(tbl_elems, parsed_tables):
                        if parsed.num_rows == 0 or parsed.num_cols == 0:
                            continue
                        tbl_counter += 1
                        tbl_id = f"tbl_{tbl_counter:04d}"
                        tbl_elem["metadata"]["table_id"] = tbl_id
                        tbl_elem["text"] = f"[Table {tbl_id}: {parsed.num_rows}x{parsed.num_cols}]"
                        extracted_tables.append(table_to_ir_dict(tbl_id, tbl_elem["id"], prof.page, source_file, parsed))

            elements.extend(page_elems)

    tag_running_headers_footers(elements, {p["number"]: float(p["height_pt"]) for p in page_sizes})

    ir = {
        "doc_id": pdf_path.stem,
        "source_file": source_file,
        "doc_type": "pdf",
        "metadata": {
            "coord_unit": "pt",
            "parser": f"pdf (pdfplumber {pdfplumber.__version__})",
            "reading_order_status": "xy_cut",
            "pages": page_sizes,
            "unreadable_pages": unreadable_pages,
            "warnings": warnings,
        },
        "elements": elements,
        "tables": extracted_tables,
        "diagrams": [],
    }
    if structure:
        apply_structure(ir, toc=toc, inplace=True)
    return ir
