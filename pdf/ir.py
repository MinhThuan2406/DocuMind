from __future__ import annotations

from pathlib import Path
from typing import Any

import pdfplumber

from ._pdf_io import PdfContext, extract_toc, extract_words_raw, page_bbox
from .assemble import AssembledItem, assemble_page, tag_running_headers_footers
from .backends import DiagramExtractor, FigureGrouper, LayoutBackend, PageReader, TableExtractor, get as get_backend
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


def _collect_stats_warnings(backend: Any, prev_stats: Any, warnings: list[dict[str, Any]]) -> None:
    stats = getattr(backend, "last_stats", None)
    if stats is not None and stats is not prev_stats:
        for w in getattr(stats, "warnings", ()):
            warnings.append(dict(w))


def _link_diagram_text_elements(
    diag_dict: dict[str, Any],
    fig_elem_id: str,
    page_elems: list[dict[str, Any]],
) -> None:
    nodes = diag_dict.get("nodes") or ()
    if not nodes:
        return
    children = [
        e
        for e in page_elems
        if e.get("level") == 2 and e.get("parent_id") == fig_elem_id and e.get("type") == "text"
    ]
    if not children:
        return
    node_boxes = []
    for node in nodes:
        nb = node.get("bbox")
        if nb and len(nb) == 4:
            area = max(0.0, nb[2] - nb[0]) * max(0.0, nb[3] - nb[1])
            node_boxes.append((area, node))
    if not node_boxes:
        return
    for child in children:
        tb = child["bbox"]
        cx = (tb[0] + tb[2]) * 0.5
        cy = (tb[1] + tb[3]) * 0.5
        best_node: dict[str, Any] | None = None
        best_area: float | None = None
        for area, node in node_boxes:
            nb = node["bbox"]
            if nb[0] - 1.5 <= cx <= nb[2] + 1.5 and nb[1] - 1.5 <= cy <= nb[3] + 1.5:
                if best_area is None or area < best_area:
                    best_area = area
                    best_node = node
        if best_node is not None:
            ids = list(best_node.get("text_element_ids") or ())
            if child["id"] not in ids:
                ids.append(child["id"])
            best_node["text_element_ids"] = ids


def extract_ir(
    path: str | Path,
    *,
    figures: bool = True,
    ocr: bool = True,
    tables: bool = False,
    diagrams: bool = False,
    layout: str | LayoutBackend = "heron-onnx",
    grouper: str | FigureGrouper = "v1_anchor",
    reader: str | PageReader = "composed",
    ocr_backend: str = "rapidocr-onnx",
    table_backend: str | TableExtractor = "auto",
    diagram_backend: str | DiagramExtractor = "vector_first",
    device: str = "auto",
    min_font_size: float = 0.0,
    structure: bool = True,
) -> dict[str, Any]:
    pdf_path = Path(path)
    source_file = pdf_path.name

    with PdfContext(pdf_path) as ctx:
        doc = ctx.plumber
        toc = extract_toc(pdf_path, pdfium_doc=ctx.pdfium) if (structure and doc.pages) else ()
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
                warnings.append({
                    "page": idx,
                    "code": "suspect_ocr_layer",
                    "severity": "warning",
                    "reason": "suspect_ocr_layer",
                    "fallback": "ocr" if ocr else None,
                })
            if prof.kind == "raster" and not ocr:
                unreadable_pages.append({"page": idx, "reason": "raster_page_ocr_disabled"})

        figures_by_page: dict[int, list[FigureBox]] = {}
        candidate_pages = [
            p.page for p in profiles if not _uses_ocr_fallback(p, ocr) and p.kind != "raster" and p.figure_candidate
        ]
        if figures and candidate_pages:
            from .figures import detect_figures

            figures_by_page = detect_figures(
                pdf_path,
                candidate_pages,
                layout=layout,
                grouper=grouper,
                device=device,
                pdfium_doc=ctx.pdfium,
                pdf_hash=ctx.pdf_hash,
                warnings_out=warnings,
            )

        raster_by_page: dict[int, RasterPage] = {}
        if ocr:
            raster_pages = [p.page for p in profiles if _uses_ocr_fallback(p, ocr)]
            if raster_pages:
                from .raster import read_raster_pages

                resolved_reader = (
                    get_backend(
                        "reader",
                        reader,
                        layout=layout if isinstance(layout, str) else "heron-onnx",
                        ocr_backend=ocr_backend,
                        device=device,
                    )
                    if isinstance(reader, str)
                    else reader
                )
                prev_reader_stats = getattr(resolved_reader, "last_stats", None)
                raster_by_page = read_raster_pages(
                    pdf_path,
                    raster_pages,
                    reader=resolved_reader,
                    layout=layout if isinstance(layout, str) else "heron-onnx",
                    ocr_backend=ocr_backend,
                    device=device,
                    pdfium_doc=ctx.pdfium,
                )
                _collect_stats_warnings(resolved_reader, prev_reader_stats, warnings)

        elements: list[dict[str, Any]] = []
        extracted_tables: list[dict[str, Any]] = []
        extracted_diagrams: list[dict[str, Any]] = []
        elem_counter = 0
        tbl_counter = 0
        diag_counter = 0
        resolved_table_backend: TableExtractor | None = None
        resolved_diagram_backend: DiagramExtractor | None = None

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

                    if resolved_table_backend is None:
                        resolved_table_backend = (
                            get_backend("table", table_backend, device=device)
                            if isinstance(table_backend, str)
                            else table_backend
                        )
                    prev_tbl_stats = getattr(resolved_table_backend, "last_stats", None)
                    parsed_tables = extract_tables_on_page(
                        page,
                        [tuple(e["bbox"]) for e in tbl_elems],
                        words=words,
                        is_raster=use_ocr,
                        extractor=resolved_table_backend,
                        ocr_backend=ocr_backend,
                        device=device,
                    )
                    _collect_stats_warnings(resolved_table_backend, prev_tbl_stats, warnings)
                    for tbl_elem, parsed in zip(tbl_elems, parsed_tables):
                        if parsed.num_rows == 0 or parsed.num_cols == 0:
                            warnings.append({
                                "page": prof.page,
                                "code": "table_parse_empty",
                                "severity": "warning",
                                "reason": "table_extractor_returned_empty_grid",
                                "fallback": None,
                            })
                            continue
                        tbl_counter += 1
                        tbl_id = f"tbl_{tbl_counter:04d}"
                        tbl_elem["metadata"]["table_id"] = tbl_id
                        tbl_elem["text"] = f"[Table {tbl_id}: {parsed.num_rows}x{parsed.num_cols}]"
                        extracted_tables.append(table_to_ir_dict(tbl_id, tbl_elem["id"], prof.page, source_file, parsed))

            if diagrams:
                fig_elems = [e for e in page_elems if e["type"] == "figure" and e["level"] == 1]
                if fig_elems:
                    from .diagrams import diagram_to_ir_dict, extract_diagrams_on_page

                    if resolved_diagram_backend is None:
                        resolved_diagram_backend = (
                            get_backend("diagram", diagram_backend, device=device)
                            if isinstance(diagram_backend, str)
                            else diagram_backend
                        )
                    prev_diag_stats = getattr(resolved_diagram_backend, "last_stats", None)
                    parsed_diagrams = extract_diagrams_on_page(
                        page,
                        [tuple(e["bbox"]) for e in fig_elems],
                        words=words,
                        is_raster=use_ocr,
                        extractor=resolved_diagram_backend,
                        ocr_backend=ocr_backend,
                        device=device,
                    )
                    _collect_stats_warnings(resolved_diagram_backend, prev_diag_stats, warnings)
                    for fig_elem, parsed in zip(fig_elems, parsed_diagrams):
                        if not (parsed.nodes or parsed.edges or parsed.groups):
                            warnings.append({
                                "page": prof.page,
                                "code": "diagram_parse_empty",
                                "severity": "warning",
                                "reason": "diagram_extractor_returned_empty_graph",
                                "fallback": None,
                            })
                            continue
                        diag_counter += 1
                        diag_id = f"diag_{diag_counter:04d}"
                        fig_elem["metadata"]["diagram_id"] = diag_id
                        diag_dict = diagram_to_ir_dict(diag_id, fig_elem["id"], prof.page, source_file, parsed)
                        _link_diagram_text_elements(diag_dict, fig_elem["id"], page_elems)
                        extracted_diagrams.append(diag_dict)

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
        "diagrams": extracted_diagrams,
    }
    if structure:
        apply_structure(ir, toc=toc, inplace=True)
    return ir
