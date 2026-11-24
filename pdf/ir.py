from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import pdfplumber

from ._pdf_io import PdfContext, bbox_area, bbox_inter_area, extract_toc, extract_words_raw, page_bbox, round_bbox
from .assemble import (
    AssembledItem,
    assemble_page,
    extract_plain_text,
    merge_hybrid_region_ocr,
    tag_running_headers_footers,
)
from .backends import DiagramExtractor, FigureGrouper, LayoutBackend, PageReader, TableExtractor, get as get_backend
from .hierarchy import (
    _project_semantic_regions,
    _weighted_median_font_size,
    apply_structure,
    soft_merge_semantic_elements,
)
from .models import BBox, FigureBox, PageProfile, RasterPage, SemanticRegion, TextBlock
from .router import classify_page
from .text import extract_horiz_rules, extract_words, infer_medium_bold_families, xy_cut_order


def _element_dict(
    elem_id: str,
    page_no: int,
    reading_order: int,
    source_file: str,
    item: AssembledItem,
    parent_id: str | None,
) -> dict[str, Any]:
    bbox = [v + 0.0 for v in round_bbox(item.bbox)]
    meta: dict[str, Any] = {}
    if item.font_size is not None:
        meta["font_size"] = item.font_size
    if item.bold is not None:
        meta["bold"] = item.bold
    if item.label is not None:
        meta["layout_label"] = item.label
    if item.heading_clean_text is not None:
        meta["heading_clean_text"] = item.heading_clean_text
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


def _extract_small_vector_boxes(page: Any, bounds: BBox) -> tuple[BBox, ...]:
    pw = max(0.0, bounds[2] - bounds[0])
    ph = max(0.0, bounds[3] - bounds[1])
    boxes: list[BBox] = []
    for obj in getattr(page, "rects", ()):
        if obj.get("stroke") is False and obj.get("fill") is False:
            continue
        rx0, ry0, rx1, ry1 = float(obj["x0"]), float(obj["top"]), float(obj["x1"]), float(obj["bottom"])
        if rx0 > rx1:
            rx0, rx1 = rx1, rx0
        if ry0 > ry1:
            ry0, ry1 = ry1, ry0
        w, h = rx1 - rx0, ry1 - ry0
        if w <= 1.0 or h <= 1.0 or (pw > 0.0 and w >= 0.8 * pw) or (ph > 0.0 and h >= 0.3 * ph):
            continue
        if h <= 60.0 and w <= 300.0:
            boxes.append((rx0, ry0, rx1, ry1))
    for obj in getattr(page, "curves", ()):
        if obj.get("stroke") is False and obj.get("fill") is False:
            continue
        pts = obj.get("pts") or ()
        if not obj.get("fill"):
            if len(pts) < 4 or abs(float(pts[0][0]) - float(pts[-1][0])) > 2.0 or abs(float(pts[0][1]) - float(pts[-1][1])) > 2.0:
                continue
        rx0, ry0, rx1, ry1 = float(obj["x0"]), float(obj["top"]), float(obj["x1"]), float(obj["bottom"])
        if rx0 > rx1:
            rx0, rx1 = rx1, rx0
        if ry0 > ry1:
            ry0, ry1 = ry1, ry0
        w, h = rx1 - rx0, ry1 - ry0
        if 1.0 < w <= 300.0 and 1.0 < h <= 60.0 and (pw <= 0.0 or w < 0.8 * pw):
            boxes.append((rx0, ry0, rx1, ry1))
    return tuple(boxes)


def _uses_ocr_fallback(prof: PageProfile, ocr: bool) -> bool:
    return ocr and (prof.kind == "raster" or (prof.kind == "hybrid" and prof.suspect_ocr_layer))


def _collect_stats_warnings(backend: Any, prev_stats: Any, warnings: list[dict[str, Any]]) -> None:
    stats = getattr(backend, "last_stats", None)
    if stats is not None and stats is not prev_stats:
        for w in getattr(stats, "warnings", ()):
            warnings.append(dict(w))


def _merge_assembled_with_region_ocr(
    assembled: list[AssembledItem],
    ocr_blocks: Sequence[TextBlock],
) -> list[AssembledItem]:
    if not ocr_blocks:
        return assembled

    groups: list[tuple[AssembledItem, list[AssembledItem]]] = []
    index_to_group: dict[int, int] = {}
    for idx, item in enumerate(assembled):
        if item.level == 1:
            index_to_group[idx] = len(groups)
            groups.append((item, []))
        elif item.parent_index is not None and item.parent_index in index_to_group:
            groups[index_to_group[item.parent_index]][1].append(item)

    box_groups = [
        (bbox_area(top.bbox), g_idx, top.bbox)
        for g_idx, (top, _) in enumerate(groups)
        if top.kind in ("figure", "table")
    ]
    dirty_groups: set[int] = set()

    for blk in ocr_blocks:
        b_area = bbox_area(blk.bbox)
        best_g_idx: int | None = None
        best_key: tuple[float, float, int] | None = None
        if b_area <= 0.0:
            cx = (blk.bbox[0] + blk.bbox[2]) * 0.5
            cy = (blk.bbox[1] + blk.bbox[3]) * 0.5
            for area, g_idx, box in box_groups:
                if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
                    key = (area, -1.0, g_idx)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_g_idx = g_idx
        else:
            for area, g_idx, box in box_groups:
                ioa = bbox_inter_area(blk.bbox, box) / b_area
                if ioa >= 0.50:
                    key = (area, -ioa, g_idx)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_g_idx = g_idx

        if best_g_idx is not None:
            groups[best_g_idx][1].append(
                AssembledItem(
                    kind="text",
                    bbox=blk.bbox,
                    text=blk.text,
                    source=blk.source,
                    confidence=blk.confidence,
                    level=2,
                    parent_index=None,
                    font_size=blk.font_size,
                    bold=blk.bold,
                    label=blk.label,
                )
            )
            dirty_groups.add(best_g_idx)
        else:
            groups.append((
                AssembledItem(
                    kind="text",
                    bbox=blk.bbox,
                    text=blk.text,
                    source=blk.source,
                    confidence=blk.confidence,
                    level=1,
                    parent_index=None,
                    font_size=blk.font_size,
                    bold=blk.bold,
                    label=blk.label,
                ),
                [],
            ))

    for g_idx in dirty_groups:
        top, children = groups[g_idx]
        groups[g_idx] = (top, xy_cut_order(children, lambda c: c.bbox))

    ordered_groups = xy_cut_order(groups, lambda g: g[0].bbox)
    out: list[AssembledItem] = []
    for top, children in ordered_groups:
        parent_idx = len(out)
        out.append(top)
        for child in children:
            out.append(replace(child, parent_index=parent_idx))
    return out


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
            node_boxes.append((bbox_area(nb), node))
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


def _locate_formula_targets(
    page_elems: Sequence[dict[str, Any]],
    regions: Sequence[SemanticRegion],
) -> list[tuple[dict[str, Any], BBox]]:
    targets: list[tuple[dict[str, Any], BBox]] = []
    seen_ids: set[str] = set()
    reg_matches: list[tuple[dict[str, Any], BBox, int]] = []
    reg_counts: dict[int, int] = {}

    for elem in page_elems:
        meta = elem.setdefault("metadata", {})
        eb = elem.get("bbox")
        if not eb or len(eb) != 4:
            continue
        eb_box: BBox = (float(eb[0]), float(eb[1]), float(eb[2]), float(eb[3]))
        if elem.get("type") == "formula" or meta.get("layout_label") == "formula":
            targets.append((elem, eb_box))
            seen_ids.add(str(elem["id"]))

    if regions:
        for elem in page_elems:
            eid = str(elem["id"])
            if eid in seen_ids:
                continue
            if elem.get("type") not in ("text", "formula") or elem.get("level", 1) != 1:
                continue
            if elem.get("source", "text_layer") != "text_layer":
                continue
            meta = elem.setdefault("metadata", {})
            if meta.get("layout_label") is not None:
                continue
            eb = elem.get("bbox")
            if not eb or len(eb) != 4:
                continue
            ex0, ey0, ex1, ey1 = float(eb[0]), float(eb[1]), float(eb[2]), float(eb[3])
            elem_area = max(0.0, ex1 - ex0) * max(0.0, ey1 - ey0)
            if elem_area <= 0.0:
                continue

            best_score: tuple[float, float] | None = None
            best_idx: int | None = None
            best_label: str | None = None
            best_rb: Sequence[float] | None = None
            for r_idx, reg in enumerate(regions):
                if isinstance(reg, Mapping):
                    rb = reg.get("bbox")
                    label = reg.get("label")
                    conf = float(reg.get("confidence", 1.0))
                else:
                    rb = getattr(reg, "bbox", None)
                    label = getattr(reg, "label", None)
                    conf = float(getattr(reg, "confidence", 1.0))
                if not rb or len(rb) != 4 or not label:
                    continue
                ix = max(0.0, min(ex1, float(rb[2])) - max(ex0, float(rb[0])))
                iy = max(0.0, min(ey1, float(rb[3])) - max(ey0, float(rb[1])))
                ioa = (ix * iy) / elem_area
                if ioa >= 0.60:
                    score = (ioa, conf)
                    if best_score is None or score > best_score:
                        best_score = score
                        best_idx = r_idx
                        best_label = str(label).lower()
                        best_rb = rb

            if best_idx is None or best_label != "formula" or best_rb is None:
                continue
            meta["layout_label"] = "formula"
            union_box: BBox = (
                min(ex0, float(best_rb[0])),
                min(ey0, float(best_rb[1])),
                max(ex1, float(best_rb[2])),
                max(ey1, float(best_rb[3])),
            )
            reg_matches.append((elem, union_box, best_idx))
            reg_counts[best_idx] = reg_counts.get(best_idx, 0) + 1
            seen_ids.add(eid)

    for elem, union_box, r_idx in reg_matches:
        eb = elem["bbox"]
        crop_box = (
            union_box
            if reg_counts.get(r_idx, 0) == 1
            else (float(eb[0]), float(eb[1]), float(eb[2]), float(eb[3]))
        )
        targets.append((elem, crop_box))

    return targets


def extract_ir(
    path: str | Path,
    *,
    figures: bool = True,
    ocr: bool = True,
    tables: bool = False,
    diagrams: bool = False,
    formulas: bool = False,
    layout: str | LayoutBackend = "heron-onnx",
    grouper: str | FigureGrouper = "v1_anchor",
    reader: str | PageReader = "composed",
    ocr_backend: str = "rapidocr-onnx",
    ocr_lang: str | None = None,
    table_backend: str | TableExtractor = "auto",
    diagram_backend: str | DiagramExtractor = "vector_first",
    formula_backend: str | Any = "auto",
    device: str = "auto",
    min_font_size: float = 0.0,
    structure: bool = True,
    semantic_layout: Literal["auto", "always", "off"] = "auto",
) -> dict[str, Any]:
    if semantic_layout not in ("auto", "always", "off"):
        raise ValueError(f"invalid semantic_layout: {semantic_layout!r}; expected 'auto', 'always', or 'off'")
    pdf_path = Path(path)
    source_file = pdf_path.name

    with PdfContext(pdf_path) as ctx:
        doc = ctx.plumber
        toc = extract_toc(pdf_path, pdfium_doc=ctx.pdfium) if (structure and doc.pages) else ()
        profiles: list[PageProfile] = []
        raw_words_by_page: dict[int, list[dict[str, Any]]] = {}
        page_boxes: dict[int, BBox] = {}
        page_sizes: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        unreadable_pages: list[dict[str, Any]] = []

        for idx, page in enumerate(doc.pages, start=1):
            page_boxes[idx] = page_bbox(page)
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
                "table_candidate": prof.table_candidate,
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
        semantic_regions: dict[int, list[SemanticRegion]] = {}
        non_raster_pages = [
            p.page
            for p in profiles
            if not _uses_ocr_fallback(p, ocr) and p.kind != "raster"
        ]
        if semantic_layout == "always":
            candidate_pages = non_raster_pages if (figures or tables or formulas or structure) else []
        elif semantic_layout == "off":
            candidate_pages = [
                p.page
                for p in profiles
                if not _uses_ocr_fallback(p, ocr)
                and p.kind != "raster"
                and ((figures and p.figure_candidate) or (tables and p.table_candidate))
            ]
        else:
            candidate_pages = [
                p.page
                for p in profiles
                if not _uses_ocr_fallback(p, ocr)
                and p.kind != "raster"
                and ((figures and p.figure_candidate) or (tables and p.table_candidate) or formulas)
            ]
        if candidate_pages:
            from .figures import detect_figures

            detect_kwargs: dict[str, Any] = {
                "layout": layout,
                "grouper": grouper,
                "device": device,
                "pdfium_doc": ctx.pdfium,
                "pdf_hash": ctx.pdf_hash,
                "warnings_out": warnings,
                "semantic_regions_out": semantic_regions if (semantic_layout != "off" and (structure or formulas)) else None,
                "page_boxes": {p: page_boxes[p] for p in candidate_pages},
            }
            if semantic_layout == "always":
                detect_kwargs["figure_pages"] = {p.page for p in profiles if figures and p.figure_candidate}
                detect_kwargs["raw_picture_pages"] = {
                    p.page
                    for p in profiles
                    if figures and not p.figure_candidate and not _uses_ocr_fallback(p, ocr) and p.kind != "raster"
                }
                detect_kwargs["table_pages"] = {
                    p.page for p in profiles if p.table_candidate or (figures and p.figure_candidate)
                }

            figures_by_page = detect_figures(
                pdf_path,
                candidate_pages,
                **detect_kwargs,
            )
            if not figures and not tables:
                figures_by_page = {}
            elif not figures:
                figures_by_page = {
                    pno: [box for box in boxes if box.kind == "table"]
                    for pno, boxes in figures_by_page.items()
                }

        raster_by_page: dict[int, RasterPage] = {}
        region_ocr_by_page: dict[int, list[TextBlock]] = {}
        if ocr:
            raster_pages = [p.page for p in profiles if _uses_ocr_fallback(p, ocr)]
            partial_regions_by_page = {
                p.page: p.partial_raster_regions
                for p in profiles
                if not _uses_ocr_fallback(p, ocr) and p.kind == "hybrid" and p.partial_raster_regions
            }
            if raster_pages or partial_regions_by_page:
                resolved_layout = layout if isinstance(layout, str) else "heron-onnx"
                resolved_reader = (
                    get_backend(
                        "reader",
                        reader,
                        layout=resolved_layout,
                        ocr_backend=ocr_backend,
                        device=device,
                        **({"ocr_lang": ocr_lang} if ocr_lang is not None else {}),
                    )
                    if isinstance(reader, str)
                    else reader
                )
                if raster_pages:
                    from .raster import read_raster_pages

                    prev_reader_stats = getattr(resolved_reader, "last_stats", None)
                    reader_params = inspect.signature(resolved_reader.read).parameters
                    supports_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in reader_params.values())
                    if tables and ("enable_table_tiling" in reader_params or supports_var_kw):
                        read_kwargs: dict[str, Any] = {"enable_table_tiling": bool(tables)}
                        if "pdfium_doc" in reader_params or supports_var_kw:
                            read_kwargs["pdfium_doc"] = ctx.pdfium
                        if "page_boxes" in reader_params or supports_var_kw:
                            read_kwargs["page_boxes"] = {p: page_boxes[p] for p in raster_pages}
                        raster_by_page = resolved_reader.read(pdf_path, raster_pages, **read_kwargs)
                    else:
                        raster_by_page = read_raster_pages(
                            pdf_path,
                            raster_pages,
                            reader=resolved_reader,
                            layout=resolved_layout,
                            ocr_backend=ocr_backend,
                            ocr_lang=ocr_lang,
                            device=device,
                            pdfium_doc=ctx.pdfium,
                            page_boxes={p: page_boxes[p] for p in raster_pages},
                        )
                    _collect_stats_warnings(resolved_reader, prev_reader_stats, warnings)
                if partial_regions_by_page:
                    from .raster import read_raster_regions

                    prev_reader_stats = getattr(resolved_reader, "last_stats", None)
                    try:
                        region_ocr_by_page = read_raster_regions(
                            pdf_path,
                            partial_regions_by_page,
                            reader=resolved_reader,
                            layout=resolved_layout,
                            ocr_backend=ocr_backend,
                            ocr_lang=ocr_lang,
                            device=device,
                            pdfium_doc=ctx.pdfium,
                            page_boxes={p: page_boxes[p] for p in partial_regions_by_page},
                        )
                        _collect_stats_warnings(resolved_reader, prev_reader_stats, warnings)
                    except NotImplementedError:
                        region_ocr_by_page = {}

        elements: list[dict[str, Any]] = []
        extracted_tables: list[dict[str, Any]] = []
        extracted_diagrams: list[dict[str, Any]] = []
        extracted_formulas: list[dict[str, Any]] = []
        elem_counter = 0
        tbl_counter = 0
        diag_counter = 0
        fml_counter = 0
        resolved_table_backend: TableExtractor | None = None
        resolved_diagram_backend: DiagramExtractor | None = None
        resolved_formula_backend: Any = None

        medium_bold_families = infer_medium_bold_families(
            [w for p_words in raw_words_by_page.values() for w in p_words]
        )
        horiz_rules_by_page: dict[int, tuple[BBox, ...]] = {}
        for prof in profiles:
            page = doc.pages[prof.page - 1]
            use_ocr = _uses_ocr_fallback(prof, ocr) or prof.kind == "raster"
            words = (
                []
                if use_ocr
                else extract_words(
                    page,
                    min_font_size=min_font_size,
                    raw_words=raw_words_by_page[prof.page],
                    medium_bold_families=medium_bold_families,
                )
            )
            region_ocr_blocks = region_ocr_by_page.get(prof.page, ())
            kept_ocr: list[TextBlock] = []
            if region_ocr_blocks:
                words, kept_ocr = merge_hybrid_region_ocr(words, region_ocr_blocks)
            horiz_rules = () if use_ocr else extract_horiz_rules(page, page_boxes[prof.page])
            if horiz_rules:
                horiz_rules_by_page[prof.page] = horiz_rules
            vector_boxes = () if use_ocr else _extract_small_vector_boxes(page, page_boxes[prof.page])
            assembled = assemble_page(
                prof,
                words,
                page_boxes[prof.page],
                figures_by_page.get(prof.page, ()),
                raster_by_page.get(prof.page),
                horiz_rules=horiz_rules,
                vector_boxes=vector_boxes,
            )
            if kept_ocr:
                assembled = _merge_assembled_with_region_ocr(assembled, kept_ocr)
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
                        ocr_lang=ocr_lang,
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
                        tbl_dict = table_to_ir_dict(tbl_id, tbl_elem["id"], prof.page, source_file, parsed)
                        extracted_tables.append(tbl_dict)

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
                        ocr_lang=ocr_lang,
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

            if formulas:
                if semantic_layout != "off" and semantic_regions.get(prof.page):
                    soft_merge_semantic_elements(
                        page_elems,
                        {prof.page: semantic_regions[prof.page]},
                        labels=("formula",),
                    )
                formula_targets = _locate_formula_targets(page_elems, semantic_regions.get(prof.page, ()))
                if formula_targets:
                    from .formulas import extract_formulas_on_page, formula_to_ir_dict

                    if resolved_formula_backend is None:
                        resolved_formula_backend = (
                            get_backend("formula", formula_backend, device=device)
                            if isinstance(formula_backend, str)
                            else formula_backend
                        )
                    prev_fml_stats = getattr(resolved_formula_backend, "last_stats", None)
                    parsed_formulas = extract_formulas_on_page(
                        ctx.pdfium[prof.page - 1],
                        [crop_box for _, crop_box in formula_targets],
                        extractor=resolved_formula_backend,
                        raster_page=raster_by_page.get(prof.page),
                        device=device,
                    )
                    _collect_stats_warnings(resolved_formula_backend, prev_fml_stats, warnings)
                    for (fml_elem, _), parsed in zip(formula_targets, parsed_formulas):
                        if not parsed.latex.strip():
                            warnings.append({
                                "page": prof.page,
                                "code": "formula_parse_empty",
                                "severity": "warning",
                                "reason": "formula_extractor_returned_empty_latex",
                                "fallback": None,
                            })
                            continue
                        fml_counter += 1
                        fml_id = f"fml_{fml_counter:04d}"
                        fml_elem["metadata"]["formula_id"] = fml_id
                        fml_elem["metadata"]["latex"] = parsed.latex
                        fml_dict = formula_to_ir_dict(
                            fml_id,
                            fml_elem["id"],
                            prof.page,
                            source_file,
                            tuple(fml_elem["bbox"]),
                            parsed,
                        )
                        extracted_formulas.append(fml_dict)

            if (
                any(float(e["confidence"]) < 0.70 for e in page_elems)
                or any(float(t["confidence"]) < 0.70 for t in extracted_tables if t["page"] == prof.page)
                or any(float(d["confidence"]) < 0.70 for d in extracted_diagrams if d["page"] == prof.page)
                or any(float(f["confidence"]) < 0.70 for f in extracted_formulas if f["page"] == prof.page)
            ):
                warnings.append({
                    "page": prof.page,
                    "code": "low_confidence",
                    "severity": "warning",
                    "reason": "confidence_below_0.70",
                    "fallback": None,
                })
            elements.extend(page_elems)

    page_heights = {p["number"]: float(p["height_pt"]) for p in page_sizes}
    if structure and semantic_layout != "off" and semantic_regions:
        _project_semantic_regions(
            elements,
            semantic_regions,
            page_heights=page_heights,
            body_font=_weighted_median_font_size(elements),
        )
    toc_titles = [title for _, title, _ in toc] if structure else ()
    tag_running_headers_footers(
        elements,
        page_heights,
        toc_titles=toc_titles,
        horiz_rules_by_page=horiz_rules_by_page,
    )
    if structure and semantic_layout != "off" and semantic_regions:
        soft_merge_semantic_elements(elements, semantic_regions)

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
        "formulas": extracted_formulas,
    }
    if structure:
        apply_structure(
            ir,
            toc=toc,
            semantic_regions=semantic_regions if semantic_layout != "off" else None,
            inplace=True,
        )
    if tables and ir["tables"]:
        from .tables import link_table_continuations

        ir["tables"] = link_table_continuations(ir["tables"], ir["elements"])
    return ir
