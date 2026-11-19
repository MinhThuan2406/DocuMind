from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PIL import Image

from ._pdf_io import crop_page_rgb_with_bbox, round_bbox
from .backends import TableExtractor, get as get_backend
from .backends.tatr_onnx import table_to_html
from .models import BBox, ExtractedTable, Word

DEFAULT_TABLE_DPI = 250
_CROP_PAD_PT = 4.0


def _round_bbox(bbox: BBox) -> list[float]:
    return [v + 0.0 for v in round_bbox(bbox)]


def crop_page_region(page: Any, bbox_pt: BBox, dpi: int = DEFAULT_TABLE_DPI) -> tuple[Image.Image, BBox]:
    return crop_page_rgb_with_bbox(page, bbox_pt, dpi, pad_pt=_CROP_PAD_PT)


def tokens_in_box(tokens: Sequence[tuple[str, BBox]], box: BBox) -> list[tuple[str, BBox]]:
    out: list[tuple[str, BBox]] = []
    for text, tb in tokens:
        cx = (tb[0] + tb[2]) * 0.5
        cy = (tb[1] + tb[3]) * 0.5
        if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
            out.append((text, tb))
    return out


def ocr_crop_tokens(ocr_rec: Any, crop: Image.Image, crop_box: BBox) -> list[tuple[str, BBox]]:
    cw, ch = crop.size
    sx = (crop_box[2] - crop_box[0]) / max(1, cw)
    sy = (crop_box[3] - crop_box[1]) / max(1, ch)
    return [
        (
            ln.text,
            (
                crop_box[0] + ln.bbox[0] * sx,
                crop_box[1] + ln.bbox[1] * sy,
                crop_box[0] + ln.bbox[2] * sx,
                crop_box[1] + ln.bbox[3] * sy,
            ),
        )
        for ln in ocr_rec.recognize(crop)
    ]


def table_to_ir_dict(
    tbl_id: str,
    elem_id: str,
    page_no: int,
    source_file: str,
    table: ExtractedTable,
) -> dict[str, Any]:
    bbox = _round_bbox(table.bbox)
    cells = []
    for c in table.cells:
        cell_dict: dict[str, Any] = {
            "row": c.row,
            "col": c.col,
            "rowspan": c.rowspan,
            "colspan": c.colspan,
            "text": c.text,
            "is_header": c.is_header,
            "is_merged": c.rowspan > 1 or c.colspan > 1,
            "bbox": None if c.bbox is None else _round_bbox(c.bbox),
        }
        cell_conf = getattr(c, "confidence", None)
        if cell_conf is not None:
            cell_dict["confidence"] = round(float(cell_conf), 4)
        cells.append(cell_dict)
    return {
        "id": tbl_id,
        "page": page_no,
        "bbox": bbox,
        "num_rows": table.num_rows,
        "num_cols": table.num_cols,
        "source": table.source,
        "confidence": table.confidence,
        "continuation_of": getattr(table, "continuation_of", None),
        "cells": cells,
        "html": table_to_html(table),
        "source_ref": {"file": source_file, "page": page_no, "bbox": bbox},
        "metadata": {
            "element_id": elem_id,
            "source": table.source,
            "confidence": table.confidence,
        },
    }


def _is_running_element(elem: dict[str, Any]) -> bool:
    etype = str(elem.get("type", "")).lower()
    if etype in ("header", "footer"):
        return True
    meta = elem.get("metadata") or {}
    return bool(meta.get("is_running_furniture"))


def _is_heading_element(elem: dict[str, Any]) -> bool:
    etype = str(elem.get("type", "")).lower()
    if etype in ("heading", "title", "section_header"):
        return True
    meta = elem.get("metadata") or {}
    lbl = str(meta.get("layout_label", "")).lower()
    return (
        lbl in ("title", "section_header")
        or str(meta.get("role", "")).lower() == "heading"
        or meta.get("heading_level") is not None
        or bool(meta.get("is_heading"))
    )


def _header_signature(tbl: dict[str, Any]) -> list[str] | None:
    hdr_cells = [c for c in tbl.get("cells", ()) if c.get("row") == 0 and c.get("is_header")]
    if not hdr_cells:
        return None
    hdr_cells.sort(key=lambda c: int(c.get("col", 0)))
    return [" ".join(str(c.get("text", "")).split()).lower() for c in hdr_cells]


def link_table_continuations(
    tables: list[dict[str, Any]],
    elements: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(tables) < 2:
        return [dict(t) for t in tables]

    out = [dict(t) for t in tables]
    elem_pos_by_id: dict[str, int] = {}
    body_by_page: dict[int, list[dict[str, Any]]] = {}
    doc_max_y = 0.0
    for idx, elem in enumerate(elements):
        eid = elem.get("id")
        if eid is not None:
            elem_pos_by_id[str(eid)] = idx
        pno = int(elem.get("page", 0))
        bbox = elem.get("bbox")
        if bbox and len(bbox) == 4:
            doc_max_y = max(doc_max_y, float(bbox[3]))
        if not _is_running_element(elem):
            body_by_page.setdefault(pno, []).append(elem)

    for tbl in out:
        bbox = tbl.get("bbox")
        if bbox and len(bbox) == 4:
            doc_max_y = max(doc_max_y, float(bbox[3]))
    est_h = max(1.0, doc_max_y)

    by_page: dict[int, list[int]] = {}
    for ti, tbl in enumerate(out):
        by_page.setdefault(int(tbl.get("page", 0)), []).append(ti)

    for pno, prev_indices in sorted(by_page.items()):
        next_indices = by_page.get(pno + 1)
        if not next_indices:
            continue
        prev_idx = max(prev_indices, key=lambda i: float(out[i]["bbox"][3]))
        cur_idx = min(next_indices, key=lambda i: float(out[i]["bbox"][1]))
        prev_tbl = out[prev_idx]
        cur_tbl = out[cur_idx]

        ncols = int(prev_tbl.get("num_cols", 0))
        if ncols <= 0 or int(cur_tbl.get("num_cols", 0)) != ncols:
            continue

        pb = prev_tbl["bbox"]
        cb = cur_tbl["bbox"]
        x_ov = max(0.0, min(float(pb[2]), float(cb[2])) - max(float(pb[0]), float(cb[0])))
        min_w = max(1.0, min(float(pb[2]) - float(pb[0]), float(cb[2]) - float(cb[0])))
        if x_ov < 0.5 * min_w:
            continue

        if float(pb[3]) < 0.45 * est_h or float(cb[1]) > 0.45 * est_h:
            continue

        prev_eid = str((prev_tbl.get("metadata") or {}).get("element_id", ""))
        cur_eid = str((cur_tbl.get("metadata") or {}).get("element_id", ""))
        has_below = any(
            str(e.get("id", "")) != prev_eid
            and str(e.get("parent_id", "")) != prev_eid
            and e.get("bbox")
            and float(e["bbox"][1]) >= float(pb[3]) - 2.0
            for e in body_by_page.get(pno, ())
        )
        has_above = any(
            str(e.get("id", "")) != cur_eid
            and str(e.get("parent_id", "")) != cur_eid
            and e.get("bbox")
            and float(e["bbox"][3]) <= float(cb[1]) + 2.0
            for e in body_by_page.get(pno + 1, ())
        )
        if has_below or has_above:
            continue

        pos_prev = elem_pos_by_id.get(prev_eid)
        pos_cur = elem_pos_by_id.get(cur_eid)
        if pos_prev is not None and pos_cur is not None and pos_prev < pos_cur:
            if any(_is_heading_element(elements[k]) for k in range(pos_prev + 1, pos_cur)):
                continue

        cur_hdr = _header_signature(cur_tbl)
        if cur_hdr is not None:
            prev_hdr = _header_signature(prev_tbl)
            if prev_hdr is None or prev_hdr != cur_hdr:
                continue

        cur_tbl["continuation_of"] = str(prev_tbl["id"])

    return out


def extract_tables_on_page(
    page: Any,
    table_boxes_pt: Sequence[BBox],
    *,
    words: Sequence[Word] | None = None,
    is_raster: bool = False,
    extractor: str | TableExtractor = "auto",
    ocr_backend: str = "rapidocr-onnx",
    device: str = "auto",
    dpi: int = DEFAULT_TABLE_DPI,
) -> list[ExtractedTable]:
    if not table_boxes_pt:
        return []
    backend = get_backend("table", extractor, device=device) if isinstance(extractor, str) else extractor
    extract_page = getattr(backend, "extract_from_page", None)
    if callable(extract_page):
        return list(
            extract_page(
                page,
                table_boxes_pt,
                words=words,
                is_raster=is_raster,
                ocr_backend=ocr_backend,
                device=device,
                dpi=dpi,
            )
        )

    ocr_rec = get_backend("ocr", ocr_backend, device=device) if is_raster else None
    if is_raster:
        native_tokens: list[tuple[str, BBox]] = []
    elif words is None:
        from .text import extract_words

        native_tokens = [(w.text, w.bbox) for w in extract_words(page)]
    else:
        native_tokens = [(w.text, w.bbox) for w in words]

    results: list[ExtractedTable] = []
    for box in table_boxes_pt:
        crop, crop_box = crop_page_region(page, box, dpi)
        if is_raster and ocr_rec is not None:
            box_tokens = ocr_crop_tokens(ocr_rec, crop, crop_box)
        else:
            box_tokens = tokens_in_box(native_tokens, crop_box)
        results.append(backend.extract(crop, crop_box, box_tokens))
    return results
