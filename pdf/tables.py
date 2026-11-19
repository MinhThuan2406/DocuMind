from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PIL import Image

from ._pdf_io import crop_page_rgb_with_bbox
from .backends import TableExtractor, get as get_backend
from .backends.tatr_onnx import table_to_html
from .models import BBox, ExtractedTable, Word

DEFAULT_TABLE_DPI = 200
_CROP_PAD_PT = 4.0


def _round_bbox(bbox: BBox) -> list[float]:
    return [round(v, 3) + 0.0 for v in bbox]


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
            cell_dict["confidence"] = cell_conf
        cells.append(cell_dict)
    return {
        "id": tbl_id,
        "page": page_no,
        "bbox": bbox,
        "num_rows": table.num_rows,
        "num_cols": table.num_cols,
        "source": table.source,
        "confidence": table.confidence,
        "cells": cells,
        "html": table_to_html(table),
        "source_ref": {"file": source_file, "page": page_no, "bbox": bbox},
        "metadata": {
            "element_id": elem_id,
            "source": table.source,
            "confidence": table.confidence,
        },
    }


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
