from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pptx

from geometry import ResolvedConnector, ResolvedShape

from .pptx_extractor import Connection, Presentation, Shape, Slide, TableCell, extract_presentation

_ELEMENT_TYPES = {"sp": "shape", "cxnSp": "connector", "grpSp": "group", "pic": "picture"}


@dataclass(frozen=True)
class _Ids:
    elements: dict[tuple[int, int], str]
    tables: dict[tuple[int, int], str]


def extract_ir(path: str | Path) -> dict[str, Any]:
    presentation = extract_presentation(path)
    source = Path(path).name
    placed = [(slide, shape) for slide in presentation.slides for shape in slide.shapes]
    tabled = [(slide, shape) for slide, shape in placed if shape.table is not None]
    ids = _Ids(
        elements={_key(slide, shape): _numbered("elem", n) for n, (slide, shape) in enumerate(placed, start=1)},
        tables={_key(slide, shape): _numbered("tbl", n) for n, (slide, shape) in enumerate(tabled, start=1)},
    )
    levels: dict[tuple[int, int], int] = {}
    elements: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    for slide, shape in placed:
        key = _key(slide, shape)
        parent = (slide.number, shape.parent_z_order)
        levels[key] = 1 if shape.parent_z_order is None else levels[parent] + 1
        elements.append(_element(slide, shape, source, ids, levels[key]))
        if shape.table is not None:
            tables.append(_table(slide, shape, source, ids))
    return {
        "doc_id": Path(path).stem,
        "source_file": source,
        "doc_type": "pptx",
        "metadata": _document_metadata(presentation),
        "elements": elements,
        "tables": tables,
        "diagrams": [],
    }


def _key(slide: Slide, shape: Shape) -> tuple[int, int]:
    return slide.number, shape.z_order


def _numbered(prefix: str, number: int) -> str:
    return f"{prefix}_{number:04d}"


def _document_metadata(presentation: Presentation) -> dict[str, Any]:
    return {
        "page_width": presentation.width_pt,
        "page_height": presentation.height_pt,
        "coord_unit": "pt",
        "parser": f"pptx_extractor (python-pptx {pptx.__version__})",
        "reading_order_status": "z_order",
        "slides": [{"number": s.number, "slide_id": s.slide_id, "hidden": s.hidden} for s in presentation.slides],
    }


def _element(slide: Slide, shape: Shape, source: str, ids: _Ids, level: int) -> dict[str, Any]:
    key = _key(slide, shape)
    table_id = ids.tables.get(key)
    box = _box(shape.geometry)
    bbox = None if box is None else list(box.bbox)
    element = {
        "id": ids.elements[key],
        "type": _element_type(shape),
        "page": slide.number,
        "bbox": bbox,
        "reading_order": shape.z_order,
        "level": level,
        "parent_id": None if shape.parent_z_order is None else ids.elements[(slide.number, shape.parent_z_order)],
        "text": _element_text(shape, table_id),
        "source_ref": _source_ref(source, slide.number, bbox, shape.id),
        "metadata": _element_metadata(slide, shape, box, table_id),
    }
    if shape.tag == "cxnSp":
        element["metadata"].update(_connector_metadata(slide, shape, ids))
    if table_id is not None:
        element["table_id"] = table_id
    return element


def _element_type(shape: Shape) -> str:
    if shape.tag == "graphicFrame":
        return "table" if shape.table is not None else "graphic_frame"
    return _ELEMENT_TYPES[shape.tag]


def _element_text(shape: Shape, table_id: str | None) -> str:
    if table_id is None:
        return shape.text or ""
    return f"[Table {table_id}: {shape.table.num_rows}x{shape.table.num_cols}]"


def _element_metadata(
    slide: Slide, shape: Shape, box: ResolvedShape | None, table_id: str | None
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "tag": shape.tag,
        "shape_id": shape.id,
        "shape_name": shape.name,
        "z_order": shape.z_order,
        "slide_id": slide.slide_id,
        "raw_xfrm": shape.raw_xfrm,
        "xfrm_source": shape.xfrm_source,
        "rotation_deg": None if box is None else box.rotation_deg,
        "mirrored": None if box is None else box.mirrored,
        "alternate_content": shape.alternate_content,
    }
    if table_id is not None:
        metadata["table_id"] = table_id
    return metadata


def _connector_metadata(slide: Slide, shape: Shape, ids: _Ids) -> dict[str, Any]:
    connector = shape.geometry
    return {
        "start": None if connector is None else list(connector.start),
        "end": None if connector is None else list(connector.end),
        "start_connection": _connection(shape.start_connection, slide.number, ids),
        "end_connection": _connection(shape.end_connection, slide.number, ids),
    }


def _connection(connection: Connection | None, page: int, ids: _Ids) -> dict[str, Any] | None:
    if connection is None:
        return None
    target = ids.elements[(page, connection.target_z_order)] if connection.status == "resolved" else None
    return {
        "target_id": connection.target_id,
        "site_index": connection.site_index,
        "status": connection.status,
        "target_element_id": target,
    }


def _table(slide: Slide, shape: Shape, source: str, ids: _Ids) -> dict[str, Any]:
    key = _key(slide, shape)
    table = shape.table
    box = _box(shape.geometry)
    bbox = None if box is None else list(box.bbox)
    return {
        "id": ids.tables[key],
        "page": slide.number,
        "bbox": bbox,
        "num_rows": table.num_rows,
        "num_cols": table.num_cols,
        "cells": [_cell(cell, table.first_row_header) for cell in table.cells],
        "source_ref": _source_ref(source, slide.number, bbox, shape.id),
        "metadata": {"element_id": ids.elements[key]},
    }


def _cell(cell: TableCell, first_row_header: bool) -> dict[str, Any]:
    return {
        "row": cell.row,
        "col": cell.col,
        "rowspan": cell.rowspan,
        "colspan": cell.colspan,
        "text": cell.text,
        "is_header": first_row_header and cell.row == 0,
        "is_merged": cell.rowspan > 1 or cell.colspan > 1,
        "bbox": None,
    }


def _source_ref(source: str, page: int, bbox: list[float] | None, shape_id: int) -> dict[str, Any]:
    return {"file": source, "page": page, "bbox": bbox, "shape_id": shape_id}


def _box(geometry: ResolvedShape | ResolvedConnector | None) -> ResolvedShape | None:
    if isinstance(geometry, ResolvedConnector):
        return geometry.box
    return geometry
