from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import pptx
from lxml import etree
from pptx.shapes.shapetree import SlideShapeFactory

from geometry import ResolvedConnector, ResolvedShape, Xfrm, emu_to_pt, resolve_connector, resolve_shape

XfrmSource = Literal["explicit", "inherited", "missing"]
ConnectionStatus = Literal["resolved", "dangling", "ambiguous", "invalid"]

_NAMESPACES = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}
_HIDDEN_VALUES = ("0", "false")


def _q(name: str) -> str:
    prefix, local_name = name.split(":")
    return f"{{{_NAMESPACES[prefix]}}}{local_name}"


_SHAPE_ELEMENTS = frozenset(_q(f"p:{tag}") for tag in ("sp", "cxnSp", "grpSp", "pic", "graphicFrame"))


@dataclass(frozen=True)
class Connection:
    target_id: int | None
    site_index: int | None
    target_z_order: int | None
    status: ConnectionStatus


@dataclass(frozen=True)
class TableCell:
    row: int
    col: int
    rowspan: int
    colspan: int
    text: str


@dataclass(frozen=True)
class Table:
    num_rows: int
    num_cols: int
    first_row_header: bool
    cells: tuple[TableCell, ...]


@dataclass(frozen=True)
class Shape:
    id: int
    name: str
    tag: str
    parent_id: int | None
    parent_z_order: int | None
    z_order: int
    text: str | None
    raw_xfrm: dict[str, Any] | None
    xfrm_source: XfrmSource
    alternate_content: dict[str, str | None] | None
    geometry: ResolvedShape | ResolvedConnector | None
    start_connection: Connection | None = None
    end_connection: Connection | None = None
    table: Table | None = None


@dataclass(frozen=True)
class Slide:
    number: int
    slide_id: int
    hidden: bool
    shapes: tuple[Shape, ...]


@dataclass(frozen=True)
class Presentation:
    width_pt: float | None
    height_pt: float | None
    slides: tuple[Slide, ...]


@dataclass(frozen=True)
class _Context:
    inherited: dict[etree._Element, Xfrm]
    parent_id: int | None = None
    parent_z_order: int | None = None
    ancestors: tuple[Xfrm, ...] = ()
    alternate_content: dict[str, str | None] | None = None


@dataclass(frozen=True)
class _Reference:
    target_id: int | None
    site_index: int | None


@dataclass(frozen=True)
class _Pending:
    shape: Shape
    start: _Reference | None
    end: _Reference | None


def extract_presentation(path: str | Path) -> Presentation:
    deck = pptx.Presentation(path)
    return Presentation(
        width_pt=_points(deck.slide_width),
        height_pt=_points(deck.slide_height),
        slides=tuple(_extract_slide(number, slide) for number, slide in enumerate(deck.slides, start=1)),
    )


def _points(length: int | None) -> float | None:
    return None if length is None else emu_to_pt(length)


def _extract_slide(number: int, slide: pptx.slide.Slide) -> Slide:
    pending: list[_Pending] = []
    sp_tree = slide._element.find(f"{_q('p:cSld')}/{_q('p:spTree')}")
    _walk(sp_tree, _Context(inherited=_inherited_frames(slide)), pending)
    owners = _owners(pending)
    shapes = tuple(
        replace(
            entry.shape,
            start_connection=_connect(entry.start, owners),
            end_connection=_connect(entry.end, owners),
        )
        for entry in pending
    )
    return Slide(
        number=number,
        slide_id=slide.slide_id,
        hidden=slide._element.get("show") in _HIDDEN_VALUES,
        shapes=shapes,
    )


def _walk(container: etree._Element, context: _Context, pending: list[_Pending]) -> None:
    for child in container:
        if child.tag == _q("mc:AlternateContent"):
            selected = _select_branch(child)
            if selected is not None:
                branch, alternate_content = selected
                _walk(branch, replace(context, alternate_content=alternate_content), pending)
        elif child.tag in _SHAPE_ELEMENTS:
            shape, placement = _describe(child, len(pending), context)
            start, end = _references(child)
            pending.append(_Pending(shape, start, end))
            if shape.tag == "grpSp":
                group = replace(
                    context,
                    parent_id=shape.id,
                    parent_z_order=shape.z_order,
                    ancestors=(*context.ancestors, placement),
                )
                _walk(child, group, pending)


# One branch per AlternateContent: the first Choice, else Fallback, that directly holds a recognised shape.
def _select_branch(alternate: etree._Element) -> tuple[etree._Element, dict[str, str | None]] | None:
    branches = [
        (choice, {"branch": "Choice", "requires": choice.get("Requires")})
        for choice in alternate.findall(_q("mc:Choice"))
    ]
    branches += [
        (fallback, {"branch": "Fallback", "requires": None})
        for fallback in alternate.findall(_q("mc:Fallback"))
    ]
    for branch, metadata in branches:
        if any(child.tag in _SHAPE_ELEMENTS for child in branch):
            return branch, metadata
    return None


def _owners(pending: list[_Pending]) -> dict[int, list[int]]:
    owners: dict[int, list[int]] = {}
    for entry in pending:
        owners.setdefault(entry.shape.id, []).append(entry.shape.z_order)
    return owners


def _connect(reference: _Reference | None, owners: dict[int, list[int]]) -> Connection | None:
    if reference is None:
        return None
    if reference.target_id is None or reference.site_index is None:
        return Connection(reference.target_id, reference.site_index, None, "invalid")
    matches = owners.get(reference.target_id, [])
    if len(matches) == 1:
        return Connection(reference.target_id, reference.site_index, matches[0], "resolved")
    status = "ambiguous" if len(matches) > 1 else "dangling"
    return Connection(reference.target_id, reference.site_index, None, status)


def _describe(element: etree._Element, z_order: int, context: _Context) -> tuple[Shape, Xfrm]:
    tag = etree.QName(element).localname
    cnv_pr = element.find(f"./*/{_q('p:cNvPr')}")
    xfrm = _xfrm_element(element, tag)
    placement, source = _placement(element, tag, xfrm, context.inherited)
    geometry = None if source == "missing" else _resolve(tag, placement, context.ancestors)
    shape = Shape(
        id=int(cnv_pr.get("id")),
        name=cnv_pr.get("name", ""),
        tag=tag,
        parent_id=context.parent_id,
        parent_z_order=context.parent_z_order,
        z_order=z_order,
        text=_text(element),
        raw_xfrm=_raw_xfrm(xfrm),
        xfrm_source=source,
        alternate_content=context.alternate_content,
        geometry=geometry,
        table=_table(element),
    )
    return shape, placement


def _xfrm_element(element: etree._Element, tag: str) -> etree._Element | None:
    if tag == "graphicFrame":
        return element.find(_q("p:xfrm"))
    if tag == "grpSp":
        return element.find(f"{_q('p:grpSpPr')}/{_q('a:xfrm')}")
    return element.find(f"{_q('p:spPr')}/{_q('a:xfrm')}")


def _placement(
    element: etree._Element,
    tag: str,
    xfrm: etree._Element | None,
    inherited: dict[etree._Element, Xfrm],
) -> tuple[Xfrm, XfrmSource]:
    if xfrm is not None:
        placement, source = Xfrm.from_element(xfrm), "explicit"
    elif element in inherited:
        placement, source = inherited[element], "inherited"
    else:
        return Xfrm(), "missing"
    if tag == "graphicFrame":
        # MS-OI29500: Office ignores rot and flips on graphic frames.
        placement = replace(placement, rot=0, flip_h=False, flip_v=False)
    return placement, source


def _resolve(tag: str, placement: Xfrm, ancestors: tuple[Xfrm, ...]) -> ResolvedShape | ResolvedConnector:
    if tag == "cxnSp":
        return resolve_connector(placement, ancestors)
    return resolve_shape(placement, ancestors)


def _raw_xfrm(xfrm: etree._Element | None) -> dict[str, Any] | None:
    if xfrm is None:
        return None
    return {
        "off": _pair(xfrm.find(_q("a:off")), "x", "y"),
        "ext": _pair(xfrm.find(_q("a:ext")), "cx", "cy"),
        "chOff": _pair(xfrm.find(_q("a:chOff")), "x", "y"),
        "chExt": _pair(xfrm.find(_q("a:chExt")), "cx", "cy"),
        "rot": xfrm.get("rot"),
        "flipH": xfrm.get("flipH"),
        "flipV": xfrm.get("flipV"),
    }


def _pair(node: etree._Element | None, first: str, second: str) -> dict[str, str | None] | None:
    if node is None:
        return None
    return {first: node.get(first), second: node.get(second)}


def _text(element: etree._Element) -> str | None:
    body = element.find(_q("p:txBody"))
    if body is None:
        return None
    return _paragraph_text(body)


# a:br becomes a vertical tab, as python-pptx paragraph.text does.
def _paragraph_text(body: etree._Element) -> str:
    return "\n".join(
        "".join("\v" if node.tag == _q("a:br") else node.text or "" for node in paragraph.iter(_q("a:t"), _q("a:br")))
        for paragraph in body.findall(_q("a:p"))
    )


def _table(element: etree._Element) -> Table | None:
    tbl = element.find(f"{_q('a:graphic')}/{_q('a:graphicData')}/{_q('a:tbl')}")
    if tbl is None:
        return None
    rows = tbl.findall(_q("a:tr"))
    properties = tbl.find(_q("a:tblPr"))
    return Table(
        num_rows=len(rows),
        num_cols=len(tbl.findall(f"{_q('a:tblGrid')}/{_q('a:gridCol')}")),
        first_row_header=properties is not None and _is_true(properties.get("firstRow")),
        cells=tuple(_table_cells(rows)),
    )


def _table_cells(rows: list[etree._Element]) -> Iterator[TableCell]:
    for row_index, row in enumerate(rows):
        for col_index, cell in enumerate(row.findall(_q("a:tc"))):
            if _is_true(cell.get("hMerge")) or _is_true(cell.get("vMerge")):
                continue
            body = cell.find(_q("a:txBody"))
            yield TableCell(
                row=row_index,
                col=col_index,
                rowspan=int(cell.get("rowSpan") or 1),
                colspan=int(cell.get("gridSpan") or 1),
                text="" if body is None else _paragraph_text(body),
            )


def _is_true(value: str | None) -> bool:
    return value in ("1", "true")


def _references(element: etree._Element) -> tuple[_Reference | None, _Reference | None]:
    if etree.QName(element).localname != "cxnSp":
        return None, None
    cnv_cxn = element.find(f"{_q('p:nvCxnSpPr')}/{_q('p:cNvCxnSpPr')}")
    return _reference(cnv_cxn.find(_q("a:stCxn"))), _reference(cnv_cxn.find(_q("a:endCxn")))


def _reference(node: etree._Element | None) -> _Reference | None:
    if node is None:
        return None
    return _Reference(target_id=_to_int(node.get("id")), site_index=_to_int(node.get("idx")))


def _to_int(value: str | None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_placeholder(element: etree._Element) -> bool:
    return element.find(f"./*/{_q('p:nvPr')}/{_q('p:ph')}") is not None


# Walks the XML: python-pptx's slide.shapes cannot iterate a p:contentPart.
def _inherited_frames(slide: pptx.slide.Slide) -> dict[etree._Element, Xfrm]:
    frames: dict[etree._Element, Xfrm] = {}
    for element in slide._element.iter(*_SHAPE_ELEMENTS):
        tag = etree.QName(element).localname
        if _is_placeholder(element) and _xfrm_element(element, tag) is None:
            shape = SlideShapeFactory(element, slide.shapes)
            left, top, width, height = shape.left, shape.top, shape.width, shape.height
            if all(value is not None for value in (left, top, width, height)):
                frames[element] = Xfrm(off_x=left, off_y=top, ext_cx=width, ext_cy=height)
    return frames
