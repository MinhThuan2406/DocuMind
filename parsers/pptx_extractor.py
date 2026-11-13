from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import pptx
from lxml import etree
from pptx.shapes.base import BaseShape
from pptx.shapes.group import GroupShape

from geometry import ResolvedConnector, ResolvedShape, Xfrm, emu_to_pt, resolve_connector, resolve_shape

XfrmSource = Literal["explicit", "inherited", "missing"]

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
class Shape:
    id: int
    name: str
    tag: str
    parent_id: int | None
    z_order: int
    text: str | None
    raw_xfrm: dict[str, Any] | None
    xfrm_source: XfrmSource
    alternate_content: dict[str, str | None] | None
    geometry: ResolvedShape | ResolvedConnector | None


@dataclass(frozen=True)
class Slide:
    number: int
    slide_id: int
    hidden: bool
    shapes: tuple[Shape, ...]


@dataclass(frozen=True)
class Presentation:
    width_pt: float
    height_pt: float
    slides: tuple[Slide, ...]


@dataclass(frozen=True)
class _Context:
    inherited: dict[etree._Element, Xfrm]
    parent_id: int | None = None
    ancestors: tuple[Xfrm, ...] = ()
    alternate_content: dict[str, str | None] | None = None


def extract_presentation(path: str | Path) -> Presentation:
    deck = pptx.Presentation(path)
    return Presentation(
        width_pt=emu_to_pt(deck.slide_width),
        height_pt=emu_to_pt(deck.slide_height),
        slides=tuple(_extract_slide(number, slide) for number, slide in enumerate(deck.slides, start=1)),
    )


def _extract_slide(number: int, slide: pptx.slide.Slide) -> Slide:
    shapes: list[Shape] = []
    sp_tree = slide._element.find(f"{_q('p:cSld')}/{_q('p:spTree')}")
    _walk(sp_tree, _Context(inherited=_inherited_frames(slide)), shapes)
    return Slide(
        number=number,
        slide_id=slide.slide_id,
        hidden=slide._element.get("show") in _HIDDEN_VALUES,
        shapes=tuple(shapes),
    )


def _walk(container: etree._Element, context: _Context, shapes: list[Shape]) -> None:
    for child in container:
        if child.tag == _q("mc:AlternateContent"):
            selected = _select_branch(child)
            if selected is not None:
                branch, alternate_content = selected
                _walk(branch, replace(context, alternate_content=alternate_content), shapes)
        elif child.tag in _SHAPE_ELEMENTS:
            shape, placement = _describe(child, len(shapes), context)
            shapes.append(shape)
            if shape.tag == "grpSp":
                group = replace(context, parent_id=shape.id, ancestors=(*context.ancestors, placement))
                _walk(child, group, shapes)


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
        z_order=z_order,
        text=_text(element),
        raw_xfrm=_raw_xfrm(xfrm),
        xfrm_source=source,
        alternate_content=context.alternate_content,
        geometry=geometry,
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
    return "\n".join(
        "".join(node.text or "" for node in paragraph.iter(_q("a:t")))
        for paragraph in body.findall(_q("a:p"))
    )


def _is_placeholder(element: etree._Element) -> bool:
    return element.find(f"./*/{_q('p:nvPr')}/{_q('p:ph')}") is not None


def _inherited_frames(slide: pptx.slide.Slide) -> dict[etree._Element, Xfrm]:
    frames: dict[etree._Element, Xfrm] = {}
    for shape in _pptx_shapes(slide.shapes):
        element = shape._element
        tag = etree.QName(element).localname
        if _is_placeholder(element) and _xfrm_element(element, tag) is None:
            left, top, width, height = shape.left, shape.top, shape.width, shape.height
            if all(value is not None for value in (left, top, width, height)):
                frames[element] = Xfrm(off_x=left, off_y=top, ext_cx=width, ext_cy=height)
    return frames


def _pptx_shapes(shapes: Iterable[BaseShape]) -> Iterator[BaseShape]:
    for shape in shapes:
        yield shape
        if isinstance(shape, GroupShape):
            yield from _pptx_shapes(shape.shapes)
