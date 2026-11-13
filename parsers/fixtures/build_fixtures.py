import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from lxml import etree
from pptx import Presentation
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.oxml.ns import qn

SLIDE_W = 12192000
SLIDE_H = 6858000
NS_MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"
NS_A14 = "http://schemas.microsoft.com/office/drawing/2010/main"
CORE_DATE = datetime(2026, 1, 1)
CORE_AUTHOR = "fixture-builder"
MARKER_OFF = (1000000, 1000000)
MARKER_EXT = (10000000, 1000000)

Point = tuple[int, int]


def _xfrm_of(el: etree._Element) -> etree._Element:
    tag = etree.QName(el).localname
    if tag == "graphicFrame":
        return el.find(qn("p:xfrm"))
    if tag == "grpSp":
        return el.find(qn("p:grpSpPr")).find(qn("a:xfrm"))
    return el.find(qn("p:spPr")).get_or_add_xfrm()


def _named(values: Point | None, names: tuple[str, str]) -> dict[str, str] | None:
    if values is None:
        return None
    return {name: str(value) for name, value in zip(names, values)}


def set_xfrm(el: etree._Element, off: Point, ext: Point, ch_off: Point | None = None,
             ch_ext: Point | None = None, rot: str | None = None,
             flip_h: str | None = None, flip_v: str | None = None) -> dict:
    xfrm = _xfrm_of(el)
    xfrm.clear()
    if rot is not None:
        xfrm.set("rot", rot)
    if flip_h is not None:
        xfrm.set("flipH", flip_h)
    if flip_v is not None:
        xfrm.set("flipV", flip_v)
    etree.SubElement(xfrm, qn("a:off"), x=str(off[0]), y=str(off[1]))
    etree.SubElement(xfrm, qn("a:ext"), cx=str(ext[0]), cy=str(ext[1]))
    if ch_off is not None:
        etree.SubElement(xfrm, qn("a:chOff"), x=str(ch_off[0]), y=str(ch_off[1]))
        etree.SubElement(xfrm, qn("a:chExt"), cx=str(ch_ext[0]), cy=str(ch_ext[1]))
    return {
        "off": _named(off, ("x", "y")),
        "ext": _named(ext, ("cx", "cy")),
        "chOff": _named(ch_off, ("x", "y")),
        "chExt": _named(ch_ext, ("cx", "cy")),
        "rot": rot,
        "flipH": flip_h,
        "flipV": flip_v,
    }


def set_ident(el: etree._Element, cid: int, name: str) -> None:
    cnv = el[0][0]
    assert cnv.tag == qn("p:cNvPr"), cnv.tag
    cnv.set("id", str(cid))
    cnv.set("name", name)


def _ref(ref: Point | None) -> dict[str, str] | None:
    return _named(ref, ("id", "idx"))


def set_cxn(el: etree._Element, st: Point | None, end: Point | None) -> tuple[dict | None, dict | None]:
    cnv_cxn = el.find(qn("p:nvCxnSpPr")).find(qn("p:cNvCxnSpPr"))
    refs = [(ref, tag) for ref, tag in ((st, "a:stCxn"), (end, "a:endCxn")) if ref is not None]
    for index, (ref, tag) in enumerate(refs):
        cnv_cxn.insert(index, cnv_cxn.makeelement(qn(tag), {"id": str(ref[0]), "idx": str(ref[1])}))
    return _ref(st), _ref(end)


class Fixture:
    def __init__(self) -> None:
        self.slides: list[dict] = []
        self.shapes: list[dict] = []
        self.tables: list[dict] = []

    def to_dict(self) -> dict:
        return {
            "slide_width": SLIDE_W,
            "slide_height": SLIDE_H,
            "slides": sorted(self.slides, key=lambda slide: slide["presentation_index"]),
            "shapes": sorted(self.shapes, key=lambda rec: rec["slide"]),
            "tables": self.tables,
        }


class Slide:
    def __init__(self, fx: Fixture, prs: Presentation, layout_name: str, marker: str,
                 pres_index: int = 1, part: str = "slide1.xml", show: str | None = None) -> None:
        self.fx = fx
        self.pres_index = pres_index
        self.part = part
        layout = next(item for item in prs.slide_layouts if item.name == layout_name)
        self.obj = prs.slides.add_slide(layout)
        if show is not None:
            self.obj._element.set("show", show)
        fx.slides.append({"presentation_index": pres_index, "part": part,
                          "marker": marker, "show": show})

    def record(self, el: etree._Element, cid: int, name: str, tag: str, *, text: str | None = None,
               parent: int | None = None, xfrm: dict | None = None,
               cxn: tuple[dict | None, dict | None] = (None, None)) -> dict:
        set_ident(el, cid, name)
        rec = {
            "slide": self.pres_index, "part": self.part, "id": cid, "name": name, "tag": tag,
            "text": text, "xfrm": xfrm, "parent": parent,
            "stCxn": cxn[0], "endCxn": cxn[1], "alternate_content": None,
        }
        self.fx.shapes.append(rec)
        return rec

    def textbox(self, cid: int, name: str, text: str, off: Point, ext: Point) -> None:
        shp = self.obj.shapes.add_textbox(*off, *ext)
        shp.text_frame.text = text
        self.record(shp._element, cid, name, "sp", text=text, xfrm=set_xfrm(shp._element, off, ext))

    def box(self, shapes: Any, cid: int, name: str, text: str, off: Point, ext: Point, *,
            parent: int | None = None, prst: MSO_SHAPE = MSO_SHAPE.RECTANGLE,
            rot: str | None = None, flip_h: str | None = None,
            flip_v: str | None = None) -> tuple[Any, dict]:
        shp = shapes.add_shape(prst, *off, *ext)
        shp.text_frame.text = text
        xfrm = set_xfrm(shp._element, off, ext, rot=rot, flip_h=flip_h, flip_v=flip_v)
        return shp, self.record(shp._element, cid, name, "sp", text=text, parent=parent, xfrm=xfrm)

    def connector(self, shapes: Any, cid: int, name: str, off: Point, ext: Point, *,
                  parent: int | None = None, st: Point | None = None, end: Point | None = None,
                  rot: str | None = None, flip_h: str | None = None,
                  flip_v: str | None = None) -> tuple[Any, dict]:
        shp = shapes.add_connector(MSO_CONNECTOR.STRAIGHT, off[0], off[1], off[0] + ext[0], off[1] + ext[1])
        xfrm = set_xfrm(shp._element, off, ext, rot=rot, flip_h=flip_h, flip_v=flip_v)
        cxn = set_cxn(shp._element, st, end)
        return shp, self.record(shp._element, cid, name, "cxnSp", parent=parent, xfrm=xfrm, cxn=cxn)

    def group(self, shapes: Any, cid: int, name: str, parent: int | None = None) -> tuple[Any, dict]:
        grp = shapes.add_group_shape()
        return grp, self.record(grp._element, cid, name, "grpSp", parent=parent)

    def table(self, cid: int, name: str, off: Point, ext: Point, rows: int, cols: int) -> Any:
        frame = self.obj.shapes.add_table(rows, cols, *off, *ext)
        self.record(frame._element, cid, name, "graphicFrame", xfrm=set_xfrm(frame._element, off, ext))
        return frame.table


def new_presentation() -> Presentation:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    prs._element.find(qn("p:sldSz")).attrib.pop("type", None)
    return prs


def set_core_properties(prs: Presentation) -> None:
    props = prs.core_properties
    props.author = CORE_AUTHOR
    props.last_modified_by = CORE_AUTHOR
    props.created = CORE_DATE
    props.modified = CORE_DATE
    props.revision = 1


def build_slide_order(prs: Presentation, fx: Fixture) -> None:
    pres_index = {"C": 1, "A": 2, "B": 3}
    for part_number, letter in enumerate("ABC", start=1):
        slide = Slide(fx, prs, "Blank", f"SLIDE-{letter}", pres_index=pres_index[letter],
                      part=f"slide{part_number}.xml")
        slide.textbox(2, f"Marker {letter}", f"SLIDE-{letter}", MARKER_OFF, MARKER_EXT)
    # python-pptx has no sldIdLst reorder API, so the children are moved by hand.
    id_list = prs.slides._sldIdLst
    created = list(id_list)
    for el in created:
        id_list.remove(el)
    for index in (2, 0, 1):
        id_list.append(created[index])


def build_hidden_slide(prs: Presentation, fx: Fixture) -> None:
    visible = Slide(fx, prs, "Blank", "VISIBLE-SLIDE")
    visible.textbox(2, "Marker", "VISIBLE-SLIDE", MARKER_OFF, MARKER_EXT)
    hidden = Slide(fx, prs, "Blank", "HIDDEN-SLIDE", pres_index=2, part="slide2.xml", show="0")
    hidden.textbox(2, "Marker", "HIDDEN-SLIDE", MARKER_OFF, MARKER_EXT)


def build_shapes_and_connectors(prs: Presentation, fx: Fixture) -> None:
    slide = Slide(fx, prs, "Blank", "BOX-A")
    shapes = slide.obj.shapes
    box_a, _ = slide.box(shapes, 2, "Box A", "BOX-A", (600000, 900000), (2000000, 1000000))
    box_b, _ = slide.box(shapes, 3, "Box B", "BOX-B", (4000000, 900000), (2000000, 1000000))
    slide.box(shapes, 4, "Box C", "BOX-C", (600000, 3900000), (2000000, 1000000))
    slide.box(shapes, 5, "Box D", "BOX-D", (4000000, 3900000), (2000000, 1000000), prst=MSO_SHAPE.OVAL)

    conn1 = shapes.add_connector(MSO_CONNECTOR.STRAIGHT, 2600000, 1400000, 4000000, 1400000)
    conn1.begin_connect(box_a, 3)
    conn1.end_connect(box_b, 1)
    assert (conn1.begin_x, conn1.begin_y, conn1.end_x, conn1.end_y) == (2600000, 1400000, 4000000, 1400000)
    slide.record(conn1._element, 6, "Connector 1 Both Ends", "cxnSp",
                 xfrm=set_xfrm(conn1._element, (2600000, 1400000), (1400000, 0)),
                 cxn=({"id": "2", "idx": "3"}, {"id": "3", "idx": "1"}))

    slide.connector(shapes, 7, "Connector 2 Start Only", (1600000, 4900000), (2000000, 400000), st=(4, 2))
    slide.connector(shapes, 8, "Connector 3 Loose", (6200000, 2200000), (1800000, 800000))
    slide.connector(shapes, 9, "Connector 4 Dangling", (2600000, 2400000), (1000000, 1000000), st=(999, 0))
    slide.connector(shapes, 10, "Connector 5 FlipV", (8200000, 1000000), (1200000, 1000000), flip_v="1")
    slide.connector(shapes, 11, "Connector 6 FlipH", (9800000, 1000000), (1200000, 1000000), flip_h="1")
    slide.connector(shapes, 12, "Connector 7 Rot90", (8200000, 4300000), (1500000, 400000), rot="5400000")


def build_nested_groups(prs: Presentation, fx: Fixture) -> None:
    slide = Slide(fx, prs, "Blank", "INNER-1")
    outer, outer_rec = slide.group(slide.obj.shapes, 2, "Outer Group")
    inner, inner_rec = slide.group(outer.shapes, 3, "Inner Group", parent=2)
    slide.box(inner.shapes, 4, "Inner Box 1", "INNER-1", (1600000, 1600000), (400000, 300000), parent=3)
    slide.box(inner.shapes, 5, "Inner Box 2", "INNER-2", (2000000, 2100000), (300000, 300000), parent=3)
    slide.box(outer.shapes, 6, "Outer Child Box", "OUTER-CHILD", (2800000, 1450000), (600000, 600000), parent=2)
    slide.connector(outer.shapes, 7, "Outer Connector", (2000000, 1750000), (800000, 0),
                    parent=2, st=(4, 3), end=(6, 1))
    slide.box(slide.obj.shapes, 8, "Top Box", "TOP-BOX", (8000000, 1000000), (2000000, 1000000))
    slide.connector(slide.obj.shapes, 9, "Top Connector", (4500000, 1500000), (3500000, 600000),
                    st=(8, 1), end=(5, 3), flip_h="1")
    # python-pptx recomputes group extents on every child insert, so frames are written last.
    inner_rec["xfrm"] = set_xfrm(inner._element, (1500000, 1500000), (1000000, 1000000),
                                 ch_off=(1500000, 1500000), ch_ext=(1000000, 1000000), rot="2700000")
    outer_rec["xfrm"] = set_xfrm(outer._element, (1000000, 1000000), (6000000, 3000000),
                                 ch_off=(500000, 1000000), ch_ext=(3000000, 3000000))


def build_flip_rot(prs: Presentation, fx: Fixture) -> None:
    slide = Slide(fx, prs, "Blank", "FLIP-V")
    shapes = slide.obj.shapes
    size = (2400000, 1200000)
    slide.box(shapes, 2, "Flip V Only", "FLIP-V", (800000, 800000), size, flip_v="1")
    slide.box(shapes, 3, "Flip H Only", "FLIP-H", (3800000, 800000), size, flip_h="1")
    slide.box(shapes, 4, "Flip Both", "FLIP-HV", (6800000, 800000), size, flip_h="1", flip_v="1")
    slide.box(shapes, 5, "Rot 90", "ROT-90", (800000, 3600000), size, rot="5400000")
    slide.box(shapes, 6, "Rot 45 Flip H", "ROT-45-FLIP-H", (3800000, 3600000), size, rot="2700000", flip_h="1")
    slide.box(shapes, 7, "Rot 390 Flip V", "ROT-390-FLIP-V", (6800000, 3600000), size,
              rot="23400000", flip_v="1")


def _cell(row: int, col: int, grid_span: str | None = None, row_span: str | None = None,
          h_merge: str | None = None, v_merge: str | None = None) -> dict:
    return {"row": row, "col": col, "text": f"R{row}C{col}", "gridSpan": grid_span,
            "rowSpan": row_span, "hMerge": h_merge, "vMerge": v_merge}


TABLE_CELLS = [
    _cell(0, 0, grid_span="2"), _cell(0, 1, h_merge="1"), _cell(0, 2), _cell(0, 3),
    _cell(1, 0), _cell(1, 1), _cell(1, 2), _cell(1, 3, row_span="2"),
    _cell(2, 0, grid_span="2", row_span="2"), _cell(2, 1, row_span="2", h_merge="1"),
    _cell(2, 2), _cell(2, 3, v_merge="1"),
    _cell(3, 0, grid_span="2", v_merge="1"), _cell(3, 1, h_merge="1", v_merge="1"),
    _cell(3, 2), _cell(3, 3),
]


def build_tables_merged(prs: Presentation, fx: Fixture) -> None:
    slide = Slide(fx, prs, "Blank", "R0C0")
    table = slide.table(2, "Merged Table", (1000000, 1000000), (10000000, 4000000), rows=4, cols=4)
    table.cell(0, 0).merge(table.cell(0, 1))
    table.cell(1, 3).merge(table.cell(2, 3))
    table.cell(2, 0).merge(table.cell(3, 1))
    # merge() moves spanned text into the origin cell, so text is set after merging.
    for cell in TABLE_CELLS:
        table.cell(cell["row"], cell["col"]).text = cell["text"]
    fx.tables.append({"slide": slide.pres_index, "part": slide.part, "id": 2,
                      "name": "Merged Table", "cells": TABLE_CELLS})


def build_placeholders_inherit(prs: Presentation, fx: Fixture) -> None:
    slide = Slide(fx, prs, "Title and Content", "Inherited Title")
    title = slide.obj.placeholders[0]
    body = slide.obj.placeholders[1]
    title.text_frame.text = "Inherited Title"
    title_rec = slide.record(title._element, 2, "Title 1", "sp", text="Inherited Title")
    title_rec["placeholder"] = {"type": "title", "idx": None}
    title_rec["inherited_geometry"] = {"left": 457200, "top": 274638, "width": 8229600, "height": 1143000}
    body.text_frame.text = "Explicit body text"
    body_rec = slide.record(body._element, 3, "Content Placeholder 2", "sp", text="Explicit body text",
                            xfrm=set_xfrm(body._element, (1000000, 2000000), (9000000, 3500000)))
    body_rec["placeholder"] = {"type": None, "idx": "1"}


def build_alternate_content(prs: Presentation, fx: Fixture) -> None:
    slide = Slide(fx, prs, "Blank", "ORDINARY-BOX")
    sp_tree = slide.obj.shapes._spTree
    choice_shp, choice_rec = slide.box(slide.obj.shapes, 2, "Choice Shape", "CHOICE-SHAPE",
                                       (1000000, 1000000), (3000000, 1000000))
    fallback_shp, fallback_rec = slide.box(slide.obj.shapes, 3, "Fallback Shape", "FALLBACK-SHAPE",
                                           (1000000, 1000000), (3000000, 1000000))
    ac = sp_tree.makeelement(f"{{{NS_MC}}}AlternateContent", {}, nsmap={"mc": NS_MC})
    choice = etree.SubElement(ac, f"{{{NS_MC}}}Choice", Requires="a14")
    fallback = etree.SubElement(ac, f"{{{NS_MC}}}Fallback")
    choice.append(choice_shp._element)
    fallback.append(fallback_shp._element)
    sp_tree.append(ac)
    choice_rec["alternate_content"] = {"branch": "Choice", "requires": "a14"}
    fallback_rec["alternate_content"] = {"branch": "Fallback", "requires": None}
    slide.box(slide.obj.shapes, 4, "Ordinary Box", "ORDINARY-BOX", (6000000, 1000000), (3000000, 1000000))
    # a14 appears only in an attribute value, so keep_ns_prefixes stops lxml from dropping it.
    etree.cleanup_namespaces(slide.obj._element, top_nsmap={"mc": NS_MC, "a14": NS_A14},
                             keep_ns_prefixes=["p", "a", "r", "mc", "a14"])


def build_duplicate_ids(prs: Presentation, fx: Fixture) -> None:
    slide = Slide(fx, prs, "Blank", "ANCHOR")
    shapes = slide.obj.shapes
    slide.box(shapes, 2, "Anchor Box", "ANCHOR", (1000000, 1000000), (2500000, 1000000))
    _, one = slide.box(shapes, 5, "Dup Box One", "DUP-ONE", (5000000, 1000000), (2500000, 1000000))
    _, two = slide.box(shapes, 5, "Dup Box Two", "DUP-TWO", (5000000, 3000000), (2500000, 1000000))
    _, conn = slide.connector(shapes, 6, "Dup Connector", (3500000, 1500000), (1500000, 0),
                              st=(5, 1), end=(2, 3), flip_h="1")
    for rec in (one, two, conn):
        rec["ambiguous_id"] = True


BUILDERS = [
    ("slide_order.pptx", build_slide_order),
    ("hidden_slide.pptx", build_hidden_slide),
    ("shapes_and_connectors.pptx", build_shapes_and_connectors),
    ("nested_groups.pptx", build_nested_groups),
    ("flip_rot.pptx", build_flip_rot),
    ("tables_merged.pptx", build_tables_merged),
    ("placeholders_inherit.pptx", build_placeholders_inherit),
    ("alternate_content.pptx", build_alternate_content),
    ("duplicate_ids.pptx", build_duplicate_ids),
]

CONVENTIONS = [
    "slide: 1-based presentation index (order in p:sldIdLst). part: zip part name under ppt/slides/.",
    "marker: a text string that occurs on that slide. On slide_order and hidden_slide it is the only text "
    "box. On single-slide fixtures it is one chosen shape text.",
    "tag: sp, cxnSp, grpSp or graphicFrame, which is the local name of the XML element.",
    "xfrm: raw attributes of a:xfrm (p:xfrm for graphicFrame) as strings exactly as written. "
    "null means the attribute or child is absent. xfrm itself null means no xfrm element.",
    "parent: cNvPr id of the enclosing p:grpSp, or null at top level.",
    "stCxn and endCxn: raw a:stCxn and a:endCxn values (id, idx) as strings, or null.",
    "alternate_content: null, or {branch, requires} when the shape sits inside mc:AlternateContent.",
    "tables: cell rows and cols are 0-based. gridSpan, rowSpan, hMerge and vMerge are raw strings or null.",
]


def main(argv: list[str]) -> None:
    out_root = Path(argv[1]).resolve() if len(argv) > 1 else Path(__file__).resolve().parent
    (out_root / "pptx").mkdir(parents=True, exist_ok=True)
    fixtures = {}
    for name, build in BUILDERS:
        prs = new_presentation()
        fx = Fixture()
        build(prs, fx)
        set_core_properties(prs)
        prs.save(str(out_root / "pptx" / name))
        fixtures[name] = fx.to_dict()
        print(f"built {name}")
    manifest = {
        "format_version": 1,
        "generator": "parsers/fixtures/build_fixtures.py",
        "conventions": CONVENTIONS,
        "fixtures": fixtures,
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main(sys.argv)
