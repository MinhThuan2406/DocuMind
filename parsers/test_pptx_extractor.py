import dataclasses
import json
import math
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pptx  # noqa: E402
from lxml import etree  # noqa: E402

from geometry import (  # noqa: E402
    EMU_PER_POINT,
    ResolvedConnector,
    ResolvedShape,
    Xfrm,
    resolve_connector,
    resolve_shape,
)
from parsers.pptx_extractor import extract_presentation  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
PPTX_DIR = FIXTURE_DIR / "pptx"
MANIFEST = json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))["fixtures"]
NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"
NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
NS_P14 = "http://schemas.microsoft.com/office/powerpoint/2010/main"
TOLERANCE = 1e-5
RECT_SITES = {0: (0.5, 0.0), 1: (0.0, 0.5), 2: (0.5, 1.0), 3: (1.0, 0.5)}
UNKNOWN = "<p14:contentPart/>"
_DECLARATIONS = f'xmlns:a="{NS_A}" xmlns:mc="{NS_MC}" xmlns:p="{NS_P}" xmlns:p14="{NS_P14}"'


def _open(name):
    return extract_presentation(PPTX_DIR / name)


def _shape(slide, name):
    matches = [shape for shape in slide.shapes if shape.name == name]
    assert len(matches) == 1, (name, len(matches))
    return matches[0]


def _manifest_record(fixture, name):
    matches = [r for r in MANIFEST[fixture]["shapes"] if r["name"] == name]
    assert len(matches) == 1, (fixture, name, len(matches))
    return matches[0]


def _is_fallback(record):
    return record["alternate_content"] is not None and record["alternate_content"]["branch"] == "Fallback"


def _expected_records(fixture, slide_number):
    return [r for r in MANIFEST[fixture]["shapes"] if r["slide"] == slide_number and not _is_fallback(r)]


def _manifest_xfrm(raw):
    if raw is None:
        return Xfrm()
    off, ext = raw["off"], raw["ext"]
    ch_off = raw["chOff"] or {"x": "0", "y": "0"}
    ch_ext = raw["chExt"] or {"cx": "0", "cy": "0"}
    return Xfrm(
        off_x=int(off["x"]), off_y=int(off["y"]), ext_cx=int(ext["cx"]), ext_cy=int(ext["cy"]),
        rot=int(raw["rot"] or 0), flip_h=raw["flipH"], flip_v=raw["flipV"],
        ch_off_x=int(ch_off["x"]), ch_off_y=int(ch_off["y"]),
        ch_ext_cx=int(ch_ext["cx"]), ch_ext_cy=int(ch_ext["cy"]),
    )


def _manifest_ancestors(fixture, record):
    groups = {(r["slide"], r["id"]): r for r in MANIFEST[fixture]["shapes"] if r["tag"] == "grpSp"}
    chain = []
    parent = record["parent"]
    while parent is not None:
        group = groups[(record["slide"], parent)]
        chain.insert(0, _manifest_xfrm(group["xfrm"]))
        parent = group["parent"]
    return tuple(chain)


def _values(resolved):
    if isinstance(resolved, ResolvedConnector):
        return resolved.start, resolved.end, _values(resolved.box)
    return (resolved.corners, resolved.bbox, resolved.center, resolved.width, resolved.height,
            resolved.rotation_deg, resolved.mirrored)


def _close(actual, expected):
    if isinstance(expected, tuple):
        return len(actual) == len(expected) and all(_close(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, float):
        return math.isclose(actual, expected, rel_tol=0.0, abs_tol=TOLERANCE)
    return actual == expected


def _emu_box(off_x, off_y, ext_cx, ext_cy):
    return (off_x / EMU_PER_POINT, off_y / EMU_PER_POINT,
            (off_x + ext_cx) / EMU_PER_POINT, (off_y + ext_cy) / EMU_PER_POINT)


def test_slide_order_follows_presentation_list():
    prs = _open("slide_order.pptx")
    assert [s.number for s in prs.slides] == [1, 2, 3]
    with zipfile.ZipFile(PPTX_DIR / "slide_order.pptx") as z:
        pres = ET.fromstring(z.read("ppt/presentation.xml"))
    listed = [int(el.get("id")) for el in pres.iterfind(f"{{{NS_P}}}sldIdLst/{{{NS_P}}}sldId")]
    assert [s.slide_id for s in prs.slides] == listed
    assert [s.shapes[0].text for s in prs.slides] == ["SLIDE-C", "SLIDE-A", "SLIDE-B"]
    assert all(not s.hidden for s in prs.slides)


def test_hidden_slide_flag():
    prs = _open("hidden_slide.pptx")
    assert [s.hidden for s in prs.slides] == [False, True]
    assert [s.shapes[0].text for s in prs.slides] == ["VISIBLE-SLIDE", "HIDDEN-SLIDE"]


def test_slide_size_in_points_for_every_fixture():
    for fixture, record in MANIFEST.items():
        prs = _open(fixture)
        assert prs.width_pt == record["slide_width"] / EMU_PER_POINT, fixture
        assert prs.height_pt == record["slide_height"] / EMU_PER_POINT, fixture


def test_shapes_match_manifest_for_every_fixture():
    for fixture in MANIFEST:
        for slide in _open(fixture).slides:
            expected = [
                (r["id"], r["name"], r["tag"], r["parent"], r["xfrm"], r["alternate_content"], r["text"])
                for r in _expected_records(fixture, slide.number)
            ]
            actual = [
                (s.id, s.name, s.tag, s.parent_id, s.raw_xfrm, s.alternate_content, s.text)
                for s in slide.shapes
            ]
            assert actual == expected, (fixture, slide.number)
            assert [s.z_order for s in slide.shapes] == list(range(len(slide.shapes))), fixture


def test_geometry_matches_manifest_chain_for_every_fixture():
    checked = 0
    for fixture in MANIFEST:
        for slide in _open(fixture).slides:
            records = _expected_records(fixture, slide.number)
            assert len(slide.shapes) == len(records), (fixture, slide.number)
            for shape, record in zip(slide.shapes, records):
                if record["xfrm"] is None:
                    continue
                xfrm = _manifest_xfrm(record["xfrm"])
                if record["tag"] == "graphicFrame":
                    xfrm = dataclasses.replace(xfrm, rot=0, flip_h=False, flip_v=False)
                ancestors = _manifest_ancestors(fixture, record)
                if record["tag"] == "cxnSp":
                    expected = resolve_connector(xfrm, ancestors)
                else:
                    expected = resolve_shape(xfrm, ancestors)
                assert _close(_values(shape.geometry), _values(expected)), (fixture, record["name"])
                checked += 1
    assert checked == sum(
        1 for fixture in MANIFEST for r in MANIFEST[fixture]["shapes"] if not _is_fallback(r) and r["xfrm"]
    ), checked


def test_bbox_and_connector_endpoints_are_hand_computed():
    slide = _open("shapes_and_connectors.pptx").slides[0]
    box = _shape(slide, "Box A").geometry
    assert _close(box.bbox, _emu_box(600_000, 900_000, 2_000_000, 1_000_000))
    assert _close(box.width, 2_000_000 / EMU_PER_POINT) and _close(box.height, 1_000_000 / EMU_PER_POINT)
    connector = _shape(slide, "Connector 1 Both Ends").geometry
    assert isinstance(connector, ResolvedConnector)
    assert _close(connector.start, (2_600_000 / EMU_PER_POINT, 1_400_000 / EMU_PER_POINT))
    assert _close(connector.end, (4_000_000 / EMU_PER_POINT, 1_400_000 / EMU_PER_POINT))
    assert connector.box.rotation_deg == 0.0 and connector.box.mirrored is False
    table = _open("tables_merged.pptx").slides[0]
    frame = _shape(table, "Merged Table")
    assert frame.tag == "graphicFrame" and frame.text is None
    assert _close(frame.geometry.bbox, _emu_box(1_000_000, 1_000_000, 10_000_000, 4_000_000))


def test_flip_and_rotation_keep_raw_values_and_normalise_geometry():
    slide = _open("flip_rot.pptx").slides[0]
    flip_v = _shape(slide, "Flip V Only")
    assert flip_v.raw_xfrm["flipV"] == "1" and flip_v.raw_xfrm["flipH"] is None
    assert flip_v.geometry.mirrored is True and _close(flip_v.geometry.rotation_deg, 180.0)
    rot_flip_v = _shape(slide, "Rot 390 Flip V")
    assert rot_flip_v.raw_xfrm["rot"] == "23400000"
    assert rot_flip_v.geometry.mirrored is True and _close(rot_flip_v.geometry.rotation_deg, 210.0)
    rot_90 = _shape(slide, "Rot 90")
    assert rot_90.raw_xfrm["rot"] == "5400000" and _close(rot_90.geometry.rotation_deg, 90.0)


def test_placeholder_title_inherits_position_and_body_is_explicit():
    slide = _open("placeholders_inherit.pptx").slides[0]
    title = _shape(slide, "Title 1")
    assert title.xfrm_source == "inherited" and title.raw_xfrm is None
    inherited = _manifest_record("placeholders_inherit.pptx", "Title 1")["inherited_geometry"]
    expected = _emu_box(inherited["left"], inherited["top"], inherited["width"], inherited["height"])
    assert _close(title.geometry.bbox, expected)
    body = _shape(slide, "Content Placeholder 2")
    assert body.xfrm_source == "explicit" and body.raw_xfrm is not None
    assert _close(body.geometry.bbox, _emu_box(1_000_000, 2_000_000, 9_000_000, 3_500_000))


def test_alternate_content_keeps_choice_and_drops_fallback():
    slide = _open("alternate_content.pptx").slides[0]
    assert [s.name for s in slide.shapes] == ["Choice Shape", "Ordinary Box"]
    assert slide.shapes[0].alternate_content == {"branch": "Choice", "requires": "a14"}
    assert slide.shapes[1].alternate_content is None


def test_duplicate_ids_keep_every_shape_with_distinct_z_order():
    slide = _open("duplicate_ids.pptx").slides[0]
    assert [(s.id, s.name) for s in slide.shapes] == [
        (2, "Anchor Box"), (5, "Dup Box One"), (5, "Dup Box Two"), (6, "Dup Connector"),
    ]
    assert [s.z_order for s in slide.shapes] == [0, 1, 2, 3]


def test_nested_groups_parent_ids_and_preorder_z_order():
    slide = _open("nested_groups.pptx").slides[0]
    rows = [(s.name, s.id, s.parent_id, s.z_order) for s in slide.shapes]
    assert rows == [
        ("Outer Group", 2, None, 0),
        ("Inner Group", 3, 2, 1),
        ("Inner Box 1", 4, 3, 2),
        ("Inner Box 2", 5, 3, 3),
        ("Outer Child Box", 6, 2, 4),
        ("Outer Connector", 7, 2, 5),
        ("Top Box", 8, None, 6),
        ("Top Connector", 9, None, 7),
    ]


def test_extraction_is_deterministic():
    for fixture in MANIFEST:
        assert _open(fixture) == _open(fixture), fixture


def test_sp_and_cxnsp_stay_distinct():
    slide = _open("shapes_and_connectors.pptx").slides[0]
    sps = [s for s in slide.shapes if s.tag == "sp"]
    connectors = [s for s in slide.shapes if s.tag == "cxnSp"]
    assert len(sps) == 4 and len(connectors) == 7
    assert all(isinstance(s.geometry, ResolvedShape) for s in sps)
    assert all(isinstance(s.geometry, ResolvedConnector) for s in connectors)


def test_graphic_frame_ignores_rotation_and_flips_for_geometry_only():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rotated_table.pptx"
        with zipfile.ZipFile(PPTX_DIR / "tables_merged.pptx") as src, zipfile.ZipFile(path, "w") as dst:
            for info in src.infolist():
                data = src.read(info.filename)
                if info.filename == "ppt/slides/slide1.xml":
                    marked = data.replace(b"<p:xfrm>", b'<p:xfrm rot="5400000" flipH="1">', 1)
                    assert marked != data
                    data = marked
                dst.writestr(info, data)
        frame = extract_presentation(path).slides[0].shapes[0]
    assert frame.raw_xfrm["rot"] == "5400000" and frame.raw_xfrm["flipH"] == "1"
    assert frame.geometry.mirrored is False and _close(frame.geometry.rotation_deg, 0.0)
    assert _close(frame.geometry.bbox, _emu_box(1_000_000, 1_000_000, 10_000_000, 4_000_000))


def test_shape_without_xfrm_and_not_placeholder_has_no_geometry():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "no_xfrm.pptx"
        deck = pptx.Presentation()
        box = deck.slides.add_slide(deck.slide_layouts[6]).shapes.add_textbox(0, 0, 100, 100)
        box.text_frame.text = "loose"
        box._element.spPr.remove(box._element.spPr.find(f"{{{NS_A}}}xfrm"))
        deck.save(path)
        shape = extract_presentation(path).slides[0].shapes[0]
    assert shape.xfrm_source == "missing" and shape.raw_xfrm is None and shape.geometry is None
    assert shape.text == "loose"


def _fragment(xml):
    return etree.fromstring(f"<wrap {_DECLARATIONS}>{xml}</wrap>")[0]


def _write_deck(path, *fragments):
    deck = pptx.Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    sp_tree = slide._element.find(f"{{{NS_P}}}cSld/{{{NS_P}}}spTree")
    for xml in fragments:
        sp_tree.append(_fragment(xml))
    deck.save(path)


def _sp(shape_id, name, off=(0, 0), ext=(1_000_000, 1_000_000)):
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{shape_id}" name="{name}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
        f'<p:spPr><a:xfrm><a:off x="{off[0]}" y="{off[1]}"/><a:ext cx="{ext[0]}" cy="{ext[1]}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:sp>'
    )


def _pic(shape_id, name):
    return (
        f'<p:pic><p:nvPicPr><p:cNvPr id="{shape_id}" name="{name}"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>'
        '<p:blipFill><a:blip/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
        '<p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="1000000" cy="1000000"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'
    )


def _alternate(*branches):
    return f"<mc:AlternateContent>{''.join(branches)}</mc:AlternateContent>"


def _choice(requires, *children):
    return f'<mc:Choice Requires="{requires}">{"".join(children)}</mc:Choice>'


def _fallback(*children):
    return f"<mc:Fallback>{''.join(children)}</mc:Fallback>"


def _cxn(shape_id, name, start=None, end=None, off=(0, 0), ext=(1_000_000, 0)):
    refs = "".join(
        f'<a:{tag} id="{ref[0]}" idx="{ref[1]}"/>'
        for tag, ref in (("stCxn", start), ("endCxn", end))
        if ref is not None
    )
    return (
        f'<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="{shape_id}" name="{name}"/>'
        f"<p:cNvCxnSpPr>{refs}</p:cNvCxnSpPr><p:nvPr/></p:nvCxnSpPr>"
        f'<p:spPr><a:xfrm><a:off x="{off[0]}" y="{off[1]}"/><a:ext cx="{ext[0]}" cy="{ext[1]}"/></a:xfrm>'
        '<a:prstGeom prst="straightConnector1"><a:avLst/></a:prstGeom></p:spPr></p:cxnSp>'
    )


def _group(group_id, name, *children):
    return (
        f'<p:grpSp><p:nvGrpSpPr><p:cNvPr id="{group_id}" name="{name}"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="4000000" cy="4000000"/>'
        '<a:chOff x="0" y="0"/><a:chExt cx="4000000" cy="4000000"/></a:xfrm></p:grpSpPr>'
        + "".join(children) + "</p:grpSp>"
    )


def _manifest_box(fixture, name):
    xfrm = _manifest_record(fixture, name)["xfrm"]
    return int(xfrm["off"]["x"]), int(xfrm["off"]["y"]), int(xfrm["ext"]["cx"]), int(xfrm["ext"]["cy"])


def _prst_by_id(fixture):
    with zipfile.ZipFile(PPTX_DIR / fixture) as z:
        root = ET.fromstring(z.read("ppt/slides/slide1.xml"))
    prst = {}
    for sp in root.iter(f"{{{NS_P}}}sp"):
        cnv = sp.find(f"{{{NS_P}}}nvSpPr/{{{NS_P}}}cNvPr")
        geometry = sp.find(f"{{{NS_P}}}spPr/{{{NS_A}}}prstGeom")
        prst[int(cnv.get("id"))] = geometry.get("prst") if geometry is not None else None
    return prst


def _pair(connection):
    return None if connection is None else (connection.target_id, connection.site_index)


def _manifest_pair(raw):
    return None if raw is None else (int(raw["id"]), int(raw["idx"]))


def _ancestor_names(shapes, shape):
    by_id = {s.id: s for s in shapes}
    names = []
    parent = shape.parent_id
    while parent is not None:
        names.append(by_id[parent].name)
        parent = by_id[parent].parent_id
    return names


def test_only_first_recognised_choice_is_walked():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "two_choices.pptx"
        _write_deck(path, _alternate(
            _choice("a14", _sp(10, "First Choice")),
            _choice("p14", _sp(11, "Second Choice")),
            _fallback(_sp(12, "Fallback Shape")),
        ))
        shapes = extract_presentation(path).slides[0].shapes
    assert [s.name for s in shapes] == ["First Choice"], [s.name for s in shapes]
    assert shapes[0].alternate_content == {"branch": "Choice", "requires": "a14"}


def test_unrecognised_choice_falls_back_to_fallback_branch():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "choice_unknown.pptx"
        _write_deck(path, _alternate(_choice("p14", UNKNOWN), _fallback(_pic(10, "Fallback Picture"))))
        shapes = extract_presentation(path).slides[0].shapes
    assert [(s.name, s.tag) for s in shapes] == [("Fallback Picture", "pic")], shapes
    assert shapes[0].alternate_content == {"branch": "Fallback", "requires": None}


def test_alternate_content_without_recognised_branch_emits_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nothing_recognised.pptx"
        _write_deck(
            path,
            _sp(10, "Before"),
            _alternate(_choice("p14", UNKNOWN), _fallback(UNKNOWN)),
            _sp(11, "After"),
        )
        shapes = extract_presentation(path).slides[0].shapes
    assert [s.name for s in shapes] == ["Before", "After"], [s.name for s in shapes]
    assert all(s.alternate_content is None for s in shapes)


def test_alternate_content_inside_group_keeps_parent_and_geometry():
    group_xfrm = (
        '<p:grpSpPr><a:xfrm><a:off x="1000000" y="1000000"/><a:ext cx="2000000" cy="2000000"/>'
        '<a:chOff x="0" y="0"/><a:chExt cx="1000000" cy="1000000"/></a:xfrm></p:grpSpPr>'
    )
    inner = _alternate(_choice("a14", _sp(21, "Grouped Choice", off=(100000, 200000), ext=(300000, 400000))))
    group = (
        '<p:grpSp><p:nvGrpSpPr><p:cNvPr id="20" name="Group"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        + group_xfrm + inner + "</p:grpSp>"
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "grouped_alternate.pptx"
        _write_deck(path, group)
        shapes = extract_presentation(path).slides[0].shapes
    assert [(s.name, s.id, s.parent_id) for s in shapes] == [
        ("Group", 20, None),
        ("Grouped Choice", 21, 20),
    ], [(s.name, s.id, s.parent_id) for s in shapes]
    group_frame = Xfrm(off_x=1000000, off_y=1000000, ext_cx=2000000, ext_cy=2000000,
                       ch_ext_cx=1000000, ch_ext_cy=1000000)
    child_frame = Xfrm(off_x=100000, off_y=200000, ext_cx=300000, ext_cy=400000)
    expected = resolve_shape(child_frame, (group_frame,))
    assert _close(_values(shapes[1].geometry), _values(expected))


def test_connection_references_match_manifest_for_every_fixture():
    checked = 0
    for fixture in MANIFEST:
        for slide in _open(fixture).slides:
            records = _expected_records(fixture, slide.number)
            assert len(slide.shapes) == len(records), (fixture, slide.number)
            for shape, record in zip(slide.shapes, records):
                label = (fixture, record["name"])
                if record["tag"] != "cxnSp":
                    assert shape.start_connection is None and shape.end_connection is None, label
                assert _pair(shape.start_connection) == _manifest_pair(record["stCxn"]), label
                assert _pair(shape.end_connection) == _manifest_pair(record["endCxn"]), label
                checked += (record["stCxn"] is not None) + (record["endCxn"] is not None)
    assert checked == sum(
        (r["stCxn"] is not None) + (r["endCxn"] is not None)
        for fixture in MANIFEST for r in MANIFEST[fixture]["shapes"] if not _is_fallback(r)
    ), checked


def test_shapes_and_connectors_reference_statuses():
    slide = _open("shapes_and_connectors.pptx").slides[0]
    shapes = slide.shapes
    connector_one = _shape(slide, "Connector 1 Both Ends")
    assert connector_one.start_connection.status == "resolved"
    assert connector_one.end_connection.status == "resolved"
    assert shapes[connector_one.start_connection.target_z_order].id == 2
    assert shapes[connector_one.start_connection.target_z_order].name == "Box A"
    assert shapes[connector_one.end_connection.target_z_order].id == 3
    assert shapes[connector_one.end_connection.target_z_order].name == "Box B"
    start_only = _shape(slide, "Connector 2 Start Only")
    assert start_only.start_connection.status == "resolved"
    assert shapes[start_only.start_connection.target_z_order].name == "Box C"
    assert start_only.end_connection is None
    loose = _shape(slide, "Connector 3 Loose")
    assert loose.start_connection is None and loose.end_connection is None
    dangling = _shape(slide, "Connector 4 Dangling")
    assert dangling.start_connection.status == "dangling"
    assert dangling.start_connection.target_id == 999 and dangling.start_connection.target_z_order is None


def test_nested_group_connectors_resolve_inside_and_across_groups():
    slide = _open("nested_groups.pptx").slides[0]
    shapes = slide.shapes
    inner = _shape(slide, "Outer Connector")
    assert inner.start_connection.status == "resolved" and inner.end_connection.status == "resolved"
    start = shapes[inner.start_connection.target_z_order]
    end = shapes[inner.end_connection.target_z_order]
    assert (start.name, end.name) == ("Inner Box 1", "Outer Child Box")
    assert _ancestor_names(shapes, start) == ["Inner Group", "Outer Group"]
    assert end.parent_id == inner.parent_id and _ancestor_names(shapes, end) == ["Outer Group"]
    top = _shape(slide, "Top Connector")
    assert top.parent_id is None
    assert shapes[top.start_connection.target_z_order].name == "Top Box"
    deep = shapes[top.end_connection.target_z_order]
    assert deep.name == "Inner Box 2"
    assert _ancestor_names(shapes, deep) == ["Inner Group", "Outer Group"]


def test_duplicate_id_target_is_ambiguous_and_unique_target_resolves():
    slide = _open("duplicate_ids.pptx").slides[0]
    connector = _shape(slide, "Dup Connector")
    assert connector.start_connection.status == "ambiguous"
    assert connector.start_connection.target_id == 5 and connector.start_connection.target_z_order is None
    assert connector.end_connection.status == "resolved"
    assert slide.shapes[connector.end_connection.target_z_order].name == "Anchor Box"


def test_forward_group_and_invalid_references_resolve_without_raising():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "references.pptx"
        _write_deck(
            path,
            _cxn(10, "Forward Connector", end=(11, 1)),
            _sp(11, "Late Target"),
            _group(30, "Target Group", _sp(31, "Grouped Box")),
            _cxn(12, "Group Connector", start=(30, 0)),
            _cxn(13, "Invalid Connector", start=("abc", 1)),
        )
        shapes = extract_presentation(path).slides[0].shapes
    by_name = {s.name: s for s in shapes}
    forward = by_name["Forward Connector"].end_connection
    assert forward.status == "resolved" and shapes[forward.target_z_order].name == "Late Target"
    group = by_name["Group Connector"].start_connection
    assert group.status == "resolved" and shapes[group.target_z_order].tag == "grpSp"
    invalid = by_name["Invalid Connector"].start_connection
    assert invalid.target_id is None and invalid.target_z_order is None
    assert invalid.site_index == 1 and invalid.status == "invalid"


def test_flipped_connector_endpoints_follow_box_flips():
    slide = _open("shapes_and_connectors.pptx").slides[0]
    x, y, cx, cy = _manifest_box("shapes_and_connectors.pptx", "Connector 5 FlipV")
    flip_v = _shape(slide, "Connector 5 FlipV").geometry
    assert _close(flip_v.start, (x / EMU_PER_POINT, (y + cy) / EMU_PER_POINT))
    assert _close(flip_v.end, ((x + cx) / EMU_PER_POINT, y / EMU_PER_POINT))
    x, y, cx, cy = _manifest_box("shapes_and_connectors.pptx", "Connector 6 FlipH")
    flip_h = _shape(slide, "Connector 6 FlipH").geometry
    assert _close(flip_h.start, ((x + cx) / EMU_PER_POINT, y / EMU_PER_POINT))
    assert _close(flip_h.end, (x / EMU_PER_POINT, (y + cy) / EMU_PER_POINT))


def test_connector_one_matches_rect_connection_sites():
    fixture = "shapes_and_connectors.pptx"
    connector = _shape(_open(fixture).slides[0], "Connector 1 Both Ends").geometry
    prst = _prst_by_id(fixture)
    record = _manifest_record(fixture, "Connector 1 Both Ends")
    checked = []
    for point, reference in ((connector.start, record["stCxn"]), (connector.end, record["endCxn"])):
        target_id, site = int(reference["id"]), int(reference["idx"])
        if prst[target_id] != "rect":
            continue
        target = next(r for r in MANIFEST[fixture]["shapes"] if r["id"] == target_id)
        x, y, cx, cy = _manifest_box(fixture, target["name"])
        fx, fy = RECT_SITES[site]
        site_x = (x + fx * cx) / EMU_PER_POINT
        site_y = (y + fy * cy) / EMU_PER_POINT
        assert _close(point, (site_x, site_y)), (target["name"], site)
        checked.append(target["name"])
    assert checked == ["Box A", "Box B"], checked


def _table_frame(shape_id, name):
    return (
        f'<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="{shape_id}" name="{name}"/>'
        "<p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr>"
        '<p:xfrm><a:off x="0" y="0"/><a:ext cx="2000000" cy="500000"/></p:xfrm>'
        '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table"><a:tbl>'
        '<a:tblPr firstRow="1"/><a:tblGrid><a:gridCol w="1000000"/><a:gridCol w="1000000"/></a:tblGrid>'
        '<a:tr h="500000"><a:tc><a:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>A</a:t></a:r></a:p></a:txBody>'
        '<a:tcPr/></a:tc><a:tc><a:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>B</a:t></a:r></a:p></a:txBody>'
        "<a:tcPr/></a:tc></a:tr></a:tbl></a:graphicData></a:graphic></p:graphicFrame>"
    )


def _chart_frame(shape_id, name):
    return (
        f'<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="{shape_id}" name="{name}"/>'
        "<p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr>"
        '<p:xfrm><a:off x="0" y="0"/><a:ext cx="2000000" cy="500000"/></p:xfrm>'
        '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/chart"/></a:graphic>'
        "</p:graphicFrame>"
    )


def test_table_cells_match_manifest_origin_cells():
    table = _shape(_open("tables_merged.pptx").slides[0], "Merged Table").table
    manifest = MANIFEST["tables_merged.pptx"]["tables"][0]
    expected = tuple(
        (c["row"], c["col"], int(c["rowSpan"] or 1), int(c["gridSpan"] or 1), c["text"])
        for c in manifest["cells"]
        if not (c["hMerge"] or c["vMerge"])
    )
    assert tuple((c.row, c.col, c.rowspan, c.colspan, c.text) for c in table.cells) == expected
    assert table.num_rows == max(c["row"] for c in manifest["cells"]) + 1 == 4
    assert table.num_cols == 4 and table.first_row_header is True


def test_table_inside_group_is_read_and_other_frames_carry_no_table():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "grouped_table.pptx"
        _write_deck(
            path,
            _group(40, "Holder", _table_frame(41, "Grouped Table")),
            _chart_frame(42, "Chart"),
            _sp(43, "Plain"),
        )
        shapes = extract_presentation(path).slides[0].shapes
    by_name = {s.name: s for s in shapes}
    grouped = by_name["Grouped Table"].table
    assert grouped.num_rows == 1 and grouped.num_cols == 2 and grouped.first_row_header is True
    assert [(c.row, c.col, c.text) for c in grouped.cells] == [(0, 0, "A"), (0, 1, "B")]
    assert by_name["Chart"].table is None and by_name["Holder"].table is None and by_name["Plain"].table is None


def test_only_tables_merged_frame_carries_a_table_in_fixtures():
    for fixture in MANIFEST:
        for slide in _open(fixture).slides:
            for shape in slide.shapes:
                expected = fixture == "tables_merged.pptx" and shape.tag == "graphicFrame"
                assert (shape.table is not None) == expected, (fixture, shape.name)


def test_parent_z_order_points_at_enclosing_group():
    shapes = _open("nested_groups.pptx").slides[0].shapes
    assert [(s.name, s.parent_z_order) for s in shapes] == [
        ("Outer Group", None),
        ("Inner Group", 0),
        ("Inner Box 1", 1),
        ("Inner Box 2", 1),
        ("Outer Child Box", 0),
        ("Outer Connector", 0),
        ("Top Box", None),
        ("Top Connector", None),
    ]


def test_multi_paragraph_text_joins_with_newline():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "paragraphs.pptx"
        deck = pptx.Presentation()
        box = deck.slides.add_slide(deck.slide_layouts[6]).shapes.add_textbox(0, 0, 100, 100)
        box.text_frame.text = "first\nsecond"
        deck.save(path)
        slide = extract_presentation(path).slides[0]
    assert slide.shapes[0].text == "first\nsecond"


def test_unreadable_file_raises():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "broken.pptx"
        path.write_bytes(b"not a package")
        raised = False
        try:
            extract_presentation(path)
        except Exception:
            raised = True
    assert raised


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
