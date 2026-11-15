import functools
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from lxml import etree  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE_TYPE  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent
PPTX_DIR = FIXTURE_DIR / "pptx"
FIXTURES = json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))["fixtures"]

NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"
NS_A14 = "http://schemas.microsoft.com/office/drawing/2010/main"
NS_CP = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
NS_DC = "http://purl.org/dc/elements/1.1/"
NS_DCTERMS = "http://purl.org/dc/terms/"

SLIDE_W = 12192000
SLIDE_H = 6858000
SHAPE_TAGS = {"sp", "cxnSp", "grpSp", "pic", "graphicFrame"}
CORE_KEYS = ("slide", "part", "id", "name", "tag", "text", "xfrm", "parent",
             "stCxn", "endCxn", "alternate_content")


def q(ns: str, name: str) -> str:
    return f"{{{ns}}}{name}"


def local(el: etree._Element) -> str:
    return etree.QName(el).localname


@functools.cache
def zip_slides(fname: str) -> list[tuple[int, str, etree._Element]]:
    with zipfile.ZipFile(PPTX_DIR / fname) as z:
        pres = etree.fromstring(z.read("ppt/presentation.xml"))
        rels = etree.fromstring(z.read("ppt/_rels/presentation.xml.rels"))
        target = {rel.get("Id"): rel.get("Target") for rel in rels}
        slides = []
        for number, sld_id in enumerate(pres.iterfind(q(NS_P, "sldIdLst") + "/" + q(NS_P, "sldId")), start=1):
            part = "ppt/" + target[sld_id.get(q(NS_R, "id"))]
            slides.append((number, part.rsplit("/", 1)[-1], etree.fromstring(z.read(part))))
    return slides


def _read_xfrm(el: etree._Element) -> etree._Element | None:
    tag = local(el)
    if tag == "graphicFrame":
        return el.find(q(NS_P, "xfrm"))
    if tag == "grpSp":
        return el.find(q(NS_P, "grpSpPr") + "/" + q(NS_A, "xfrm"))
    return el.find(q(NS_P, "spPr") + "/" + q(NS_A, "xfrm"))


def _pair(node: etree._Element | None, first: str, second: str) -> dict | None:
    if node is None:
        return None
    return {first: node.get(first), second: node.get(second)}


def _xfrm_record(el: etree._Element) -> dict | None:
    xfrm = _read_xfrm(el)
    if xfrm is None:
        return None
    return {
        "off": _pair(xfrm.find(q(NS_A, "off")), "x", "y"),
        "ext": _pair(xfrm.find(q(NS_A, "ext")), "cx", "cy"),
        "chOff": _pair(xfrm.find(q(NS_A, "chOff")), "x", "y"),
        "chExt": _pair(xfrm.find(q(NS_A, "chExt")), "cx", "cy"),
        "rot": xfrm.get("rot"),
        "flipH": xfrm.get("flipH"),
        "flipV": xfrm.get("flipV"),
    }


def _cxn_ref(el: etree._Element, side: str) -> dict | None:
    if local(el) != "cxnSp":
        return None
    cnv_cxn = el.find(q(NS_P, "nvCxnSpPr") + "/" + q(NS_P, "cNvCxnSpPr"))
    node = cnv_cxn.find(q(NS_A, side))
    return None if node is None else {"id": node.get("id"), "idx": node.get("idx")}


def _text(el: etree._Element) -> str | None:
    if local(el) != "sp":
        return None
    body = el.find(q(NS_P, "txBody"))
    if body is None:
        return None
    return "".join(t.text or "" for t in body.iter(q(NS_A, "t")))


def _describe(el: etree._Element, slide_no: int, part: str, parent_id: int | None, alt: dict | None) -> dict:
    cnv = el[0][0]
    return {
        "slide": slide_no,
        "part": part,
        "id": int(cnv.get("id")),
        "name": cnv.get("name"),
        "tag": local(el),
        "text": _text(el),
        "xfrm": _xfrm_record(el),
        "parent": parent_id,
        "stCxn": _cxn_ref(el, "stCxn"),
        "endCxn": _cxn_ref(el, "endCxn"),
        "alternate_content": alt,
    }


def _walk(container: etree._Element, slide_no: int, part: str, parent_id: int | None,
          alt: dict | None) -> list[dict]:
    records = []
    for child in container:
        if child.tag == q(NS_MC, "AlternateContent"):
            for branch in child:
                branch_alt = {"branch": local(branch), "requires": branch.get("Requires")}
                records.extend(_walk(branch, slide_no, part, parent_id, branch_alt))
        elif child.tag.startswith("{%s}" % NS_P) and local(child) in SHAPE_TAGS:
            rec = _describe(child, slide_no, part, parent_id, alt)
            records.append(rec)
            if local(child) == "grpSp":
                records.extend(_walk(child, slide_no, part, rec["id"], alt))
    return records


def derived_records(fname: str) -> list[dict]:
    records = []
    for slide_no, part, root in zip_slides(fname):
        sp_tree = root.find(q(NS_P, "cSld") + "/" + q(NS_P, "spTree"))
        records.extend(_walk(sp_tree, slide_no, part, None, None))
    return records


def check_shapes(fname: str) -> None:
    expected = FIXTURES[fname]["shapes"]
    actual = derived_records(fname)
    got_names = [r["name"] for r in actual]
    want_names = [r["name"] for r in expected]
    assert got_names == want_names, f"{fname}: names xml={got_names} manifest={want_names}"
    for exp, act in zip(expected, actual):
        for key in CORE_KEYS:
            assert act[key] == exp[key], \
                f"{fname} / {exp['name']} / {key}: xml={act[key]!r} manifest={exp[key]!r}"


def _slide_text(root: etree._Element) -> str:
    return "".join(t.text or "" for t in root.iter(q(NS_A, "t")))


def _manifest_shape(fname: str, name: str) -> dict:
    matches = [r for r in FIXTURES[fname]["shapes"] if r["name"] == name]
    assert len(matches) == 1, (fname, name, len(matches))
    return matches[0]


def _xml_shape(fname: str, name: str) -> dict:
    matches = [r for r in derived_records(fname) if r["name"] == name]
    assert len(matches) == 1, (fname, name, len(matches))
    return matches[0]


def _pptx_walk(shapes, out: list) -> list:
    for shp in shapes:
        out.append(shp)
        if shp.shape_type == MSO_SHAPE_TYPE.GROUP:
            _pptx_walk(shp.shapes, out)
    return out


def test_fixture_files_exist_and_stay_small():
    for name in FIXTURES:
        path = PPTX_DIR / name
        assert path.is_file(), path
        assert path.stat().st_size < 100 * 1024, (name, path.stat().st_size)


def test_manifest_covers_every_pptx_on_disk():
    on_disk = sorted(p.name for p in PPTX_DIR.glob("*.pptx"))
    assert on_disk == sorted(FIXTURES), (on_disk, sorted(FIXTURES))


def test_slide_size_is_16_by_9_without_type_attribute():
    for fname, fixture in FIXTURES.items():
        with zipfile.ZipFile(PPTX_DIR / fname) as z:
            pres = etree.fromstring(z.read("ppt/presentation.xml"))
        sz = pres.find(q(NS_P, "sldSz"))
        assert sz.get("cx") == str(SLIDE_W) and sz.get("cy") == str(SLIDE_H), (fname, dict(sz.attrib))
        assert "type" not in sz.attrib, (fname, dict(sz.attrib))
        assert fixture["slide_width"] == SLIDE_W and fixture["slide_height"] == SLIDE_H


def test_core_properties_are_fixed_for_every_fixture():
    for fname in FIXTURES:
        with zipfile.ZipFile(PPTX_DIR / fname) as z:
            core = etree.fromstring(z.read("docProps/core.xml"))
        assert core.findtext(q(NS_DCTERMS, "created")) == "2026-01-01T00:00:00Z", fname
        assert core.findtext(q(NS_DCTERMS, "modified")) == "2026-01-01T00:00:00Z", fname
        assert core.findtext(q(NS_DC, "creator")) == "fixture-builder", fname
        assert core.findtext(q(NS_CP, "lastModifiedBy")) == "fixture-builder", fname


def test_names_unique_per_slide_and_top_level_boxes_inside_slide():
    for fname, fixture in FIXTURES.items():
        recs = derived_records(fname)
        for n in {r["slide"] for r in recs}:
            names = [r["name"] for r in recs if r["slide"] == n]
            assert len(names) == len(set(names)), (fname, n, names)
        for r in recs:
            if r["parent"] is None and r["xfrm"] is not None:
                x, y = int(r["xfrm"]["off"]["x"]), int(r["xfrm"]["off"]["y"])
                cx, cy = int(r["xfrm"]["ext"]["cx"]), int(r["xfrm"]["ext"]["cy"])
                assert 0 <= x and 0 <= y and x + cx <= SLIDE_W and y + cy <= SLIDE_H, (fname, r["name"])
        assert len(fixture["slides"]) == len(zip_slides(fname)), fname


def test_slide_order_follows_sldIdLst_not_part_names():
    got = [(n, part, _slide_text(root)) for n, part, root in zip_slides("slide_order.pptx")]
    expected = [(s["presentation_index"], s["part"], s["marker"])
                for s in FIXTURES["slide_order.pptx"]["slides"]]
    assert got == expected, (got, expected)
    assert got == [(1, "slide3.xml", "SLIDE-C"), (2, "slide1.xml", "SLIDE-A"),
                   (3, "slide2.xml", "SLIDE-B")], got
    check_shapes("slide_order.pptx")


def test_every_slide_marker_occurs_as_text_on_its_slide():
    for fname, fixture in FIXTURES.items():
        for (n, part, root), manifest_slide in zip(zip_slides(fname), fixture["slides"]):
            assert manifest_slide["marker"] is not None, (fname, n)
            assert manifest_slide["marker"] in _slide_text(root), (fname, n, manifest_slide["marker"])


def test_slide_order_python_pptx_reads_presentation_order():
    prs = Presentation(PPTX_DIR / "slide_order.pptx")
    markers = [slide.shapes[0].text_frame.text for slide in prs.slides]
    assert markers == ["SLIDE-C", "SLIDE-A", "SLIDE-B"], markers


def test_hidden_slide_has_show_zero_on_second_slide():
    got = [(n, root.get("show")) for n, _, root in zip_slides("hidden_slide.pptx")]
    expected = [(s["presentation_index"], s["show"]) for s in FIXTURES["hidden_slide.pptx"]["slides"]]
    assert got == expected == [(1, None), (2, "0")], got
    prs = Presentation(PPTX_DIR / "hidden_slide.pptx")
    assert prs.slides[0]._element.get("show") is None
    assert prs.slides[1]._element.get("show") == "0"
    check_shapes("hidden_slide.pptx")


def test_shapes_and_connectors_refs_flips_and_rotation():
    check_shapes("shapes_and_connectors.pptx")
    c1 = _xml_shape("shapes_and_connectors.pptx", "Connector 1 Both Ends")
    assert c1["stCxn"] == {"id": "2", "idx": "3"} and c1["endCxn"] == {"id": "3", "idx": "1"}
    c2 = _xml_shape("shapes_and_connectors.pptx", "Connector 2 Start Only")
    assert c2["stCxn"] == {"id": "4", "idx": "2"} and c2["endCxn"] is None
    c3 = _xml_shape("shapes_and_connectors.pptx", "Connector 3 Loose")
    assert c3["stCxn"] is None and c3["endCxn"] is None
    c4 = _xml_shape("shapes_and_connectors.pptx", "Connector 4 Dangling")
    assert c4["stCxn"] == {"id": "999", "idx": "0"}
    ids = {r["id"] for r in derived_records("shapes_and_connectors.pptx")}
    assert 999 not in ids, "dangling reference must point at a missing id"
    c5 = _xml_shape("shapes_and_connectors.pptx", "Connector 5 FlipV")
    assert c5["xfrm"]["flipV"] == "1" and c5["xfrm"]["flipH"] is None and c5["xfrm"]["rot"] is None
    c6 = _xml_shape("shapes_and_connectors.pptx", "Connector 6 FlipH")
    assert c6["xfrm"]["flipH"] == "1" and c6["xfrm"]["flipV"] is None and c6["xfrm"]["rot"] is None
    c7 = _xml_shape("shapes_and_connectors.pptx", "Connector 7 Rot90")
    assert c7["xfrm"]["rot"] == "5400000" and c7["xfrm"]["flipH"] is None and c7["xfrm"]["flipV"] is None


def test_nested_groups_frames_parents_and_depth_two_reference():
    check_shapes("nested_groups.pptx")
    outer = _xml_shape("nested_groups.pptx", "Outer Group")
    assert outer["xfrm"]["ext"] == {"cx": "6000000", "cy": "3000000"}
    assert outer["xfrm"]["chExt"] == {"cx": "3000000", "cy": "3000000"}
    assert outer["xfrm"]["chOff"] == {"x": "500000", "y": "1000000"}
    inner = _xml_shape("nested_groups.pptx", "Inner Group")
    assert inner["xfrm"]["rot"] == "2700000" and inner["parent"] == outer["id"]
    by_id = {r["id"]: r for r in derived_records("nested_groups.pptx")}
    end_id = int(_xml_shape("nested_groups.pptx", "Top Connector")["endCxn"]["id"])
    target = by_id[end_id]
    assert target["name"] == "Inner Box 2"
    assert by_id[target["parent"]]["name"] == "Inner Group"
    assert by_id[by_id[target["parent"]]["parent"]]["name"] == "Outer Group"
    prs = Presentation(PPTX_DIR / "nested_groups.pptx")
    top = [s.name for s in prs.slides[0].shapes]
    assert top == ["Outer Group", "Top Box", "Top Connector"], top
    grp = next(s for s in prs.slides[0].shapes if s.name == "Outer Group")
    assert [s.name for s in grp.shapes] == ["Inner Group", "Outer Child Box", "Outer Connector"]
    inner_py = next(s for s in grp.shapes if s.name == "Inner Group")
    assert [s.name for s in inner_py.shapes] == ["Inner Box 1", "Inner Box 2"]


def test_flip_and_rotation_raw_attributes():
    check_shapes("flip_rot.pptx")
    xf = {r["name"]: r["xfrm"] for r in derived_records("flip_rot.pptx")}
    assert (xf["Flip V Only"]["flipV"], xf["Flip V Only"]["flipH"], xf["Flip V Only"]["rot"]) == ("1", None, None)
    assert (xf["Flip H Only"]["flipH"], xf["Flip H Only"]["flipV"], xf["Flip H Only"]["rot"]) == ("1", None, None)
    assert (xf["Flip Both"]["flipH"], xf["Flip Both"]["flipV"]) == ("1", "1")
    assert (xf["Rot 90"]["rot"], xf["Rot 90"]["flipH"], xf["Rot 90"]["flipV"]) == ("5400000", None, None)
    assert (xf["Rot 45 Flip H"]["rot"], xf["Rot 45 Flip H"]["flipH"]) == ("2700000", "1")
    assert (xf["Rot 390 Flip V"]["rot"], xf["Rot 390 Flip V"]["flipV"]) == ("23400000", "1")


def test_tables_merged_cell_attributes():
    check_shapes("tables_merged.pptx")
    expected = FIXTURES["tables_merged.pptx"]["tables"][0]["cells"]
    root = zip_slides("tables_merged.pptx")[0][2]
    tbl = root.find(".//" + q(NS_A, "tbl"))
    got = []
    for r, tr in enumerate(tbl.findall(q(NS_A, "tr"))):
        for c, tc in enumerate(tr.findall(q(NS_A, "tc"))):
            got.append({
                "row": r, "col": c,
                "text": "".join(t.text or "" for t in tc.iter(q(NS_A, "t"))),
                "gridSpan": tc.get("gridSpan"), "rowSpan": tc.get("rowSpan"),
                "hMerge": tc.get("hMerge"), "vMerge": tc.get("vMerge"),
            })
    assert got == expected, [(g, e) for g, e in zip(got, expected) if g != e]
    assert len({c["text"] for c in got}) == 16, "every cell must carry unique text"
    prs = Presentation(PPTX_DIR / "tables_merged.pptx")
    table = next(s for s in prs.slides[0].shapes if s.has_table).table
    assert table.cell(0, 0).is_merge_origin and table.cell(0, 0).span_width == 2
    assert table.cell(0, 1).is_spanned
    assert table.cell(1, 3).is_merge_origin and table.cell(1, 3).span_height == 2
    assert table.cell(2, 0).is_merge_origin
    assert table.cell(2, 0).span_width == 2 and table.cell(2, 0).span_height == 2
    assert table.cell(3, 0).is_spanned and table.cell(3, 1).is_spanned


def test_placeholders_inherit_title_and_override_body():
    check_shapes("placeholders_inherit.pptx")
    root = zip_slides("placeholders_inherit.pptx")[0][2]
    sps = {el[0][0].get("name"): el for el in root.iter(q(NS_P, "sp"))}
    title_el, body_el = sps["Title 1"], sps["Content Placeholder 2"]
    assert title_el.find(q(NS_P, "spPr")).find(q(NS_A, "xfrm")) is None
    assert body_el.find(q(NS_P, "spPr")).find(q(NS_A, "xfrm")) is not None
    title_m = _manifest_shape("placeholders_inherit.pptx", "Title 1")
    body_m = _manifest_shape("placeholders_inherit.pptx", "Content Placeholder 2")
    assert title_m["xfrm"] is None and body_m["xfrm"] is not None
    ph_title = title_el.find(q(NS_P, "nvSpPr") + "/" + q(NS_P, "nvPr") + "/" + q(NS_P, "ph"))
    ph_body = body_el.find(q(NS_P, "nvSpPr") + "/" + q(NS_P, "nvPr") + "/" + q(NS_P, "ph"))
    assert {"type": ph_title.get("type"), "idx": ph_title.get("idx")} == title_m["placeholder"]
    assert {"type": ph_body.get("type"), "idx": ph_body.get("idx")} == body_m["placeholder"]
    prs = Presentation(PPTX_DIR / "placeholders_inherit.pptx")
    slide = prs.slides[0]
    assert slide.slide_layout.name == "Title and Content"
    title_py = next(s for s in slide.shapes if s.name == "Title 1")
    inherited = title_m["inherited_geometry"]
    assert (title_py.left, title_py.top, title_py.width, title_py.height) == \
        (inherited["left"], inherited["top"], inherited["width"], inherited["height"])
    body_py = next(s for s in slide.shapes if s.name == "Content Placeholder 2")
    off, ext = body_m["xfrm"]["off"], body_m["xfrm"]["ext"]
    assert (body_py.left, body_py.top, body_py.width, body_py.height) == \
        (int(off["x"]), int(off["y"]), int(ext["cx"]), int(ext["cy"]))


def test_alternate_content_structure():
    check_shapes("alternate_content.pptx")
    root = zip_slides("alternate_content.pptx")[0][2]
    assert root.nsmap.get("mc") == NS_MC and root.nsmap.get("a14") == NS_A14, root.nsmap
    sp_tree = root.find(q(NS_P, "cSld") + "/" + q(NS_P, "spTree"))
    wrappers = sp_tree.findall(q(NS_MC, "AlternateContent"))
    assert len(wrappers) == 1, len(wrappers)
    ac = wrappers[0]
    choice = ac.find(q(NS_MC, "Choice"))
    fallback = ac.find(q(NS_MC, "Fallback"))
    assert choice.get("Requires") == "a14"
    choice_sp = choice.find(q(NS_P, "sp"))
    fallback_sp = fallback.find(q(NS_P, "sp"))
    assert _text(choice_sp) == "CHOICE-SHAPE" and _text(fallback_sp) == "FALLBACK-SHAPE"
    assert choice_sp[0][0].get("id") != fallback_sp[0][0].get("id")
    ordinary = [el for el in sp_tree if el.tag == q(NS_P, "sp")]
    assert [_text(el) for el in ordinary] == ["ORDINARY-BOX"]
    assert _manifest_shape("alternate_content.pptx", "Choice Shape")["alternate_content"] == \
        {"branch": "Choice", "requires": "a14"}
    assert _manifest_shape("alternate_content.pptx", "Fallback Shape")["alternate_content"] == \
        {"branch": "Fallback", "requires": None}
    prs = Presentation(PPTX_DIR / "alternate_content.pptx")
    visible = [s.name for s in prs.slides[0].shapes]
    assert visible == ["Ordinary Box"], visible
    with zipfile.ZipFile(PPTX_DIR / "alternate_content.pptx") as z:
        assert b"ns0:" not in z.read("ppt/slides/slide1.xml")


def test_duplicate_ids_are_ambiguous_and_flagged():
    check_shapes("duplicate_ids.pptx")
    counts = {}
    for r in derived_records("duplicate_ids.pptx"):
        counts[r["id"]] = counts.get(r["id"], 0) + 1
    assert sorted(i for i, n in counts.items() if n > 1) == [5]
    conn = _xml_shape("duplicate_ids.pptx", "Dup Connector")
    assert conn["stCxn"] == {"id": "5", "idx": "1"}
    flagged = sorted(r["name"] for r in FIXTURES["duplicate_ids.pptx"]["shapes"]
                     if r.get("ambiguous_id") is True)
    assert flagged == ["Dup Box One", "Dup Box Two", "Dup Connector"], flagged


def test_python_pptx_loads_every_fixture_and_reads_manifest_values():
    for fname, fixture in FIXTURES.items():
        prs = Presentation(PPTX_DIR / fname)
        assert len(prs.slides) == len(fixture["slides"]), fname
        for slide_no, slide in enumerate(prs.slides, start=1):
            seen = _pptx_walk(slide.shapes, [])
            expected = [r for r in fixture["shapes"]
                        if r["slide"] == slide_no and r["alternate_content"] is None]
            assert [(s.shape_id, s.name) for s in seen] == [(r["id"], r["name"]) for r in expected], \
                (fname, slide_no)
            for shp, rec in zip(seen, expected):
                if rec["xfrm"] is None:
                    continue
                off, ext = rec["xfrm"]["off"], rec["xfrm"]["ext"]
                got = (shp.left, shp.top, shp.width, shp.height)
                want = (int(off["x"]), int(off["y"]), int(ext["cx"]), int(ext["cy"]))
                assert got == want, (fname, rec["name"], got, want)


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
