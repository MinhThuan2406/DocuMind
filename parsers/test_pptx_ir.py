import json
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pptx  # noqa: E402
from lxml import etree  # noqa: E402

from geometry import ResolvedConnector  # noqa: E402
from parsers.pptx_extractor import extract_presentation  # noqa: E402
from parsers.pptx_ir import extract_ir  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
PPTX_DIR = FIXTURE_DIR / "pptx"
MANIFEST = json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))["fixtures"]
NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
_DECLARATIONS = f'xmlns:a="{NS_A}" xmlns:p="{NS_P}"'
TOLERANCE = 1e-6


def _ir(fixture):
    return extract_ir(PPTX_DIR / fixture)


def _by_name(ir):
    return {e["metadata"]["shape_name"]: e for e in ir["elements"]}


def _is_fallback(record):
    return record["alternate_content"] is not None and record["alternate_content"]["branch"] == "Fallback"


def _expected_per_page(fixture):
    return Counter(r["slide"] for r in MANIFEST[fixture]["shapes"] if not _is_fallback(r))


def _depth(by_id, element):
    depth = 1
    while element["parent_id"] is not None and depth <= len(by_id):
        element = by_id[element["parent_id"]]
        depth += 1
    return depth


def _assert_plain(value):
    if isinstance(value, dict):
        assert all(isinstance(key, str) for key in value), value
        for item in value.values():
            _assert_plain(item)
    elif isinstance(value, list):
        for item in value:
            _assert_plain(item)
    else:
        assert value is None or isinstance(value, (str, int, float, bool)), (type(value), value)


def _geometry_box(shape):
    if shape.geometry is None:
        return None
    return shape.geometry.box if isinstance(shape.geometry, ResolvedConnector) else shape.geometry


def _fragment(xml):
    return etree.fromstring(f"<wrap {_DECLARATIONS}>{xml}</wrap>")[0]


def _write_deck(path, *fragments):
    deck = pptx.Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    sp_tree = slide._element.find(f"{{{NS_P}}}cSld/{{{NS_P}}}spTree")
    for xml in fragments:
        sp_tree.append(_fragment(xml))
    deck.save(path)


def _loose_sp(shape_id, name):
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{shape_id}" name="{name}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
        "<p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>loose</a:t></a:r></a:p></p:txBody></p:sp>"
    )


def _pic(shape_id, name):
    return (
        f'<p:pic><p:nvPicPr><p:cNvPr id="{shape_id}" name="{name}"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>'
        "<p:blipFill><a:blip/><a:stretch><a:fillRect/></a:stretch></p:blipFill>"
        '<p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="1000000" cy="1000000"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'
    )


def _chart(shape_id, name):
    return (
        f'<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="{shape_id}" name="{name}"/>'
        "<p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr>"
        '<p:xfrm><a:off x="0" y="0"/><a:ext cx="2000000" cy="500000"/></p:xfrm>'
        '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/chart"/></a:graphic>'
        "</p:graphicFrame>"
    )


def test_top_level_keys_source_file_and_json_round_trip():
    ir = _ir("shapes_and_connectors.pptx")
    assert set(ir) == {"doc_id", "source_file", "doc_type", "metadata", "elements", "tables", "diagrams"}
    assert ir["doc_id"] == "shapes_and_connectors" and ir["doc_type"] == "pptx"
    assert ir["source_file"] == "shapes_and_connectors.pptx" and ir["diagrams"] == []
    _assert_plain(ir)
    text = json.dumps(ir)
    assert json.loads(text) == ir
    for fixture in MANIFEST:
        text = json.dumps(_ir(fixture))
        for absolute in (str(PPTX_DIR), str(ROOT)):
            assert absolute not in text and json.dumps(absolute)[1:-1] not in text, (fixture, absolute)


def test_element_ids_pages_reading_order_parents_and_levels_for_every_fixture():
    for fixture in MANIFEST:
        ir = _ir(fixture)
        elements = ir["elements"]
        assert len(elements) == sum(_expected_per_page(fixture).values()), fixture
        assert [e["id"] for e in elements] == [f"elem_{n:04d}" for n in range(1, len(elements) + 1)], fixture
        assert Counter(e["page"] for e in elements) == _expected_per_page(fixture), fixture
        by_id = {e["id"]: e for e in elements}
        for page in {e["page"] for e in elements}:
            on_page = [e for e in elements if e["page"] == page]
            assert [e["reading_order"] for e in on_page] == list(range(len(on_page))), (fixture, page)
            assert [e["reading_order"] for e in on_page] == [e["metadata"]["z_order"] for e in on_page], fixture
        for element in elements:
            if element["parent_id"] is not None:
                parent = by_id[element["parent_id"]]
                assert parent["type"] == "group" and parent["page"] == element["page"], (fixture, element["id"])
            assert element["level"] == _depth(by_id, element), (fixture, element["id"])


def test_type_mapping_and_bbox_follow_native_tags():
    expected_types = {
        "shapes_and_connectors.pptx": ["shape"] * 4 + ["connector"] * 7,
        "nested_groups.pptx": ["group", "group", "shape", "shape", "shape", "connector", "shape", "connector"],
        "tables_merged.pptx": ["table"],
    }
    for fixture, types in expected_types.items():
        assert [e["type"] for e in _ir(fixture)["elements"]] == types, fixture
    for fixture in MANIFEST:
        elements = _ir(fixture)["elements"]
        shapes = [s for slide in extract_presentation(PPTX_DIR / fixture).slides for s in slide.shapes]
        for element, shape in zip(elements, shapes):
            box = _geometry_box(shape)
            assert element["metadata"]["shape_name"] == shape.name, (fixture, shape.name)
            assert element["bbox"] == (None if box is None else list(box.bbox)), (fixture, shape.name)


def test_element_without_xfrm_has_null_bbox_and_pic_and_chart_types():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "loose.pptx"
        _write_deck(path, _loose_sp(10, "Loose"), _pic(11, "Picture"), _chart(12, "Chart"))
        ir = extract_ir(path)
    loose, picture, chart = ir["elements"]
    assert (loose["type"], loose["bbox"], loose["text"]) == ("shape", None, "loose")
    assert loose["metadata"]["xfrm_source"] == "missing" and loose["metadata"]["rotation_deg"] is None
    assert loose["metadata"]["mirrored"] is None and loose["source_ref"]["bbox"] is None
    assert picture["type"] == "picture" and picture["bbox"] is not None
    assert chart["type"] == "graphic_frame" and "table_id" not in chart
    assert ir["tables"] == []


def test_connector_endpoints_and_target_element_ids_for_every_fixture():
    statuses = set()
    for fixture in MANIFEST:
        ir = _ir(fixture)
        presentation = extract_presentation(PPTX_DIR / fixture)
        connectors = [e for e in ir["elements"] if e["type"] == "connector"]
        shapes = [s for slide in presentation.slides for s in slide.shapes if s.tag == "cxnSp"]
        assert len(connectors) == len(shapes), fixture
        for element, shape in zip(connectors, shapes):
            geometry = shape.geometry
            assert element["metadata"]["start"] == (None if geometry is None else list(geometry.start))
            assert element["metadata"]["end"] == (None if geometry is None else list(geometry.end))
            for side in ("start_connection", "end_connection"):
                connection = element["metadata"][side]
                raw = getattr(shape, side)
                if raw is None:
                    assert connection is None, (fixture, shape.name, side)
                    continue
                statuses.add(connection["status"])
                assert connection["status"] == raw.status and connection["target_id"] == raw.target_id
                if connection["status"] == "resolved":
                    matches = [
                        e for e in ir["elements"]
                        if e["page"] == element["page"] and e["source_ref"]["shape_id"] == connection["target_id"]
                    ]
                    assert len(matches) == 1, (fixture, shape.name)
                    assert connection["target_element_id"] == matches[0]["id"], (fixture, shape.name)
                else:
                    assert connection["target_element_id"] is None, (fixture, shape.name, side)
    assert {"resolved", "dangling", "ambiguous"} <= statuses, statuses


def test_nested_group_connectors_resolve_inside_and_two_levels_deep():
    ir = _ir("nested_groups.pptx")
    by_id = {e["id"]: e for e in ir["elements"]}
    by_name = _by_name(ir)
    top = by_name["Top Connector"]
    assert top["level"] == 1 and top["parent_id"] is None
    deep = by_id[top["metadata"]["end_connection"]["target_element_id"]]
    assert deep["metadata"]["shape_name"] == "Inner Box 2" and deep["level"] == 3
    assert _depth(by_id, deep) == 3
    inner = by_name["Outer Connector"]
    start = by_id[inner["metadata"]["start_connection"]["target_element_id"]]
    end = by_id[inner["metadata"]["end_connection"]["target_element_id"]]
    assert (start["metadata"]["shape_name"], end["metadata"]["shape_name"]) == ("Inner Box 1", "Outer Child Box")


def test_raw_flip_flags_survive_next_to_mirrored_geometry():
    flip = _by_name(_ir("flip_rot.pptx"))["Flip V Only"]
    assert flip["metadata"]["raw_xfrm"]["flipV"] == "1" and flip["metadata"]["raw_xfrm"]["flipH"] is None
    assert flip["metadata"]["mirrored"] is True
    assert abs(flip["metadata"]["rotation_deg"] - 180.0) < TOLERANCE


def test_tables_merged_cells_spans_headers_and_cross_references():
    ir = _ir("tables_merged.pptx")
    table = ir["tables"][0]
    element = ir["elements"][0]
    manifest = MANIFEST["tables_merged.pptx"]["tables"][0]
    expected = [
        (c["row"], c["col"], int(c["rowSpan"] or 1), int(c["gridSpan"] or 1), c["text"])
        for c in manifest["cells"]
        if not (c["hMerge"] or c["vMerge"])
    ]
    assert [(c["row"], c["col"], c["rowspan"], c["colspan"], c["text"]) for c in table["cells"]] == expected
    assert [(c["row"], c["col"]) for c in table["cells"] if c["is_header"]] == [(0, 0), (0, 2), (0, 3)]
    assert [(c["row"], c["col"]) for c in table["cells"] if c["is_merged"]] == [(0, 0), (1, 3), (2, 0)]
    assert (table["id"], table["num_rows"], table["num_cols"]) == ("tbl_0001", 4, 4)
    assert element["type"] == "table" and element["text"] == "[Table tbl_0001: 4x4]"
    assert element["table_id"] == table["id"] == element["metadata"]["table_id"] == "tbl_0001"
    assert table["metadata"] == {"element_id": element["id"]}
    assert table["bbox"] == element["bbox"] and table["source_ref"]["shape_id"] == element["metadata"]["shape_id"]


def test_slide_order_and_hidden_flags_follow_presentation_order():
    ir = _ir("slide_order.pptx")
    with zipfile.ZipFile(PPTX_DIR / "slide_order.pptx") as z:
        pres = ET.fromstring(z.read("ppt/presentation.xml"))
    listed = [int(el.get("id")) for el in pres.iterfind(f"{{{NS_P}}}sldIdLst/{{{NS_P}}}sldId")]
    assert [s["slide_id"] for s in ir["metadata"]["slides"]] == listed
    assert [s["number"] for s in ir["metadata"]["slides"]] == [1, 2, 3]
    assert [(e["page"], e["text"]) for e in ir["elements"]] == [(1, "SLIDE-C"), (2, "SLIDE-A"), (3, "SLIDE-B")]
    ir = _ir("hidden_slide.pptx")
    assert [s["hidden"] for s in ir["metadata"]["slides"]] == [False, True]
    assert [(e["page"], e["text"]) for e in ir["elements"]] == [(1, "VISIBLE-SLIDE"), (2, "HIDDEN-SLIDE")]


def test_extraction_is_deterministic_for_every_fixture():
    for fixture in MANIFEST:
        first, second = _ir(fixture), _ir(fixture)
        assert first == second, fixture
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True), fixture


def test_alternate_content_only_selected_branch_produces_element():
    ir = _ir("alternate_content.pptx")
    assert [e["metadata"]["shape_name"] for e in ir["elements"]] == ["Choice Shape", "Ordinary Box"]
    assert ir["elements"][0]["metadata"]["alternate_content"] == {"branch": "Choice", "requires": "a14"}
    assert ir["elements"][1]["metadata"]["alternate_content"] is None


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
