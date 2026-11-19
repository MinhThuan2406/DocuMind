import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdf.hierarchy import _parse_caption_prefix, apply_structure, build_section_tree  # noqa: E402
from pdf.models import SemanticRegion  # noqa: E402


def _elem(
    elem_id: str,
    *,
    kind: str = "text",
    page: int = 1,
    bbox: list[float] | None = None,
    reading_order: int = 0,
    level: int = 1,
    parent_id: str | None = None,
    text: str = "",
    source: str = "text_layer",
    font_size: float | None = 11.0,
    bold: bool | None = False,
    layout_label: str | None = None,
    role: str | None = None,
) -> dict:
    meta: dict = {}
    if font_size is not None:
        meta["font_size"] = font_size
    if bold is not None:
        meta["bold"] = bold
    if layout_label is not None:
        meta["layout_label"] = layout_label
    if role is not None:
        meta["role"] = role
    return {
        "id": elem_id,
        "type": kind,
        "page": page,
        "bbox": bbox or [50.0, 50.0 + reading_order * 40.0, 500.0, 75.0 + reading_order * 40.0],
        "reading_order": reading_order,
        "level": level,
        "parent_id": parent_id,
        "text": text,
        "source": source,
        "confidence": 1.0,
        "source_ref": {"file": "synthetic.pdf", "page": page, "bbox": bbox or [50.0, 50.0, 500.0, 75.0]},
        "metadata": meta,
    }


def _ir(elements: list[dict]) -> dict:
    return {
        "doc_id": "synthetic",
        "source_file": "synthetic.pdf",
        "doc_type": "pdf",
        "metadata": {"coord_unit": "pt", "pages": [{"number": 1}, {"number": 2}]},
        "elements": elements,
        "tables": [],
        "diagrams": [],
    }


def test_native_numbered_and_font_size_hierarchy_across_pages():
    long_body = (
        "This is a detailed body paragraph describing the system architecture and how modules "
        "interact across execution contexts during normal startup and shutdown sequences."
    )
    bold_prose = (
        "This bold paragraph explains two distinct phases of initialization. "
        "It continues into a second sentence so it must remain body text rather than a heading."
    )
    elements = [
        _elem("e1", page=1, reading_order=0, text="1. Introduction", font_size=18.0, bold=True),
        _elem("e2", page=1, reading_order=1, text=long_body, font_size=11.0, bold=False),
        _elem("e3", page=1, reading_order=2, text="1.1 System Scope", font_size=14.0, bold=True),
        _elem("e4", page=1, reading_order=3, text="1.5 ms latency was measured in tests.", font_size=11.0, bold=False),
        _elem("e5", page=1, reading_order=4, text="1.1.1 Functional Constraints", font_size=12.0, bold=True),
        _elem("e6", page=1, reading_order=5, text=bold_prose, font_size=11.0, bold=True),
        _elem("e7", page=2, reading_order=0, text="...", font_size=14.0, bold=True),
        _elem("e8", page=2, reading_order=1, text="Figure 2.1 shows an overview of the bus.", font_size=11.0, bold=False),
        _elem("e9", page=2, reading_order=2, text="2. Architecture", font_size=18.0, bold=True),
        _elem("e10", page=2, reading_order=3, text="Appendix A Glossary", font_size=16.0, bold=True),
        _elem("e11", page=2, reading_order=4, text="A.1 Terms", font_size=13.0, bold=True),
        _elem("e12", page=2, reading_order=5, text="Body under glossary terms.", font_size=11.0, bold=False),
    ]
    raw = _ir(elements)
    enriched = apply_structure(raw)

    assert "outline" not in raw["metadata"]
    assert "role" not in raw["elements"][0]["metadata"]
    assert json.loads(json.dumps(enriched)) == enriched

    by_id = {e["id"]: e for e in enriched["elements"]}
    assert by_id["e1"]["metadata"]["role"] == "heading"
    assert by_id["e1"]["metadata"]["heading_level"] == 1
    assert by_id["e1"]["metadata"]["section_parent_id"] is None

    assert by_id["e2"]["metadata"]["section_parent_id"] == "e1"
    assert by_id["e3"]["metadata"]["heading_level"] == 2
    assert by_id["e3"]["metadata"]["section_parent_id"] == "e1"
    assert by_id["e4"]["metadata"]["section_parent_id"] == "e3"
    assert by_id["e4"]["metadata"].get("role") is None

    assert by_id["e5"]["metadata"]["heading_level"] == 3
    assert by_id["e5"]["metadata"]["section_parent_id"] == "e3"
    assert by_id["e6"]["metadata"]["section_parent_id"] == "e5"
    assert by_id["e6"]["metadata"].get("role") is None
    assert by_id["e7"]["metadata"]["section_parent_id"] == "e5"
    assert by_id["e8"]["metadata"]["section_parent_id"] == "e5"
    assert by_id["e8"]["metadata"].get("role") is None

    assert by_id["e9"]["metadata"]["heading_level"] == 1
    assert by_id["e9"]["metadata"]["section_parent_id"] is None
    assert by_id["e10"]["metadata"]["heading_level"] == 1
    assert by_id["e11"]["metadata"]["heading_level"] == 2
    assert by_id["e11"]["metadata"]["section_parent_id"] == "e10"
    assert by_id["e12"]["metadata"]["section_parent_id"] == "e11"

    tree = build_section_tree(enriched["elements"])
    assert [n["id"] for n in tree] == ["e1", "e9", "e10"]
    assert [c["id"] for c in tree[0]["children"]] == ["e3"]
    assert [c["id"] for c in tree[0]["children"][0]["children"]] == ["e5"]
    assert tree[0]["children"][0]["children"][0]["element_ids"] == ["e6", "e7", "e8"]
    assert [c["id"] for c in tree[2]["children"]] == ["e11"]


def test_unnumbered_font_size_style_ranking_and_inplace():
    elements = [
        _elem("h1", page=1, reading_order=0, text="Overview", font_size=20.0, bold=True),
        _elem("b1", page=1, reading_order=1, text="General overview text spanning enough characters to dominate.", font_size=10.0, bold=False),
        _elem("h2", page=1, reading_order=2, text="Design Goals", font_size=14.0, bold=True),
        _elem("h3", page=1, reading_order=3, text="Latency Budget", font_size=10.0, bold=True),
        _elem("b2", page=1, reading_order=4, text="Target latency is below 5 ms for all requests in steady state.", font_size=10.0, bold=False),
    ]
    ir = _ir(elements)
    out = apply_structure(ir, inplace=True)
    assert out is ir
    by_id = {e["id"]: e for e in out["elements"]}
    assert by_id["h1"]["metadata"]["heading_level"] == 1
    assert by_id["h2"]["metadata"]["heading_level"] == 2
    assert by_id["h3"]["metadata"]["heading_level"] == 3
    assert by_id["h3"]["metadata"]["section_parent_id"] == "h2"
    assert by_id["b2"]["metadata"]["section_parent_id"] == "h3"

    # Idempotency and empty IR checks.
    second = apply_structure(out)
    assert second == out
    empty_out = apply_structure(_ir([]))
    assert empty_out["metadata"]["outline"] == []
    assert build_section_tree(empty_out["elements"]) == []


def test_chapter_heading_vs_bare_cross_reference():
    elements = [
        _elem("c1", page=1, reading_order=0, text="Chapter 1: Getting Started", font_size=14.0, bold=False),
        _elem("b1", page=1, reading_order=1, text="Chapter 8", font_size=11.0, bold=False),
        _elem("b2", page=1, reading_order=2, text="Reference body paragraph of sufficient length for median.", font_size=11.0, bold=False),
    ]
    out = apply_structure(_ir(elements))
    by_id = {e["id"]: e for e in out["elements"]}
    assert by_id["c1"]["metadata"]["role"] == "heading"
    assert by_id["c1"]["metadata"]["heading_level"] == 1
    assert by_id["b1"]["metadata"].get("role") is None
    assert by_id["b1"]["metadata"]["section_parent_id"] == "c1"


def test_skips_running_headers_footers_and_level2_children():
    elements = [
        _elem("hdr", page=1, reading_order=0, text="Document ID 56", font_size=16.0, bold=True, role="page_header"),
        _elem("h1", page=1, reading_order=1, text="1. Architecture", font_size=16.0, bold=True),
        _elem("fig1", kind="figure", page=1, bbox=[60.0, 120.0, 300.0, 260.0], reading_order=2, font_size=None, bold=None),
        _elem(
            "fig_child",
            page=1,
            bbox=[80.0, 150.0, 200.0, 170.0],
            reading_order=3,
            level=2,
            parent_id="fig1",
            text="1.1 Component Node",
            font_size=16.0,
            bold=True,
        ),
        _elem("ftr", page=1, reading_order=4, text="Page 1 of 10", font_size=14.0, bold=True, layout_label="page_footer"),
    ]
    out = apply_structure(_ir(elements))
    by_id = {e["id"]: e for e in out["elements"]}

    assert by_id["hdr"]["metadata"]["role"] == "page_header"
    assert "heading_level" not in by_id["hdr"]["metadata"]
    assert "section_parent_id" not in by_id["hdr"]["metadata"]

    assert by_id["ftr"]["metadata"]["layout_label"] == "page_footer"
    assert "heading_level" not in by_id["ftr"]["metadata"]
    assert "section_parent_id" not in by_id["ftr"]["metadata"]

    assert by_id["fig_child"]["level"] == 2
    assert by_id["fig_child"]["parent_id"] == "fig1"
    assert "role" not in by_id["fig_child"]["metadata"]
    assert "heading_level" not in by_id["fig_child"]["metadata"]
    assert "section_parent_id" not in by_id["fig_child"]["metadata"]

    assert by_id["fig1"]["level"] == 1
    assert by_id["fig1"]["parent_id"] is None
    assert by_id["fig1"]["metadata"]["section_parent_id"] == "h1"


def test_raster_layout_labels_title_section_header_and_caption():
    elements = [
        _elem(
            "r_title",
            page=1,
            bbox=[50.0, 40.0, 400.0, 70.0],
            reading_order=0,
            text="System Specification",
            source="ocr",
            font_size=0.0,
            bold=False,
            layout_label="title",
        ),
        _elem(
            "r_sec",
            page=1,
            bbox=[50.0, 90.0, 300.0, 110.0],
            reading_order=1,
            text="Hardware Interface",
            source="ocr",
            font_size=0.0,
            bold=False,
            layout_label="section_header",
        ),
        _elem(
            "r_fig",
            kind="figure",
            page=1,
            bbox=[60.0, 130.0, 320.0, 260.0],
            reading_order=2,
            source="layout_picture",
            font_size=None,
            bold=None,
        ),
        _elem(
            "r_cap",
            page=1,
            bbox=[60.0, 268.0, 320.0, 286.0],
            reading_order=3,
            text="Block diagram of hardware bus",
            source="ocr",
            font_size=0.0,
            bold=False,
            layout_label="caption",
        ),
    ]
    out = apply_structure(_ir(elements))
    by_id = {e["id"]: e for e in out["elements"]}

    assert by_id["r_title"]["metadata"]["role"] == "heading"
    assert by_id["r_title"]["metadata"]["heading_level"] == 1
    assert by_id["r_sec"]["metadata"]["role"] == "heading"
    assert by_id["r_sec"]["metadata"]["heading_level"] == 2
    assert by_id["r_sec"]["metadata"]["section_parent_id"] == "r_title"

    assert by_id["r_cap"]["metadata"]["role"] == "caption"
    assert by_id["r_cap"]["metadata"]["caption_of"] == "r_fig"
    assert by_id["r_cap"]["metadata"]["section_parent_id"] == "r_sec"
    assert by_id["r_fig"]["metadata"]["caption_id"] == "r_cap"
    assert by_id["r_fig"]["metadata"]["caption_text"] == "Block diagram of hardware bus"


def test_caption_linking_above_below_kind_match_and_columns():
    elements = [
        # Table with caption above; a Figure is placed slightly closer below to verify kind matching.
        _elem("cap_tbl", page=1, bbox=[50.0, 75.0, 260.0, 92.0], reading_order=0, text="Table 1: Pin Configuration", bold=True),
        _elem("tbl1", kind="table", page=1, bbox=[50.0, 100.0, 260.0, 180.0], reading_order=1, font_size=None, bold=None),
        # Figure with caption below, plus right-column figure & caption on the same row.
        _elem("fig_left", kind="figure", page=1, bbox=[50.0, 220.0, 240.0, 320.0], reading_order=2, font_size=None, bold=None),
        _elem("cap_fig_left", page=1, bbox=[50.0, 328.0, 240.0, 344.0], reading_order=3, text="Fig. 1.1: Left Signal Flow", bold=True),
        _elem("fig_right", kind="figure", page=1, bbox=[320.0, 220.0, 520.0, 320.0], reading_order=4, font_size=None, bold=None),
        _elem("cap_fig_right", page=1, bbox=[320.0, 328.0, 520.0, 344.0], reading_order=5, text="Figure 1.2 - Right Signal Flow", bold=True),
        # Caption placed between a table (closer) and a figure (slightly further) must still match Figure -> figure.
        _elem("tbl2", kind="table", page=2, bbox=[60.0, 100.0, 300.0, 180.0], reading_order=0, font_size=None, bold=None),
        _elem("cap_mid", page=2, bbox=[60.0, 186.0, 300.0, 202.0], reading_order=1, text="Figure 2.1: Layered Stack", bold=True),
        _elem("fig2", kind="figure", page=2, bbox=[60.0, 215.0, 300.0, 340.0], reading_order=2, font_size=None, bold=None),
        # Unlabeled short adjacent caption directly below a table without explicit caption.
        _elem("cap_adj", page=2, bbox=[60.0, 82.0, 280.0, 96.0], reading_order=3, text="Summary of timing limits", font_size=10.0, bold=False),
    ]
    out = apply_structure(_ir(elements))
    by_id = {e["id"]: e for e in out["elements"]}

    assert by_id["cap_tbl"]["metadata"]["caption_of"] == "tbl1"
    assert by_id["tbl1"]["metadata"]["caption_id"] == "cap_tbl"
    assert by_id["tbl1"]["metadata"]["caption_text"] == "Table 1: Pin Configuration"

    assert by_id["cap_fig_left"]["metadata"]["caption_of"] == "fig_left"
    assert by_id["fig_left"]["metadata"]["caption_id"] == "cap_fig_left"
    assert by_id["cap_fig_right"]["metadata"]["caption_of"] == "fig_right"
    assert by_id["fig_right"]["metadata"]["caption_id"] == "cap_fig_right"

    assert by_id["cap_mid"]["metadata"]["caption_of"] == "fig2"
    assert by_id["fig2"]["metadata"]["caption_id"] == "cap_mid"

    assert by_id["cap_adj"]["metadata"]["role"] == "caption"
    assert by_id["cap_adj"]["metadata"]["caption_of"] == "tbl2"
    assert by_id["tbl2"]["metadata"]["caption_id"] == "cap_adj"

    # Captions must not be promoted to headings even when bold.
    for cid in ("cap_tbl", "cap_fig_left", "cap_fig_right", "cap_mid", "cap_adj"):
        assert by_id[cid]["metadata"]["role"] == "caption"
        assert "heading_level" not in by_id[cid]["metadata"]


def test_landscape_slide_titles_group_in_slide_headings():
    body = "Body text of the slide that is long enough to dominate the weighted median font size."
    elements = [
        _elem("t1", page=1, bbox=[60.0, 14.0, 300.0, 40.0], reading_order=0, text="Introduction", font_size=16.0, bold=True),
        _elem("s1", page=1, bbox=[60.0, 76.0, 300.0, 96.0], reading_order=1, text="Purpose", font_size=16.0, bold=True),
        _elem("b1", page=1, bbox=[60.0, 120.0, 700.0, 140.0], reading_order=2, text=body, font_size=16.0, bold=False),
        _elem("s2", page=1, bbox=[60.0, 300.0, 300.0, 320.0], reading_order=3, text="Inputs", font_size=16.0, bold=True),
        _elem("t2", page=2, bbox=[60.0, 14.0, 300.0, 40.0], reading_order=0, text="Architecture", font_size=16.0, bold=True),
        _elem("b2", page=2, bbox=[60.0, 120.0, 700.0, 140.0], reading_order=1, text=body, font_size=16.0, bold=False),
    ]
    ir = _ir(elements)
    ir["metadata"]["pages"] = [
        {"number": 1, "width_pt": 810.0, "height_pt": 540.0},
        {"number": 2, "width_pt": 810.0, "height_pt": 540.0},
    ]
    by_id = {e["id"]: e for e in apply_structure(ir)["elements"]}

    assert by_id["t1"]["metadata"]["heading_level"] == 1
    assert by_id["s1"]["metadata"]["heading_level"] == 2
    assert by_id["s1"]["metadata"]["section_parent_id"] == "t1"
    assert by_id["s2"]["metadata"]["section_parent_id"] == "t1"
    assert by_id["b1"]["metadata"]["section_parent_id"] == "s1"
    assert by_id["t2"]["metadata"]["heading_level"] == 1
    assert by_id["t2"]["metadata"]["section_parent_id"] is None
    assert by_id["b2"]["metadata"]["section_parent_id"] == "t2"

    portrait = _ir([dict(e, metadata=dict(e["metadata"])) for e in elements])
    portrait["metadata"]["pages"] = [{"number": 1, "width_pt": 540.0, "height_pt": 810.0}, {"number": 2, "width_pt": 540.0, "height_pt": 810.0}]
    levels = {e["id"]: e["metadata"].get("heading_level") for e in apply_structure(portrait)["elements"]}
    assert levels["t1"] == levels["s1"] == levels["t2"] == 1


def test_slide_title_is_not_stolen_as_caption_and_may_be_long():
    long_title = "Interfaces Interaction of Layers Example Memory Closer Look at the stack and its modules"
    body = "Body text of the slide that is long enough to dominate the weighted median font size."
    elements = [
        _elem("t1", page=1, bbox=[21.0, 6.0, 400.0, 50.0], reading_order=0, text=long_title, font_size=16.0, bold=True),
        _elem("f1", kind="figure", page=1, bbox=[30.0, 56.0, 700.0, 400.0], reading_order=1, font_size=None, bold=None),
        _elem("b1", page=1, bbox=[60.0, 420.0, 700.0, 440.0], reading_order=2, text=body, font_size=16.0, bold=False),
    ]
    ir = _ir(elements)
    ir["metadata"]["pages"] = [{"number": 1, "width_pt": 810.0, "height_pt": 540.0}, {"number": 2}]
    by_id = {e["id"]: e for e in apply_structure(ir)["elements"]}

    assert len(long_title.split()) > 12
    assert by_id["t1"]["metadata"]["role"] == "heading"
    assert by_id["t1"]["metadata"]["heading_level"] == 1
    assert "caption_id" not in by_id["f1"]["metadata"]
    assert by_id["b1"]["metadata"]["section_parent_id"] == "t1"

    portrait = _ir([dict(e, metadata=dict(e["metadata"])) for e in elements])
    portrait["metadata"]["pages"] = [{"number": 1, "width_pt": 540.0, "height_pt": 810.0}]
    assert apply_structure(portrait)["elements"][0]["metadata"].get("role") != "heading"


def test_toc_bookmarks_suppress_contents_and_ignore_slide_bookmarks():
    elements = [
        _elem("toc_hdr", page=1, reading_order=0, text="Contents", font_size=16.0, bold=True),
        _elem("toc_line", page=1, reading_order=1, text="1. Overview", font_size=11.0, bold=False),
        _elem("h1", page=2, reading_order=0, text="1. Overview", font_size=11.0, bold=False),
        _elem("b1", page=2, reading_order=1, text="Detailed body paragraph explaining the overview section in full.", font_size=11.0, bold=False),
        _elem("h2", page=2, reading_order=2, text="Subtopic Without Number", font_size=11.0, bold=False),
    ]
    toc = [(1, "1. Overview", 2), (2, "Subtopic Without Number", 2)]
    out = apply_structure(_ir(elements), toc=toc)
    by_id = {e["id"]: e for e in out["elements"]}
    assert by_id["toc_hdr"]["metadata"].get("role") is None
    assert by_id["toc_line"]["metadata"].get("role") is None
    assert by_id["h1"]["metadata"]["heading_level"] == 1
    assert by_id["h2"]["metadata"]["heading_level"] == 2
    assert by_id["h2"]["metadata"]["section_parent_id"] == "h1"

    # Slide-deck Folie bookmarks must be ignored so landscape top-band logic still applies.
    slide_elems = [
        _elem("st1", page=1, bbox=[40.0, 20.0, 400.0, 50.0], reading_order=0, text="Introduction", font_size=16.0, bold=True),
        _elem("ss1", page=1, bbox=[40.0, 140.0, 400.0, 170.0], reading_order=1, text="Motivation", font_size=16.0, bold=True),
        _elem("sb1", page=1, bbox=[40.0, 200.0, 700.0, 230.0], reading_order=2, text="Detailed slide explanation text repeated to set body font size.", font_size=16.0, bold=False),
    ]
    slide_ir = _ir(slide_elems)
    slide_ir["metadata"]["pages"] = [{"number": 1, "width_pt": 810.0, "height_pt": 540.0}]
    slide_toc = [(1, "Folie 1", 1), (1, "Folie 2: Introduction", 2)]
    s_by_id = {e["id"]: e for e in apply_structure(slide_ir, toc=slide_toc)["elements"]}
    assert s_by_id["st1"]["metadata"]["heading_level"] == 1
    assert s_by_id["ss1"]["metadata"]["heading_level"] == 2
    assert s_by_id["ss1"]["metadata"]["section_parent_id"] == "st1"


def test_relative_size_tolerance_and_raster_landscape_child_level():
    elements = [
        _elem("body", page=1, reading_order=0, text="Regular body text repeated to anchor median font size accurately. " * 4, font_size=10.0, bold=False),
        _elem("h1a", page=1, reading_order=1, text="First Main Topic", font_size=20.0, bold=True),
        _elem("h1b", page=1, reading_order=2, text="Second Main Topic", font_size=19.4, bold=True),
        _elem("h2", page=1, reading_order=3, text="Sub Topic", font_size=14.0, bold=True),
    ]
    by_id = {e["id"]: e for e in apply_structure(_ir(elements))["elements"]}
    assert by_id["h1a"]["metadata"]["heading_level"] == 1
    assert by_id["h1b"]["metadata"]["heading_level"] == 1
    assert by_id["h2"]["metadata"]["heading_level"] == 2

    # Unstyled raster landscape section_header below slide_title must nest at level 2.
    raster_slide = [
        _elem("rt", page=1, bbox=[40.0, 20.0, 400.0, 50.0], reading_order=0, text="Slide Title", source="ocr", font_size=0.0, bold=False, layout_label="section_header"),
        _elem("rs", page=1, bbox=[40.0, 150.0, 400.0, 180.0], reading_order=1, text="Sub Header", source="ocr", font_size=0.0, bold=False, layout_label="section_header"),
    ]
    r_ir = _ir(raster_slide)
    r_ir["metadata"]["pages"] = [{"number": 1, "width_pt": 810.0, "height_pt": 540.0}]
    r_by_id = {e["id"]: e for e in apply_structure(r_ir)["elements"]}
    assert r_by_id["rt"]["metadata"]["heading_level"] == 1
    assert r_by_id["rs"]["metadata"]["heading_level"] == 2
    assert r_by_id["rs"]["metadata"]["section_parent_id"] == "rt"


def test_autosar_hierarchy_integration():
    vfb = ROOT / "datasets" / "AUTOSAR" / "AUTOSAR_CP_EXP_VFB.pdf"
    if not vfb.exists():
        print("SKIP AUTOSAR PDF not present")
        return

    from pdf.ir import extract_ir

    ir = extract_ir(vfb, figures=False, ocr=False)
    enriched = apply_structure(ir)
    outline = enriched["metadata"]["outline"]
    assert len(outline) >= 10
    assert any("The Virtual Functional Bus" in item["title"] and item["level"] == 1 for item in outline)
    assert any(item["title"] == "1.4.2 Specification Items" and item["level"] == 3 for item in outline)
    captions = [e for e in enriched["elements"] if e["metadata"].get("role") == "caption"]
    assert any("Figure 1.2: Structure of the document" in c["text"] for c in captions)


def test_rejects_requirement_ids_and_colon_labels():
    elements = [
        _elem("sec", page=1, reading_order=0, text="7.12.2 Requirements", font_size=12.0, bold=True),
        _elem("req1", page=1, reading_order=1, text="[SWS_Os_00001]", font_size=12.0, bold=True),
        _elem("body1", page=1, reading_order=2, text="The Operating System module shall initialize internal tables at startup.", font_size=11.0, bold=False),
        _elem("req2", page=1, reading_order=3, text="[ECUC_Os_00396] Definition of EcucModuleDef Os", font_size=12.0, bold=True),
        _elem("lbl1", page=1, reading_order=4, text="General:", font_size=12.0, bold=True),
        _elem("lbl2", page=1, reading_order=5, text="Parameters:", font_size=12.0, bold=True),
        _elem("lbl3", page=1, reading_order=6, text="Returned values:", font_size=12.0, bold=True),
        _elem("body2", page=1, reading_order=7, text="Status code returned to caller after execution.", font_size=11.0, bold=False),
    ]
    out = apply_structure(_ir(elements))
    by_id = {e["id"]: e for e in out["elements"]}

    assert by_id["sec"]["metadata"]["role"] == "heading"
    for bad_id in ("req1", "req2", "lbl1", "lbl2", "lbl3"):
        assert by_id[bad_id]["metadata"].get("role") is None
        assert "heading_level" not in by_id[bad_id]["metadata"]
        assert by_id[bad_id]["metadata"]["section_parent_id"] == "sec"
    assert by_id["body1"]["metadata"]["section_parent_id"] == "sec"
    assert by_id["body2"]["metadata"]["section_parent_id"] == "sec"


def test_caps_excess_unnumbered_styles_at_level_six():
    sizes = [28.0, 24.0, 22.0, 20.0, 18.0, 16.0, 15.0, 14.0, 12.0]
    elements = [
        _elem("body", page=1, reading_order=0, text="Regular body paragraph text repeated to anchor median body font size accurately. " * 4, font_size=10.0, bold=False),
    ]
    for idx, fs in enumerate(sizes, start=1):
        elements.append(
            _elem(f"h{idx}", page=1, reading_order=idx, text=f"Section Tier {idx}", font_size=fs, bold=(idx % 2 == 1))
        )
    out = apply_structure(_ir(elements))
    levels = [e["metadata"]["heading_level"] for e in out["elements"] if e["id"].startswith("h")]
    assert all(1 <= lv <= 6 for lv in levels)
    assert max(levels) == 6 and min(levels) == 1
    assert levels == sorted(levels)


def test_rejects_decorative_leaders_and_uml_multiplicity_and_preserves_common_tiers():
    elements = [
        _elem("body", page=1, reading_order=0, text="Regular body paragraph text repeated to anchor median body font size accurately. " * 6, font_size=14.0, bold=False),
        _elem("dots", page=1, reading_order=1, text=".............. Interface", font_size=28.0, bold=True),
        _elem("uml", page=1, reading_order=2, text="Runnable 0..*", font_size=18.0, bold=False),
        _elem("cover1", page=1, reading_order=3, text="Cover Banner", font_size=24.0, bold=False),
        _elem("cover2", page=1, reading_order=4, text="Cover Subtitle", font_size=22.0, bold=False),
        _elem("rare1", page=1, reading_order=5, text="Rare Callout A", font_size=20.5, bold=False),
        _elem("rare2", page=1, reading_order=6, text="Rare Callout B", font_size=20.0, bold=True),
        _elem("rare3", page=1, reading_order=7, text="Rare Mid Label", font_size=15.0, bold=True),
    ]
    ro = 8
    for i in range(6):
        elements.append(_elem(f"t{i}", page=2, reading_order=ro, text=f"Slide Title {i}", font_size=16.0, bold=True))
        ro += 1
        elements.append(_elem(f"s{i}", page=2, reading_order=ro, text=f"Slide Subhead {i}", font_size=14.0, bold=True))
        ro += 1
    for i in range(6):
        elements.append(_elem(f"m{i}", page=2, reading_order=ro, text=f"Diagram Group {i}", font_size=18.0, bold=False))
        ro += 1

    out = apply_structure(_ir(elements))
    by_id = {e["id"]: e for e in out["elements"]}
    assert by_id["dots"]["metadata"].get("role") is None
    assert by_id["uml"]["metadata"].get("role") is None
    # Common 16pt slide titles must remain strictly higher-level than common 14pt slide subheads.
    assert by_id["t0"]["metadata"]["heading_level"] < by_id["s0"]["metadata"]["heading_level"]
    assert 1 <= by_id["s0"]["metadata"]["heading_level"] <= 6


def test_semantic_regions_projection_on_native_elements():
    elements = [
        _elem("hdr", page=1, bbox=[50.0, 15.0, 300.0, 35.0], reading_order=0, text="Running Header", font_size=12.0, bold=True),
        _elem("t1", page=1, bbox=[50.0, 60.0, 350.0, 90.0], reading_order=1, text="Document Title", font_size=11.0, bold=False),
        _elem("s1", page=1, bbox=[50.0, 110.0, 300.0, 130.0], reading_order=2, text="System Overview", font_size=11.0, bold=False),
        _elem("fig", kind="figure", page=1, bbox=[60.0, 150.0, 300.0, 260.0], reading_order=3, font_size=None, bold=None),
        _elem("child_txt", page=1, bbox=[80.0, 180.0, 200.0, 200.0], reading_order=4, level=2, parent_id="fig", text="Node A", font_size=10.0, bold=False),
        _elem("cap", page=1, bbox=[60.0, 270.0, 300.0, 288.0], reading_order=5, text="Hardware layout sketch", font_size=11.0, bold=False),
        _elem("low_ov", page=1, bbox=[50.0, 320.0, 450.0, 340.0], reading_order=6, text="Body paragraph with minor region overlap across the tail.", font_size=11.0, bold=False),
    ]
    regions = {
        1: [
            SemanticRegion((45.0, 10.0, 310.0, 40.0), "page_header", 0.95),
            SemanticRegion((45.0, 55.0, 360.0, 95.0), "section_header", 0.70),
            SemanticRegion((45.0, 55.0, 360.0, 95.0), "title", 0.92),
            SemanticRegion((45.0, 105.0, 310.0, 135.0), "section_header", 0.88),
            SemanticRegion((75.0, 175.0, 210.0, 205.0), "title", 0.99),
            SemanticRegion((55.0, 268.0, 305.0, 290.0), "caption", 0.91),
            SemanticRegion((350.0, 320.0, 450.0, 340.0), "section_header", 0.99),
        ]
    }
    out = apply_structure(_ir(elements), semantic_regions=regions)
    by_id = {e["id"]: e for e in out["elements"]}

    assert by_id["hdr"]["metadata"]["layout_label"] == "page_header"
    assert "role" not in by_id["hdr"]["metadata"]
    assert "heading_level" not in by_id["hdr"]["metadata"]
    assert by_id["t1"]["metadata"]["layout_label"] == "title"
    assert by_id["t1"]["metadata"]["heading_level"] == 1
    assert by_id["s1"]["metadata"]["layout_label"] == "section_header"
    assert by_id["s1"]["metadata"]["heading_level"] == 2
    assert "layout_label" not in by_id["child_txt"]["metadata"]
    assert by_id["cap"]["metadata"]["layout_label"] == "caption"
    assert by_id["cap"]["metadata"]["caption_of"] == "fig"
    assert by_id["fig"]["metadata"]["caption_id"] == "cap"
    assert "layout_label" not in by_id["low_ov"]["metadata"]
    assert by_id["low_ov"]["metadata"].get("role") is None


def test_multiline_heading_and_caption_merging():
    elements = [
        _elem("h1a", page=1, bbox=[50.0, 50.0, 420.0, 70.0], reading_order=0, text="1.1 System Archi-", font_size=14.0, bold=True),
        _elem("h1b", page=1, bbox=[50.0, 74.0, 280.0, 94.0], reading_order=1, text="tecture and Interfaces", font_size=14.0, bold=True),
        _elem("b1", page=1, bbox=[50.0, 110.0, 500.0, 140.0], reading_order=2, text="Body paragraph describing the system interfaces in detail across multiple lines.", font_size=11.0, bold=False),
        _elem("h2", page=1, bbox=[50.0, 160.0, 300.0, 180.0], reading_order=3, text="1.2 Signal Flow", font_size=14.0, bold=True),
        _elem("h3", page=1, bbox=[50.0, 184.0, 300.0, 204.0], reading_order=4, text="1.3 Timing Rules", font_size=14.0, bold=True),
        _elem("fig1", kind="figure", page=1, bbox=[60.0, 220.0, 360.0, 340.0], reading_order=5, font_size=None, bold=None),
        _elem("c1a", page=1, bbox=[60.0, 346.0, 350.0, 360.0], reading_order=6, text="Figure 1.1: Overview of the", font_size=10.0, bold=True),
        _elem("c1b", page=1, bbox=[60.0, 363.0, 280.0, 377.0], reading_order=7, text="layered communication bus", font_size=10.0, bold=True),
        _elem("b2", page=1, bbox=[60.0, 381.0, 500.0, 410.0], reading_order=8, text="Figure 1.1 shows the layered communication bus. It connects modules.", font_size=11.0, bold=False),
    ]
    toc = [(2, "1.1 System Architecture and Interfaces", 1), (2, "1.2 Signal Flow", 1), (2, "1.3 Timing Rules", 1)]
    out = apply_structure(_ir(elements), toc=toc)
    by_id = {e["id"]: e for e in out["elements"]}

    assert "h1b" not in by_id
    assert by_id["h1a"]["text"] == "1.1 System Architecture and Interfaces"
    assert by_id["h1a"]["metadata"]["heading_level"] == 2
    assert by_id["h1a"]["bbox"] == [50.0, 50.0, 420.0, 94.0]
    assert "h2" in by_id and "h3" in by_id
    assert by_id["h2"]["metadata"]["role"] == "heading"
    assert by_id["h3"]["metadata"]["role"] == "heading"

    assert "c1b" not in by_id
    assert by_id["c1a"]["text"] == "Figure 1.1: Overview of the layered communication bus"
    assert by_id["c1a"]["metadata"]["caption_of"] == "fig1"
    assert by_id["fig1"]["metadata"]["caption_id"] == "c1a"
    assert by_id["fig1"]["metadata"]["caption_text"] == "Figure 1.1: Overview of the layered communication bus"
    assert by_id["b2"]["metadata"].get("role") is None

    tree = build_section_tree(out["elements"])
    assert [n["title"] for n in tree] == [
        "1.1 System Architecture and Interfaces",
        "1.2 Signal Flow",
        "1.3 Timing Rules",
    ]
    assert tree[0]["element_ids"] == ["b1"]


def test_sequence_continuity_rejects_false_numeric_and_anchors_styles():
    body = "Regular body text repeated to dominate median font size accurately. " * 3
    elements = [
        _elem("h1", page=1, reading_order=0, text="1. Introduction", font_size=16.0, bold=True),
        _elem("b1", page=1, reading_order=1, text=body, font_size=11.0, bold=False),
        _elem("h1_1", page=1, reading_order=2, text="1.1 Scope", font_size=13.0, bold=True),
        _elem("bad_qty", page=1, reading_order=3, text="42 Items", font_size=14.0, bold=True),
        _elem("bad_year_only", page=1, reading_order=4, text="2025", font_size=16.0, bold=True),
        _elem("bad_year_txt", page=1, reading_order=5, text="2025 Roadmap", font_size=14.0, bold=True),
        _elem("bad_dec", page=1, reading_order=6, text="15.5 Percent", font_size=13.0, bold=True),
        _elem("h2", page=2, reading_order=0, text="2. Architecture", font_size=16.0, bold=True),
        _elem("unnum_top", page=2, reading_order=1, text="References", font_size=16.0, bold=True),
        _elem("unnum_sub", page=2, reading_order=2, text="Terminology Notes", font_size=13.0, bold=True),
    ]
    out = apply_structure(_ir(elements))
    by_id = {e["id"]: e for e in out["elements"]}

    for bad_id in ("bad_qty", "bad_year_only", "bad_year_txt", "bad_dec"):
        assert by_id[bad_id]["metadata"].get("role") is None
        assert "heading_level" not in by_id[bad_id]["metadata"]
        assert by_id[bad_id]["metadata"]["section_parent_id"] == "h1_1"

    assert by_id["unnum_top"]["metadata"]["heading_level"] == 1
    assert by_id["unnum_sub"]["metadata"]["heading_level"] == 2
    assert by_id["unnum_sub"]["metadata"]["section_parent_id"] == "unnum_top"

    # Explicit ToC support retains an out-of-sequence numeric heading.
    toc_out = apply_structure(_ir([dict(e, metadata=dict(e["metadata"])) for e in elements]), toc=[(1, "42 Items", 1)])
    toc_by_id = {e["id"]: e for e in toc_out["elements"]}
    assert toc_by_id["bad_qty"]["metadata"]["role"] == "heading"


def test_expanded_caption_prefixes_and_bipartite_two_column_matching():
    elements = [
        _elem("fig_l", kind="figure", page=1, bbox=[50.0, 100.0, 240.0, 240.0], reading_order=0, font_size=None, bold=None),
        _elem("fig_r", kind="figure", page=1, bbox=[260.0, 100.0, 480.0, 240.0], reading_order=1, font_size=None, bold=None),
        # cap_l overlaps fig_r slightly more than fig_l, so greedy matching steals fig_r and strands cap_r.
        _elem("cap_l", page=1, bbox=[180.0, 245.0, 340.0, 260.0], reading_order=2, text="Diagram 1: Left Pipeline", bold=True),
        _elem("cap_r", page=1, bbox=[280.0, 255.0, 470.0, 270.0], reading_order=3, text="Chart 2: Right Metrics", bold=True),
        _elem("fig_abb", kind="figure", page=2, bbox=[50.0, 60.0, 240.0, 160.0], reading_order=0, font_size=None, bold=None),
        _elem("cap_abb", page=2, bbox=[50.0, 166.0, 240.0, 182.0], reading_order=1, text="Abb. 3: Modulaufbau", bold=True),
        _elem("fig_lst", kind="figure", page=2, bbox=[280.0, 60.0, 480.0, 160.0], reading_order=2, font_size=None, bold=None),
        _elem("cap_lst", page=2, bbox=[280.0, 166.0, 480.0, 182.0], reading_order=3, text="Listing 1 - Startup Hook", bold=True),
        _elem("tbl_de", kind="table", page=2, bbox=[50.0, 220.0, 480.0, 320.0], reading_order=4, font_size=None, bold=None),
        _elem("cap_de", page=2, bbox=[50.0, 200.0, 300.0, 215.0], reading_order=5, text="Tabelle 1: Zeitbudget", bold=True),
        _elem("prose_fig", page=2, bbox=[50.0, 350.0, 480.0, 370.0], reading_order=6, text="Figure 4 illustrates the state machine transitions.", bold=False),
        _elem("prose_diag", page=2, bbox=[50.0, 380.0, 480.0, 400.0], reading_order=7, text="Diagram 2 shows the startup flow.", bold=False),
    ]
    out = apply_structure(_ir(elements))
    by_id = {e["id"]: e for e in out["elements"]}

    assert by_id["cap_l"]["metadata"]["caption_of"] == "fig_l"
    assert by_id["fig_l"]["metadata"]["caption_id"] == "cap_l"
    assert by_id["cap_r"]["metadata"]["caption_of"] == "fig_r"
    assert by_id["fig_r"]["metadata"]["caption_id"] == "cap_r"

    assert by_id["cap_abb"]["metadata"]["caption_of"] == "fig_abb"
    assert by_id["cap_lst"]["metadata"]["caption_of"] == "fig_lst"
    assert by_id["cap_de"]["metadata"]["caption_of"] == "tbl_de"
    assert by_id["prose_fig"]["metadata"].get("role") is None
    assert by_id["prose_diag"]["metadata"].get("role") is None


def test_caption_propagation_to_tables_and_diagrams():
    elements = [
        _elem("cap_t", page=1, bbox=[50.0, 60.0, 300.0, 76.0], reading_order=0, text="Tabelle 1: Bus Parameters", bold=True),
        _elem("tbl_e", kind="table", page=1, bbox=[50.0, 82.0, 400.0, 180.0], reading_order=1, font_size=None, bold=None),
        _elem("fig_e", kind="figure", page=1, bbox=[50.0, 210.0, 400.0, 340.0], reading_order=2, font_size=None, bold=None),
        _elem("cap_d1", page=1, bbox=[50.0, 346.0, 300.0, 360.0], reading_order=3, text="Diagram 1: Event", bold=True),
        _elem("cap_d2", page=1, bbox=[50.0, 362.0, 300.0, 376.0], reading_order=4, text="Dispatch Graph", bold=True),
        _elem("tbl_uncap", kind="table", page=2, bbox=[50.0, 80.0, 400.0, 180.0], reading_order=0, font_size=None, bold=None),
        _elem("fig_uncap", kind="figure", page=2, bbox=[50.0, 220.0, 400.0, 320.0], reading_order=1, font_size=None, bold=None),
    ]
    elements[1]["metadata"]["table_id"] = "tbl_0001"
    elements[2]["metadata"]["diagram_id"] = "diag_0001"
    ir = _ir(elements)
    ir["tables"] = [
        {"id": "tbl_0001", "page": 1, "bbox": [50.0, 82.0, 400.0, 180.0], "metadata": {"element_id": "tbl_e"}},
        {"id": "tbl_0002", "page": 2, "bbox": [50.0, 80.0, 400.0, 180.0], "metadata": {"element_id": "tbl_uncap"}},
    ]
    ir["diagrams"] = [
        {"id": "diag_0001", "element_id": "fig_e", "page": 1, "bbox": [50.0, 210.0, 400.0, 340.0], "metadata": {"element_id": "fig_e"}},
        {"id": "diag_0002", "element_id": "fig_uncap", "page": 2, "bbox": [50.0, 220.0, 400.0, 320.0], "metadata": {"element_id": "fig_uncap"}},
    ]
    out = apply_structure(ir)
    by_id = {e["id"]: e for e in out["elements"]}

    t0, t1 = out["tables"]
    d0, d1 = out["diagrams"]
    assert t0["caption_id"] == "cap_t"
    assert t0["caption_text"] == "Tabelle 1: Bus Parameters"
    assert t0["metadata"]["caption_id"] == "cap_t"
    assert t0["metadata"]["caption_text"] == "Tabelle 1: Bus Parameters"
    assert "caption_id" not in t1 and "caption_id" not in t1["metadata"]

    assert d0["caption_id"] == "cap_d1"
    assert d0["caption_text"] == "Diagram 1: Event Dispatch Graph"
    assert d0["metadata"]["caption_id"] == "cap_d1"
    assert d0["metadata"]["caption_text"] == "Diagram 1: Event Dispatch Graph"
    assert "caption_id" not in d1 and "caption_id" not in d1["metadata"]

    for elem in out["elements"]:
        em = elem["metadata"]
        if "caption_of" in em:
            tgt = by_id[em["caption_of"]]
            assert tgt["metadata"]["caption_id"] == elem["id"]
            assert tgt["metadata"]["caption_text"] == elem["text"]
    assert apply_structure(out) == out


def test_caption_prefix_rejects_prose_and_merges_wrapped_prose_line():
    for bad in (
        "diagram of modules",
        "figure in section 2",
        "table with values",
        "Table of Contents",
        "Figure 4 illustrates the state machine",
        "Diagram 2. shows the bus",
    ):
        assert _parse_caption_prefix(bad) == (False, None), bad

    assert _parse_caption_prefix("Figure A: Alpha") == (True, "figure")
    assert _parse_caption_prefix("Table IV: Summary") == (True, "table")
    assert _parse_caption_prefix("Figure A.1: Appendix") == (True, "figure")
    assert _parse_caption_prefix("Fig. 1a: Left") == (True, "figure")

    elements = [
        _elem("fig1", kind="figure", page=1, bbox=[50.0, 50.0, 320.0, 180.0], reading_order=0, font_size=None, bold=None),
        _elem("c1a", page=1, bbox=[50.0, 186.0, 320.0, 200.0], reading_order=1, text="Figure 1: Layered", font_size=10.0, bold=True),
        _elem("c1b", page=1, bbox=[50.0, 203.0, 260.0, 217.0], reading_order=2, text="diagram of modules", font_size=10.0, bold=True),
        _elem("tbl1", kind="table", page=1, bbox=[50.0, 250.0, 320.0, 360.0], reading_order=3, font_size=None, bold=None),
        _elem("c2a", page=1, bbox=[50.0, 366.0, 320.0, 380.0], reading_order=4, text="Table 1: Configuration", font_size=10.0, bold=True),
        _elem("c2b", page=1, bbox=[50.0, 383.0, 260.0, 397.0], reading_order=5, text="table with values", font_size=10.0, bold=True),
        _elem("fig2", kind="figure", page=2, bbox=[50.0, 50.0, 320.0, 180.0], reading_order=0, font_size=None, bold=None),
        _elem("c3a", page=2, bbox=[50.0, 186.0, 320.0, 200.0], reading_order=1, text="Figure 2: Reference", font_size=10.0, bold=True),
        _elem("c3b", page=2, bbox=[50.0, 203.0, 260.0, 217.0], reading_order=2, text="figure in section 2", font_size=10.0, bold=True),
    ]
    out = apply_structure(_ir(elements))
    by_id = {e["id"]: e for e in out["elements"]}
    assert "c1b" not in by_id and "c2b" not in by_id and "c3b" not in by_id
    assert by_id["c1a"]["text"] == "Figure 1: Layered diagram of modules"
    assert by_id["c1a"]["metadata"]["caption_of"] == "fig1"
    assert by_id["c2a"]["text"] == "Table 1: Configuration table with values"
    assert by_id["c2a"]["metadata"]["caption_of"] == "tbl1"
    assert by_id["c3a"]["text"] == "Figure 2: Reference figure in section 2"
    assert by_id["c3a"]["metadata"]["caption_of"] == "fig2"


def test_validate_heading_sequence_preserves_preface_labeled_and_styled_jumps():
    body = "Regular body paragraph repeated to anchor median font size accurately. " * 4
    labeled = [
        _elem("b0", page=1, reading_order=0, text=body, font_size=11.0, bold=False),
        _elem("p0", page=1, reading_order=1, text="0.1 Preface", font_size=16.0, bold=True, layout_label="section_header"),
        _elem("p1", page=1, reading_order=2, text="1 Introduction", font_size=16.0, bold=True, layout_label="section_header"),
        _elem("p10", page=1, reading_order=3, text="10 Conclusion", font_size=16.0, bold=True, layout_label="section_header"),
        _elem("p100", page=1, reading_order=4, text="100 Appendix Summary", font_size=16.0, bold=True, layout_label="section_header"),
    ]
    out_lbl = apply_structure(_ir(labeled))
    assert [o["title"] for o in out_lbl["metadata"]["outline"]] == [
        "0.1 Preface",
        "1 Introduction",
        "10 Conclusion",
        "100 Appendix Summary",
    ]

    styled_unlabeled = [
        _elem("b0", page=1, reading_order=0, text=body, font_size=11.0, bold=False),
        _elem("p0", page=1, reading_order=1, text="0.1 Preface", font_size=16.0, bold=True),
        _elem("p1", page=1, reading_order=2, text="1 Introduction", font_size=16.0, bold=True),
        _elem("p10", page=1, reading_order=3, text="10 Conclusion", font_size=16.0, bold=True),
        _elem("p100", page=1, reading_order=4, text="100 Appendix Summary", font_size=16.0, bold=True),
        _elem("plain_10", page=1, reading_order=5, text="10. apples on table", font_size=11.0, bold=False),
        _elem("plain_yr", page=1, reading_order=6, text="2025 results", font_size=11.0, bold=False),
        _elem("plain_100", page=1, reading_order=7, text="100. items sold", font_size=11.0, bold=False),
    ]
    out_sty = apply_structure(_ir(styled_unlabeled))
    assert [o["title"] for o in out_sty["metadata"]["outline"]] == [
        "0.1 Preface",
        "1 Introduction",
        "10 Conclusion",
        "100 Appendix Summary",
    ]


def test_multiline_merge_renumbers_reading_order_contiguously_per_page():
    elements = [
        _elem("h1a", page=1, bbox=[50.0, 50.0, 350.0, 70.0], reading_order=0, text="1.1 System", font_size=14.0, bold=True),
        _elem("h1b", page=1, bbox=[50.0, 74.0, 300.0, 94.0], reading_order=1, text="Architecture", font_size=14.0, bold=True),
        _elem("b1", page=1, bbox=[50.0, 110.0, 480.0, 140.0], reading_order=2, text="Body paragraph explaining the architecture cleanly.", font_size=11.0, bold=False),
        _elem("fig1", kind="figure", page=1, bbox=[50.0, 160.0, 320.0, 260.0], reading_order=3, font_size=None, bold=None),
        _elem("c1a", page=1, bbox=[50.0, 266.0, 320.0, 280.0], reading_order=4, text="Figure 1: Pipeline", font_size=10.0, bold=True),
        _elem("c1b", page=1, bbox=[50.0, 283.0, 260.0, 297.0], reading_order=5, text="diagram of modules", font_size=10.0, bold=True),
        _elem("b2", page=1, bbox=[50.0, 320.0, 480.0, 350.0], reading_order=6, text="Closing body text on page one.", font_size=11.0, bold=False),
        _elem("fig2", kind="figure", page=2, bbox=[50.0, 60.0, 320.0, 160.0], reading_order=0, font_size=None, bold=None),
        _elem("c2a", page=2, bbox=[50.0, 166.0, 320.0, 180.0], reading_order=1, text="Figure 2: Data", font_size=10.0, bold=True),
        _elem("c2b", page=2, bbox=[50.0, 183.0, 260.0, 197.0], reading_order=2, text="flow overview", font_size=10.0, bold=True),
        _elem("b3", page=2, bbox=[50.0, 220.0, 480.0, 250.0], reading_order=3, text="Trailing text on page two.", font_size=11.0, bold=False),
    ]
    out = apply_structure(_ir(elements))
    p1_ro = [e["reading_order"] for e in out["elements"] if e["page"] == 1]
    p2_ro = [e["reading_order"] for e in out["elements"] if e["page"] == 2]
    assert p1_ro == list(range(len(p1_ro))) == [0, 1, 2, 3, 4]
    assert p2_ro == list(range(len(p2_ro))) == [0, 1, 2]


def test_narrow_gutter_does_not_cross_match_orphan_caption():
    elements = [
        _elem("fig1", kind="figure", page=1, bbox=[50.0, 100.0, 240.0, 200.0], reading_order=0, font_size=None, bold=None),
        _elem("cap1", page=1, bbox=[50.0, 206.0, 240.0, 222.0], reading_order=1, text="Figure 1: Col 1 Figure", bold=True),
        _elem("cap_orphan", page=1, bbox=[50.0, 260.0, 240.0, 276.0], reading_order=2, text="Figure 3: Col 1 Orphan", bold=True),
        _elem("fig2", kind="figure", page=1, bbox=[258.0, 100.0, 480.0, 280.0], reading_order=3, font_size=None, bold=None),
    ]
    out = apply_structure(_ir(elements))
    by_id = {e["id"]: e for e in out["elements"]}
    assert by_id["cap1"]["metadata"]["caption_of"] == "fig1"
    assert "caption_of" not in by_id["cap_orphan"]["metadata"]
    assert "caption_id" not in by_id["fig2"]["metadata"]


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
