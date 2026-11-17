import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdf.hierarchy import apply_structure, build_section_tree  # noqa: E402


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
