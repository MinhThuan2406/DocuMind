"""
Sanity tests for ir_postprocessor.postprocess_ir_structure (hierarchy + table linking).

Run:  .venv\\Scripts\\python scripts\\test_ir_postprocessor.py
(plain asserts, no pytest needed; functions are also pytest-compatible)
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ir_postprocessor import postprocess_ir_structure  # noqa: E402


def _el(eid, etype, bbox, text=""):
    # level=1 mirrors run_dataset_stress_test.py, where every element starts at level 1
    return {"id": eid, "type": etype, "bbox": bbox, "text": text,
            "level": 1, "parent_id": None, "confidence": 0.95, "metadata": {}}


def _ir(elements, tables=None, diagrams=None):
    return {
        "doc_id": "test",
        "source_file": "test.png",
        "metadata": {"page_width": 500, "page_height": 1000},
        "elements": elements,
        "tables": tables or [],
        "diagrams": diagrams or [],
    }


def _run(ir):
    out, _ = postprocess_ir_structure(ir, 500.0, 1000.0)
    return out


def _find_text(ir, text):
    return next(e for e in ir["elements"] if e.get("text") == text)


def _find_type(ir, etype):
    return next(e for e in ir["elements"] if e.get("type") == etype)


# ----------------------------------------------------------------------
# Caption <-> figure hierarchy
# ----------------------------------------------------------------------
def test_caption_before_figure_gets_figure_level_plus_one():
    # Reading order: section header, caption (above), picture, body text
    ir = _run(_ir([
        _el("a", "section_header", [50, 40, 400, 70], "Results"),
        _el("b", "caption", [50, 100, 400, 115], "Figure 1: Overview"),
        _el("c", "picture", [50, 120, 400, 400]),
        _el("d", "text", [50, 450, 400, 480], "Body text."),
    ]))
    header = _find_text(ir, "Results")
    caption = _find_text(ir, "Figure 1: Overview")
    picture = _find_type(ir, "picture")

    assert picture["parent_id"] == header["id"], picture
    assert picture["level"] == header["level"] + 1, picture
    assert caption["parent_id"] == picture["id"], caption
    assert caption["level"] == picture["level"] + 1, (caption, picture)


def test_caption_without_any_target_falls_back_to_section():
    # No table/figure on the page at all -> caption hangs off the current section
    ir = _run(_ir([
        _el("a", "section_header", [50, 40, 400, 70], "Intro"),
        _el("b", "text", [50, 80, 400, 100], "Some text."),
        _el("c", "section_header", [50, 120, 400, 150], "Method"),
        _el("d", "caption", [50, 900, 400, 915], "Orphan caption"),
    ]))
    method = _find_text(ir, "Method")
    caption = _find_text(ir, "Orphan caption")

    assert caption["parent_id"] == method["id"], caption
    assert caption["level"] == method["level"] + 1, (caption, method)


def test_caption_far_from_figure_falls_back_to_section():
    # Figure exists but is more than 300 px away -> not linked, same section fallback
    ir = _run(_ir([
        _el("a", "section_header", [50, 40, 400, 70], "Intro"),
        _el("b", "section_header", [50, 120, 400, 150], "Method"),
        _el("c", "picture", [50, 160, 400, 250]),
        _el("d", "caption", [50, 900, 400, 915], "Far caption"),
    ]))
    method = _find_text(ir, "Method")
    caption = _find_text(ir, "Far caption")
    picture = _find_type(ir, "picture")

    assert caption["parent_id"] == method["id"], caption
    assert caption["level"] == method["level"] + 1, (caption, method)


# ----------------------------------------------------------------------
# Table <-> table element linking
# ----------------------------------------------------------------------
def test_table_without_overlapping_element_gets_no_link_and_warning():
    tbl = {"id": "tbl_0001", "page": 1, "bbox": [50, 100, 300, 200],
           "num_rows": 2, "num_cols": 2, "cells": [], "html": "", "metadata": {}}
    ir = _run(_ir(
        elements=[
            _el("a", "table", [600, 600, 700, 700]),  # far away: IoU == 0
            _el("b", "text", [50, 50, 300, 80], "hello"),
        ],
        tables=[tbl],
    ))
    table_el = _find_type(ir, "table")

    assert tbl["metadata"].get("link_warning") == "no overlapping table element", tbl
    assert "table_id" not in table_el, table_el
    assert "element_id" not in tbl["metadata"], tbl
    assert table_el["text"] == "", table_el  # no "[Table tbl_0001: ...]" text was written


def test_table_with_overlap_links_to_matching_element():
    tables = [
        {"id": "tbl_0001", "page": 1, "bbox": [100, 100, 300, 200],
         "num_rows": 2, "num_cols": 3, "cells": [], "html": "", "metadata": {}},
        {"id": "tbl_0002", "page": 1, "bbox": [100, 400, 300, 500],
         "num_rows": 1, "num_cols": 1, "cells": [], "html": "", "metadata": {}},
    ]
    ir = _run(_ir(
        elements=[
            _el("a", "table", [102, 101, 298, 198]),
            _el("b", "table", [102, 401, 298, 498]),
        ],
        tables=tables,
    ))
    el_1 = next(e for e in ir["elements"] if e.get("table_id") == "tbl_0001")
    el_2 = next(e for e in ir["elements"] if e.get("table_id") == "tbl_0002")

    assert el_1["bbox"] == [102, 101, 298, 198], el_1
    assert el_2["bbox"] == [102, 401, 298, 498], el_2
    assert el_1["text"] == "[Table tbl_0001: 2x3]", el_1
    assert tables[0]["metadata"]["element_id"] == el_1["id"], tables[0]
    assert tables[1]["metadata"]["element_id"] == el_2["id"], tables[1]
    assert "link_warning" not in tables[0]["metadata"], tables[0]
    assert "link_warning" not in tables[1]["metadata"], tables[1]


# ----------------------------------------------------------------------
# Diagram membership
# ----------------------------------------------------------------------
def test_diagram_merged_elements_get_diagram_parent():
    ir = _run(_ir(
        elements=[
            _el("h", "section_header", [50, 40, 400, 70], "Architecture"),
            _el("p", "picture", [50, 100, 250, 300]),
            _el("t", "text", [60, 310, 240, 330], "Node A"),
            _el("o", "text", [50, 500, 400, 530], "Outside paragraph"),
        ],
        diagrams=[{
            "id": "diag_0001", "page": 1, "bbox": [40, 90, 260, 340],
            "cropped_image": "", "merged_element_ids": ["p", "t"],
            "text_summary": "", "nodes": [], "edges": [], "metadata": {},
        }],
    ))
    diag = ir["diagrams"][0]
    member_ids = set(diag["merged_element_ids"])
    assert len(member_ids) == 2, diag

    members = [e for e in ir["elements"] if e["id"] in member_ids]
    assert len(members) == 2, members
    for m in members:
        assert m["parent_id"] == "diag_0001", m
        assert m["level"] == 3, m

    outside = _find_text(ir, "Outside paragraph")
    assert outside["parent_id"] != "diag_0001", outside


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # report and continue
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
