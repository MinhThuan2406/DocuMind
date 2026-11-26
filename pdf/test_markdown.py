import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pdf  # noqa: E402
from pdf import extract_ir, extract_markdown, to_markdown  # noqa: E402
from pdf._test_pdf import TestPdfBuilder  # noqa: E402


def test_public_exports_and_import_isolation():
    assert pdf.to_markdown is to_markdown
    assert pdf.extract_markdown is extract_markdown
    assert to_markdown is extract_markdown
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import sys, pdf; "
                "assert hasattr(pdf, 'to_markdown') and hasattr(pdf, 'extract_markdown'); "
                "bad = [m for m in ('torch', 'transformers', 'onnxruntime', 'docling', 'pdf.diagrams') "
                "if m in sys.modules]; "
                "assert not bad, bad"
            ),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.returncode == 0


def test_headings_clean_text_outline_and_clamping():
    ir = {
        "metadata": {
            "pages": [{"number": 1}],
            "outline": [
                {"id": "h2", "title": "Outline Level Heading", "level": 3, "page": 1},
                {"id": "h4", "title": "Deep Heading", "level": 9, "page": 1},
            ],
        },
        "elements": [
            {
                "id": "h1",
                "type": "text",
                "page": 1,
                "level": 1,
                "reading_order": 0,
                "text": "1. Coverage Overview NEW",
                "metadata": {
                    "role": "heading",
                    "heading_level": 1,
                    "heading_clean_text": "1. Coverage Overview",
                },
            },
            {
                "id": "h2",
                "type": "text",
                "page": 1,
                "level": 1,
                "reading_order": 1,
                "text": "Outline Level Heading",
                "metadata": {"layout_label": "section_header"},
            },
            {
                "id": "h3",
                "type": "text",
                "page": 1,
                "level": 1,
                "reading_order": 2,
                "text": "Document Title",
                "metadata": {"layout_label": "title", "heading_level": 0},
            },
            {
                "id": "h4",
                "type": "text",
                "page": 1,
                "level": 1,
                "reading_order": 3,
                "text": "Deep Heading",
                "metadata": {"role": "heading"},
            },
        ],
    }
    md = to_markdown(ir)
    assert md == (
        "# 1. Coverage Overview\n\n"
        "### Outline Level Heading\n\n"
        "# Document Title\n\n"
        "###### Deep Heading"
    )


def test_tables_gfm_vs_merged_html_and_fallbacks():
    ir = {
        "metadata": {"pages": [{"number": 1}]},
        "elements": [
            {
                "id": "t_gfm",
                "type": "table",
                "page": 1,
                "level": 1,
                "reading_order": 0,
                "text": "[Table tbl_1: 2x2]",
                "metadata": {"table_id": "tbl_1"},
            },
            {
                "id": "t_merged",
                "type": "table",
                "page": 1,
                "level": 1,
                "reading_order": 1,
                "text": "[Table tbl_2: 2x2]",
                "metadata": {"table_id": "tbl_2"},
            },
            {
                "id": "t_merged_no_html",
                "type": "table",
                "page": 1,
                "level": 1,
                "reading_order": 2,
                "text": "",
                "metadata": {"table_id": "tbl_3"},
            },
            {
                "id": "t_missing",
                "type": "table",
                "page": 1,
                "level": 1,
                "reading_order": 3,
                "text": "Plain fallback",
                "metadata": {},
            },
            {
                "id": "t_missing_c2",
                "type": "text",
                "page": 1,
                "level": 2,
                "parent_id": "t_missing",
                "reading_order": 5,
                "text": "Row 2",
                "metadata": {},
            },
            {
                "id": "t_missing_c1",
                "type": "text",
                "page": 1,
                "level": 2,
                "parent_id": "t_missing",
                "reading_order": 4,
                "text": "Row 1",
                "metadata": {},
            },
        ],
        "tables": [
            {
                "id": "tbl_1",
                "page": 1,
                "num_rows": 2,
                "num_cols": 2,
                "cells": [
                    {"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "text": "Key | Name", "is_header": True},
                    {"row": 0, "col": 1, "rowspan": 1, "colspan": 1, "text": "Val", "is_header": True},
                    {"row": 1, "col": 0, "rowspan": 1, "colspan": 1, "text": "Line 1\nLine 2", "is_header": False},
                    {"row": 1, "col": 1, "rowspan": 1, "colspan": 1, "text": "42", "is_header": False},
                ],
                "html": "<table><tr><th>Key | Name</th><th>Val</th></tr></table>",
            },
            {
                "id": "tbl_2",
                "page": 1,
                "num_rows": 2,
                "num_cols": 2,
                "cells": [
                    {"row": 0, "col": 0, "rowspan": 1, "colspan": 2, "text": "Group", "is_header": True},
                    {"row": 1, "col": 0, "rowspan": 1, "colspan": 1, "text": "A", "is_header": False},
                    {"row": 1, "col": 1, "rowspan": 1, "colspan": 1, "text": "B", "is_header": False},
                ],
                "html": '<table><thead><tr><th colspan="2">Group</th></tr></thead><tbody><tr><td>A</td><td>B</td></tr></tbody></table>',
            },
            {
                "id": "tbl_3",
                "page": 1,
                "num_rows": 2,
                "num_cols": 2,
                "cells": [
                    {"row": 0, "col": 0, "row_span": 2, "col_span": 1, "text": "Tall", "is_header": False},
                    {"row": 0, "col": 1, "row_span": 1, "col_span": 1, "text": "X", "is_header": False},
                    {"row": 1, "col": 1, "row_span": 1, "col_span": 1, "text": "Y", "is_header": False},
                ],
            },
        ],
    }
    md = to_markdown(ir)
    parts = md.split("\n\n")
    assert parts[0] == "| Key \\| Name | Val |\n| --- | --- |\n| Line 1<br>Line 2 | 42 |"
    assert parts[1] == '<table><thead><tr><th colspan="2">Group</th></tr></thead><tbody><tr><td>A</td><td>B</td></tr></tbody></table>'
    assert parts[2] == '<table>\n<tr><td rowspan="2">Tall</td><td>X</td></tr>\n<tr><td>Y</td></tr>\n</table>'
    assert parts[3] == "Row 1\nRow 2"


def test_formulas_via_metadata_ir_list_and_fallback():
    ir_with_formulas = {
        "elements": [
            {
                "id": "f1",
                "type": "formula",
                "page": 1,
                "level": 1,
                "reading_order": 0,
                "text": "E = mc2",
                "metadata": {"latex": "E = mc^2"},
            },
            {
                "id": "f2",
                "type": "text",
                "page": 1,
                "level": 1,
                "reading_order": 1,
                "text": "raw formula 2",
                "metadata": {"layout_label": "formula", "formula_id": "fml_0001"},
            },
            {
                "id": "f3",
                "type": "formula",
                "page": 1,
                "level": 1,
                "reading_order": 2,
                "text": "raw formula 3",
                "metadata": {},
            },
            {
                "id": "f4",
                "type": "formula",
                "page": 1,
                "level": 1,
                "reading_order": 3,
                "text": "  fallback text  ",
                "metadata": {},
            },
        ],
        "formulas": [
            {
                "id": "fml_0001",
                "element_id": "f2",
                "page": 1,
                "bbox": [0, 0, 10, 10],
                "latex": r"\frac{a}{b}",
                "confidence": 0.95,
                "backend": "mock",
            },
            {
                "id": "fml_0002",
                "element_id": "f3",
                "page": 1,
                "bbox": [0, 10, 10, 20],
                "latex": r"$$\sum_{i=1}^n x_i$$",
                "confidence": 0.9,
                "backend": "mock",
            },
        ],
    }
    md = to_markdown(ir_with_formulas)
    assert md == (
        "$$\nE = mc^2\n$$\n\n"
        "$$\n\\frac{a}{b}\n$$\n\n"
        "$$\n\\sum_{i=1}^n x_i\n$$\n\n"
        "fallback text"
    )

    ir_no_formulas_key = {
        "elements": [
            {"id": "f1", "type": "formula", "page": 1, "level": 1, "reading_order": 0, "text": "x + y = z", "metadata": {}},
        ]
    }
    assert to_markdown(ir_no_formulas_key) == "x + y = z"


def test_diagrams_mermaid_and_figure_toggles():
    ir = {
        "elements": [
            {
                "id": "fig1",
                "type": "figure",
                "page": 1,
                "level": 1,
                "reading_order": 0,
                "text": "",
                "metadata": {"diagram_id": "diag_1"},
            },
            {
                "id": "fig1_c1",
                "type": "text",
                "page": 1,
                "level": 2,
                "parent_id": "fig1",
                "reading_order": 1,
                "text": "Start",
                "metadata": {},
            },
            {
                "id": "fig2",
                "type": "figure",
                "page": 1,
                "level": 1,
                "reading_order": 2,
                "text": "",
                "metadata": {"caption_text": "System Architecture"},
            },
        ],
        "diagrams": [
            {
                "id": "diag_1",
                "element_id": "fig1",
                "page": 1,
                "diagram_type": "flowchart",
                "nodes": [
                    {"id": "end", "shape_type": "rect", "label": "Start (A)", "parent_group_id": "g1"},
                    {"id": "n2", "shape_type": "diamond", "label": "Valid?", "parent_group_id": "g1"},
                    {"id": "n3", "shape_type": "round_rect", "label": "Done", "parent_group_id": None},
                ],
                "edges": [
                    {"id": "e1", "source_node_id": "end", "target_node_id": "n2", "direction": "directed", "style": "solid", "label": ""},
                    {"id": "e2", "source_node_id": "n2", "target_node_id": "n3", "direction": "directed", "style": "dashed", "label": "ok | yes"},
                ],
                "groups": [
                    {"id": "g1", "label": "Stage 1", "parent_group_id": None, "member_node_ids": ["end", "n2"]},
                ],
            }
        ],
    }

    md_default = to_markdown(ir)
    assert "```mermaid\nflowchart TD" in md_default
    assert 'subgraph g1["Stage 1"]' in md_default
    assert 'n_end["Start (A)"]' in md_default
    assert 'n2{"Valid?"}' in md_default
    assert 'n3("Done")' in md_default
    assert "n_end --> n2" in md_default
    assert 'n2 -.->|"ok &#124; yes"| n3' in md_default
    assert "*[Figure: System Architecture]*" in md_default
    assert md_default.count("Start (A)") == 1

    md_no_diag = to_markdown(ir, include_diagrams=False, include_figures=True)
    assert "```mermaid" not in md_no_diag
    assert md_no_diag == "*[Figure]*\n\n*[Figure: System Architecture]*"

    md_diag_only = to_markdown(ir, include_diagrams=True, include_figures=False)
    assert "```mermaid" in md_diag_only
    assert "System Architecture" not in md_diag_only

    assert to_markdown(ir, include_diagrams=False, include_figures=False) == ""


def test_furniture_reading_order_and_page_breaks():
    ir = {
        "metadata": {"pages": [{"number": 1}, {"number": 2}, {"number": 3}]},
        "elements": [
            {"id": "p2_b", "type": "text", "page": 2, "level": 1, "reading_order": 5, "text": "Second on P2", "metadata": {}},
            {"id": "p2_a", "type": "text", "page": 2, "level": 1, "reading_order": 1, "text": "First on P2", "metadata": {}},
            {"id": "p1_hdr", "type": "text", "page": 1, "level": 1, "reading_order": 0, "text": "Running Header", "metadata": {"role": "page_header"}},
            {"id": "p1_body", "type": "text", "page": 1, "level": 1, "reading_order": 1, "text": "Body P1", "metadata": {}},
            {"id": "p1_ftr", "type": "text", "page": 1, "level": 1, "reading_order": 2, "text": "Page 1", "metadata": {"layout_label": "page_footer"}},
            {"id": "p3_ftr", "type": "text", "page": 3, "level": 1, "reading_order": 0, "text": "Page 3", "metadata": {"role": "page_footer"}},
        ],
    }

    assert to_markdown(ir) == "Body P1\n\nFirst on P2\n\nSecond on P2"
    assert to_markdown(ir, page_breaks=True) == "Body P1\n\n---\n\nFirst on P2\n\nSecond on P2"
    assert to_markdown(ir, include_furniture=True, page_breaks=True) == (
        "Running Header\n\nBody P1\n\nPage 1\n\n---\n\nFirst on P2\n\nSecond on P2\n\n---\n\nPage 3"
    )


def test_lists_code_captions_and_multiline_prose():
    ir = {
        "elements": [
            {"id": "l1", "type": "text", "page": 1, "level": 1, "reading_order": 0, "text": "Unprefixed item", "metadata": {"layout_label": "list_item"}},
            {"id": "l2", "type": "text", "page": 1, "level": 1, "reading_order": 1, "text": "2. Numbered item", "metadata": {"layout_label": "list_item"}},
            {"id": "l3", "type": "text", "page": 1, "level": 1, "reading_order": 2, "text": "- Existing bullet\ncontinued line", "metadata": {"layout_label": "list_item"}},
            {"id": "c1", "type": "text", "page": 1, "level": 1, "reading_order": 3, "text": "def f():\n    return ```hi```", "metadata": {"layout_label": "code", "language": "python"}},
            {"id": "cap1", "type": "text", "page": 1, "level": 1, "reading_order": 4, "text": "Figure 1: Sample caption", "metadata": {"role": "caption"}},
            {"id": "p1", "type": "text", "page": 1, "level": 1, "reading_order": 5, "text": "First line of prose\nSecond line of prose", "metadata": {}},
        ]
    }
    md = to_markdown(ir)
    assert md == (
        "- Unprefixed item\n"
        "2. Numbered item\n"
        "- Existing bullet\n"
        "  continued line\n\n"
        "````python\n"
        "def f():\n"
        "    return ```hi```\n"
        "````\n\n"
        "*Figure 1: Sample caption*\n\n"
        "First line of prose\n"
        "Second line of prose"
    )


def test_end_to_end_extract_ir_to_markdown():
    b = TestPdfBuilder()
    p = b.new_page(width=600, height=800)
    p.insert_text((50, 60), "Document Heading", fontsize=20, bold=True)
    p.insert_text((50, 120), "First body paragraph with clear text.", fontsize=11)
    p.insert_text((50, 160), "Second body paragraph below.", fontsize=11)

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "sample.pdf"
        b.save(pdf_path)
        ir = extract_ir(pdf_path, figures=False, ocr=False)

    md = to_markdown(ir)
    assert "# Document Heading" in md
    assert "First body paragraph with clear text." in md
    assert "Second body paragraph below." in md


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
