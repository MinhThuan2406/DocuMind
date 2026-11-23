import inspect
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

import pdf.ir as ir_mod  # noqa: E402
from pdf._test_pdf import TestPdfBuilder  # noqa: E402
from pdf.backends import DiagramExtractor, RunStats, TableExtractor, get, register, registered, reset_cache  # noqa: E402
from pdf.backends.table_auto import AutoTableExtractor, _row_y_spans_pt, refine_table_headers  # noqa: E402
from pdf.backends.tatr_onnx import objects_to_table, table_to_html  # noqa: E402
from pdf.eval.tables import teds_score  # noqa: E402
from pdf.models import ExtractedTable, FigureBox, TableCell  # noqa: E402
from pdf.ocr.types import OcrLine  # noqa: E402
from pdf.tables import extract_tables_on_page  # noqa: E402


def test_objects_to_table_grid_headers_and_merged_cells():
    # 3x3 table in a 300x180 crop: row 0 is header with col 1-2 merged; col 0 rows 1-2 merged.
    objects = [
        ("table", (0.0, 0.0, 300.0, 180.0), 0.99),
        ("table column", (0.0, 0.0, 98.0, 180.0), 0.96),
        ("table column", (102.0, 0.0, 198.0, 180.0), 0.95),
        ("table column", (202.0, 0.0, 300.0, 180.0), 0.94),
        # Duplicate overlapping row should be deduplicated cleanly.
        ("table row", (0.0, 0.0, 300.0, 58.0), 0.97),
        ("table row", (0.0, 4.0, 300.0, 56.0), 0.70),
        ("table row", (0.0, 62.0, 300.0, 118.0), 0.95),
        ("table row", (0.0, 122.0, 300.0, 180.0), 0.93),
        ("table column header", (0.0, 0.0, 300.0, 58.0), 0.96),
        ("table spanning cell", (102.0, 0.0, 300.0, 58.0), 0.92),
        ("table spanning cell", (0.0, 62.0, 98.0, 180.0), 0.90),
    ]
    tokens = [
        ("Metric", (10.0, 15.0, 55.0, 30.0)),
        ("Values", (150.0, 15.0, 210.0, 30.0)),
        ("GroupA", (20.0, 90.0, 70.0, 110.0)),
        ("10", (130.0, 75.0, 150.0, 95.0)),
        ("20", (230.0, 75.0, 250.0, 95.0)),
        ("30", (130.0, 135.0, 150.0, 155.0)),
        ("40", (230.0, 135.0, 250.0, 155.0)),
    ]
    tbl = objects_to_table(objects, (300, 180), (0.0, 0.0, 300.0, 180.0), tokens)
    assert tbl.num_rows == 3 and tbl.num_cols == 3
    assert len(tbl.cells) == 7

    by_rc = {(c.row, c.col): c for c in tbl.cells}
    assert by_rc[(0, 0)].text == "Metric" and by_rc[(0, 0)].is_header
    assert by_rc[(0, 1)].colspan == 2 and by_rc[(0, 1)].rowspan == 1 and by_rc[(0, 1)].text == "Values"
    assert by_rc[(1, 0)].rowspan == 2 and by_rc[(1, 0)].colspan == 1 and by_rc[(1, 0)].text == "GroupA"
    assert by_rc[(2, 2)].text == "40" and not by_rc[(2, 2)].is_header

    out_html = table_to_html(tbl)
    assert '<th colspan="2">Values</th>' in out_html
    assert '<td rowspan="2">GroupA</td>' in out_html


def test_teds_metric_exact_and_span_difference():
    gt = '<table><tr><td>A</td><td colspan="2">B</td></tr><tr><td>1</td><td>2</td><td>3</td></tr></table>'
    same = "<table><tr><th>A</th><th colspan='2'>B</th></tr><tr><td>1</td><td>2</td><td>3</td></tr></table>"
    unmerged = "<table><tr><td>A</td><td>B</td><td></td></tr><tr><td>1</td><td>2</td><td>3</td></tr></table>"
    assert teds_score(same, gt, structure_only=True) == 1.0
    assert teds_score(same, gt, structure_only=False) == 1.0
    assert 0.5 < teds_score(unmerged, gt, structure_only=True) < 1.0


def test_pdfplumber_table_extractor_and_extract_ir_tables_true():
    assert "tatr-onnx" in registered("table") and "pdfplumber" in registered("table")

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "tbl.pdf"
        b = TestPdfBuilder()
        page = b.new_page(width=400, height=300)
        # Draw a crisp 3x3 grid (8 lines >= MIN_CANDIDATE_DRAWINGS).
        for x in (40, 120, 200, 280):
            page.draw_line((x, 50), (x, 170), width=0.5)
        for y in (50, 90, 130, 170):
            page.draw_line((40, y), (280, y), width=0.5)
        page.insert_text((50, 74), "Key", fontsize=10)
        page.insert_text((130, 74), "Val", fontsize=10)
        page.insert_text((210, 74), "Unit", fontsize=10)
        page.insert_text((50, 114), "Alpha", fontsize=10)
        page.insert_text((130, 114), "42", fontsize=10)
        page.insert_text((210, 114), "ms", fontsize=10)
        page.insert_text((50, 154), "Beta", fontsize=10)
        page.insert_text((130, 154), "99", fontsize=10)
        page.insert_text((210, 154), "kb", fontsize=10)
        b.save(pdf_path)

        import pdf.figures as figures_mod

        orig_detect = figures_mod.detect_figures
        figures_mod.detect_figures = lambda _p, pages, **_kw: {
            p: [FigureBox((38.0, 48.0, 282.0, 172.0), "table", "layout_table", 0.95)] for p in pages
        }
        try:
            ir = ir_mod.extract_ir(pdf_path, figures=True, ocr=False, tables=True, table_backend="pdfplumber")
        finally:
            figures_mod.detect_figures = orig_detect

    assert json.loads(json.dumps(ir)) == ir
    assert len(ir["tables"]) == 1
    tbl = ir["tables"][0]
    assert tbl["id"] == "tbl_0001"
    assert tbl["num_rows"] == 3 and tbl["num_cols"] == 3
    assert [c["text"] for c in tbl["cells"]] == ["Key", "Val", "Unit", "Alpha", "42", "ms", "Beta", "99", "kb"]

    tbl_elems = [e for e in ir["elements"] if e["type"] == "table"]
    assert len(tbl_elems) == 1
    assert tbl_elems[0]["metadata"]["table_id"] == "tbl_0001"
    assert tbl_elems[0]["text"] == "[Table tbl_0001: 3x3]"

    # Custom TableExtractor via registry works cleanly through extract_ir.
    class _FakeTableExtractor:
        name = "fake_tbl"
        last_stats = None

        def extract(self, image: Image.Image, table_bbox_pt, tokens=()):
            del image, tokens
            return objects_to_table(
                [
                    ("table row", (0.0, 0.0, 100.0, 50.0), 0.9),
                    ("table column", (0.0, 0.0, 100.0, 50.0), 0.9),
                ],
                (100, 50),
                table_bbox_pt,
                [("OnlyCell", (10.0, 10.0, 50.0, 30.0))],
            )

    register("table", "fake_tbl", lambda **_kw: _FakeTableExtractor())
    assert "fake_tbl" in registered("table")


def test_pdfplumber_merged_cells_rowspan_and_colspan():
    import pdfplumber
    from pdf.backends import get
    from pdf.tables import extract_tables_on_page

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "merged.pdf"
        b = TestPdfBuilder()
        p = b.new_page(width=400, height=300)
        # 3x3 grid: row 0 merges cols 1..2; col 0 merges rows 1..2.
        p.draw_line((40, 50), (280, 50), width=0.5)
        p.draw_line((40, 90), (280, 90), width=0.5)
        p.draw_line((120, 130), (280, 130), width=0.5)
        p.draw_line((40, 170), (280, 170), width=0.5)
        p.draw_line((40, 50), (40, 170), width=0.5)
        p.draw_line((120, 50), (120, 170), width=0.5)
        p.draw_line((200, 90), (200, 170), width=0.5)
        p.draw_line((280, 50), (280, 170), width=0.5)
        p.insert_text((50, 74), "Hdr", fontsize=10)
        p.insert_text((160, 74), "WideSpan", fontsize=10)
        p.insert_text((50, 114), "TallSpan", fontsize=10)
        p.insert_text((135, 114), "A1", fontsize=10)
        p.insert_text((215, 114), "B1", fontsize=10)
        p.insert_text((135, 154), "A2", fontsize=10)
        p.insert_text((215, 154), "B2", fontsize=10)
        b.save(pdf_path)

        with pdfplumber.open(str(pdf_path)) as doc:
            tables = extract_tables_on_page(
                doc.pages[0],
                [(38.0, 48.0, 282.0, 172.0)],
                extractor=get("table", "pdfplumber"),
            )
    tbl = tables[0]
    assert tbl.num_rows == 3 and tbl.num_cols == 3
    by_rc = {(c.row, c.col): c for c in tbl.cells}
    assert by_rc[(0, 1)].text == "WideSpan" and by_rc[(0, 1)].colspan == 2 and by_rc[(0, 1)].rowspan == 1
    assert by_rc[(1, 0)].text == "TallSpan" and by_rc[(1, 0)].rowspan == 2 and by_rc[(1, 0)].colspan == 1
    assert all(c.col + c.colspan <= tbl.num_cols and c.row + c.rowspan <= tbl.num_rows for c in tbl.cells)


def test_auto_backend_registration_protocol_and_defaults():
    assert "auto" in registered("table")
    ext = get("table", "auto")
    assert ext.name == "auto"
    assert ext.last_stats is None
    assert isinstance(ext, TableExtractor)
    assert inspect.signature(ir_mod.extract_ir).parameters["table_backend"].default == "auto"
    assert inspect.signature(ir_mod.extract_ir).parameters["tables"].default is False
    assert inspect.signature(extract_tables_on_page).parameters["extractor"].default == "auto"
    assert inspect.signature(extract_tables_on_page).parameters["words"].default is None


def test_auto_bordered_multirow_projected_headers_and_staggered_rowspan():
    import pdfplumber

    # Synthetic check for _row_y_spans_pt linear interpolation on a missing interior boundary.
    synth = ExtractedTable(
        bbox=(0.0, 0.0, 100.0, 90.0),
        num_rows=3,
        num_cols=2,
        cells=(
            TableCell(0, 0, 2, 1, "A", (0.0, 0.0, 50.0, 60.0), True),
            TableCell(0, 1, 2, 1, "B", (50.0, 0.0, 100.0, 60.0), True),
            TableCell(2, 0, 1, 1, "C", (0.0, 60.0, 50.0, 90.0), False),
            TableCell(2, 1, 1, 1, "D", (50.0, 60.0, 100.0, 90.0), False),
        ),
        source="pdfplumber",
        confidence=1.0,
    )
    assert _row_y_spans_pt(synth) == [(0.0, 30.0), (30.0, 60.0), (60.0, 90.0)]

    # Verify body rowspan crossing a projected row header stays non-header.
    cross_tbl = ExtractedTable(
        bbox=(0.0, 0.0, 100.0, 90.0),
        num_rows=3,
        num_cols=2,
        cells=(
            TableCell(0, 0, 1, 2, "Hdr", (0.0, 0.0, 100.0, 30.0), True),
            TableCell(1, 0, 2, 1, "SpanBody", (0.0, 30.0, 50.0, 90.0), False),
            TableCell(1, 1, 1, 1, "Body1", (50.0, 30.0, 100.0, 60.0), False),
            TableCell(2, 1, 1, 1, "GroupB", (50.0, 60.0, 100.0, 90.0), False),
        ),
        source="pdfplumber",
        confidence=1.0,
    )
    refined_cross = refine_table_headers(
        cross_tbl,
        [
            ("table column header", (0.0, 0.0, 100.0, 30.0), 0.9),
            ("table projected row header", (0.0, 60.0, 100.0, 90.0), 0.9),
        ],
        (100, 90),
        (0.0, 0.0, 100.0, 90.0),
    )
    cross_by_rc = {(c.row, c.col): c for c in refined_cross.cells}
    assert cross_by_rc[(0, 0)].is_header is True
    assert cross_by_rc[(1, 0)].is_header is False
    assert cross_by_rc[(2, 1)].is_header is True

    # Real pdfplumber table with multi-row header (rowspan=2 + colspan=2), projected row header, and staggered body rowspans.
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "staggered.pdf"
        b = TestPdfBuilder()
        p = b.new_page(width=420, height=300)
        # Rows at y = 40, 70, 100, 130, 160, 190, 220 (6 rows); cols at x = 40, 120, 200, 280, 360 (4 cols).
        for y in (40, 100, 130, 220):
            p.draw_line((40, y), (360, y), width=0.5)
        p.draw_line((120, 70), (280, 70), width=0.5)
        p.draw_line((120, 160), (200, 160), width=0.5)
        p.draw_line((280, 160), (360, 160), width=0.5)
        p.draw_line((40, 190), (120, 190), width=0.5)
        p.draw_line((200, 190), (280, 190), width=0.5)

        p.draw_line((40, 40), (40, 220), width=0.5)
        p.draw_line((360, 40), (360, 220), width=0.5)
        for x in (120, 280):
            p.draw_line((x, 40), (x, 100), width=0.5)
            p.draw_line((x, 130), (x, 220), width=0.5)
        p.draw_line((200, 70), (200, 100), width=0.5)
        p.draw_line((200, 130), (200, 220), width=0.5)

        p.insert_text((50, 74), "Param", fontsize=9)
        p.insert_text((170, 58), "Limits", fontsize=9)
        p.insert_text((295, 74), "Unit", fontsize=9)
        p.insert_text((140, 88), "Min", fontsize=9)
        p.insert_text((220, 88), "Max", fontsize=9)
        p.insert_text((160, 118), "Section A", fontsize=9)
        p.insert_text((55, 164), "S0", fontsize=9)
        p.insert_text((140, 148), "1", fontsize=9)
        p.insert_text((215, 164), "S2", fontsize=9)
        p.insert_text((300, 148), "ms", fontsize=9)
        p.insert_text((55, 208), "Tail0", fontsize=9)
        p.insert_text((140, 194), "2", fontsize=9)
        p.insert_text((220, 208), "3", fontsize=9)
        p.insert_text((300, 194), "kb", fontsize=9)
        b.save(pdf_path)

        class _FakeTatr:
            def detect_objects(self, crop: Image.Image):
                cw, ch = float(crop.width), float(crop.height)
                # Crop bbox is (40-4, 40-4, 360+4, 220+4) -> height 188 pt (y from 36 to 224).
                def py(y_pt: float) -> float:
                    return (y_pt - 36.0) / 188.0 * ch

                return [
                    ("table", (0.0, py(40.0), cw, py(220.0)), 0.99),
                    ("table row", (0.0, py(40.0), cw, py(70.0)), 0.96),
                    ("table row", (0.0, py(70.0), cw, py(100.0)), 0.95),
                    ("table row", (0.0, py(100.0), cw, py(130.0)), 0.94),
                    ("table row", (0.0, py(130.0), cw, py(160.0)), 0.93),
                    ("table row", (0.0, py(160.0), cw, py(190.0)), 0.92),
                    ("table row", (0.0, py(190.0), cw, py(220.0)), 0.91),
                    ("table column header", (0.0, py(40.0), cw, py(100.0)), 0.97),
                    ("table projected row header", (0.0, py(100.0), cw, py(130.0)), 0.95),
                ]

        ext = AutoTableExtractor(tatr_extractor=_FakeTatr())
        with pdfplumber.open(str(pdf_path)) as doc:
            tables = extract_tables_on_page(
                doc.pages[0],
                [(38.0, 38.0, 362.0, 222.0)],
                extractor=ext,
            )

    tbl = tables[0]
    assert tbl.source == "auto_refined" and tbl.confidence == 1.0
    assert tbl.num_rows == 6 and tbl.num_cols == 4
    by_rc = {(c.row, c.col): c for c in tbl.cells}
    assert by_rc[(0, 0)].rowspan == 2 and by_rc[(0, 0)].is_header is True
    assert by_rc[(0, 1)].colspan == 2 and by_rc[(0, 1)].is_header is True
    assert by_rc[(1, 1)].is_header is True and by_rc[(1, 2)].is_header is True
    assert by_rc[(2, 0)].colspan == 4 and by_rc[(2, 0)].is_header is True
    assert by_rc[(3, 0)].rowspan == 2 and by_rc[(3, 0)].is_header is False
    assert by_rc[(4, 1)].rowspan == 2 and by_rc[(4, 1)].is_header is False
    assert by_rc[(3, 2)].rowspan == 2 and by_rc[(3, 2)].is_header is False
    assert by_rc[(4, 3)].rowspan == 2 and by_rc[(4, 3)].is_header is False


def test_auto_bordered_key_value_no_header_and_column_only_fallback():
    import pdfplumber

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "kv.pdf"
        b = TestPdfBuilder()
        p = b.new_page(width=400, height=300)
        for x in (40, 160, 280):
            p.draw_line((x, 50), (x, 140), width=0.5)
        for y in (50, 80, 110, 140):
            p.draw_line((40, y), (280, y), width=0.5)
        p.insert_text((50, 68), "Name", fontsize=10)
        p.insert_text((170, 68), "DocuMind", fontsize=10)
        p.insert_text((50, 98), "Version", fontsize=10)
        p.insert_text((170, 98), "2.0", fontsize=10)
        p.insert_text((50, 128), "Mode", fontsize=10)
        p.insert_text((170, 128), "Auto", fontsize=10)
        b.save(pdf_path)

        class _TatrObjects:
            def __init__(self, objs):
                self._objs = objs

            def detect_objects(self, _crop: Image.Image):
                return self._objs

        with pdfplumber.open(str(pdf_path)) as doc:
            page = doc.pages[0]
            kv_ext = AutoTableExtractor(
                tatr_extractor=_TatrObjects([
                    ("table", (0.0, 0.0, 200.0, 100.0), 0.99),
                    ("table row", (0.0, 0.0, 200.0, 33.0), 0.95),
                    ("table row", (0.0, 33.0, 200.0, 66.0), 0.94),
                    ("table row", (0.0, 66.0, 200.0, 100.0), 0.93),
                ])
            )
            kv_tbl = extract_tables_on_page(page, [(38.0, 48.0, 282.0, 142.0)], extractor=kv_ext)[0]
            assert kv_tbl.source == "auto_refined" and kv_tbl.confidence == 1.0
            assert all(not c.is_header for c in kv_tbl.cells)

            for col_only_objs in (
                [],
                [("table", (0.0, 0.0, 200.0, 100.0), 0.95), ("table column", (0.0, 0.0, 100.0, 100.0), 0.90)],
            ):
                fb_ext = AutoTableExtractor(tatr_extractor=_TatrObjects(col_only_objs))
                fb_tbl = extract_tables_on_page(page, [(38.0, 48.0, 282.0, 142.0)], extractor=fb_ext)[0]
                assert fb_tbl.source == "auto_refined" and fb_tbl.confidence == 1.0
                assert all(c.is_header == (c.row == 0) for c in fb_tbl.cells)


def test_auto_borderless_0x0_and_1x1_and_zebra_nx1_fallback_to_tatr():
    import pdfplumber

    def _make_tatr(cols: int):
        class _FakeTatr:
            def __init__(self):
                self.detect_calls = 0

            def _objs_for(self, w: int, h: int):
                cw, ch = float(w), float(h)
                objs = [
                    ("table", (0.0, 0.0, cw, ch), 0.91),
                    ("table row", (0.0, 0.0, cw, ch * 0.5), 0.90),
                    ("table row", (0.0, ch * 0.5, cw, ch), 0.89),
                    ("table column header", (0.0, 0.0, cw, ch * 0.5), 0.92),
                ]
                if cols >= 2:
                    objs.append(("table column", (0.0, 0.0, cw * 0.5, ch), 0.93))
                    objs.append(("table column", (cw * 0.5, 0.0, cw, ch), 0.93))
                else:
                    objs.append(("table column", (0.0, 0.0, cw, ch), 0.93))
                return objs

            def detect_objects(self, crop: Image.Image):
                self.detect_calls += 1
                return self._objs_for(crop.width, crop.height)

            def extract(self, image: Image.Image, table_bbox_pt, tokens=()):
                return objects_to_table(self._objs_for(image.width, image.height), image.size, table_bbox_pt, tokens)

        return _FakeTatr()

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "fallback_cases.pdf"
        b = TestPdfBuilder()

        # Page 1: borderless 0x0.
        p1 = b.new_page(width=400, height=300)
        p1.insert_text((60, 74), "ColA", fontsize=10)
        p1.insert_text((200, 74), "ColB", fontsize=10)
        p1.insert_text((60, 124), "Val1", fontsize=10)
        p1.insert_text((200, 124), "Val2", fontsize=10)

        # Page 2: single outer box 1x1.
        p2 = b.new_page(width=400, height=300)
        p2.draw_rect((40, 50, 280, 150), width=0.5)
        p2.insert_text((60, 74), "ColA", fontsize=10)
        p2.insert_text((200, 74), "ColB", fontsize=10)
        p2.insert_text((60, 124), "Val1", fontsize=10)
        p2.insert_text((200, 124), "Val2", fontsize=10)

        # Page 3: boxed booktabs / zebra Nx1 (outer box + horizontal lines, no vertical split).
        p3 = b.new_page(width=400, height=300)
        p3.draw_rect((40, 50, 280, 150), width=0.5)
        p3.draw_line((40, 100), (280, 100), width=0.5)
        p3.insert_text((60, 74), "ColA", fontsize=10)
        p3.insert_text((200, 74), "ColB", fontsize=10)
        p3.insert_text((60, 124), "Val1", fontsize=10)
        p3.insert_text((200, 124), "Val2", fontsize=10)
        b.save(pdf_path)

        with pdfplumber.open(str(pdf_path)) as doc:
            box = [(38.0, 48.0, 282.0, 152.0)]

            t_0x0 = extract_tables_on_page(doc.pages[0], box, words=None, extractor=AutoTableExtractor(tatr_extractor=_make_tatr(2)))[0]
            assert t_0x0.source == "auto_tatr" and t_0x0.confidence == 0.91
            assert t_0x0.num_rows == 2 and t_0x0.num_cols == 2
            assert [c.text for c in t_0x0.cells] == ["ColA", "ColB", "Val1", "Val2"]

            t_1x1 = extract_tables_on_page(doc.pages[1], box, extractor=AutoTableExtractor(tatr_extractor=_make_tatr(2)))[0]
            assert t_1x1.source == "auto_tatr" and t_1x1.confidence == 0.91
            assert t_1x1.num_rows == 2 and t_1x1.num_cols == 2
            assert [c.text for c in t_1x1.cells] == ["ColA", "ColB", "Val1", "Val2"]

            tatr_multi = _make_tatr(2)
            t_nx1_split = extract_tables_on_page(doc.pages[2], box, extractor=AutoTableExtractor(tatr_extractor=tatr_multi))[0]
            assert tatr_multi.detect_calls == 1
            assert t_nx1_split.source == "auto_tatr" and t_nx1_split.confidence == 0.91
            assert t_nx1_split.num_rows == 2 and t_nx1_split.num_cols == 2

            tatr_single = _make_tatr(1)
            t_nx1_keep = extract_tables_on_page(doc.pages[2], box, extractor=AutoTableExtractor(tatr_extractor=tatr_single))[0]
            assert tatr_single.detect_calls == 1
            assert t_nx1_keep.source == "auto_refined" and t_nx1_keep.confidence == 1.0
            assert t_nx1_keep.num_rows == 2 and t_nx1_keep.num_cols == 1
            assert t_nx1_keep.cells[0].is_header is True and t_nx1_keep.cells[1].is_header is False


def test_auto_mixed_page_bordered_and_borderless():
    import pdfplumber

    class _FakeTatr:
        def _objs(self, crop: Image.Image):
            cw, ch = float(crop.width), float(crop.height)
            return [
                ("table", (0.0, 0.0, cw, ch), 0.88),
                ("table row", (0.0, 0.0, cw, ch * 0.5), 0.9),
                ("table row", (0.0, ch * 0.5, cw, ch), 0.9),
                ("table column", (0.0, 0.0, cw * 0.5, ch), 0.9),
                ("table column", (cw * 0.5, 0.0, cw, ch), 0.9),
                ("table column header", (0.0, 0.0, cw, ch * 0.5), 0.9),
            ]

        def detect_objects(self, crop: Image.Image):
            return self._objs(crop)

        def extract(self, image: Image.Image, table_bbox_pt, tokens=()):
            return objects_to_table(self._objs(image), image.size, table_bbox_pt, tokens)

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "mixed.pdf"
        b = TestPdfBuilder()
        p = b.new_page(width=400, height=400)
        # Top bordered 2x2 table.
        for x in (40, 160, 280):
            p.draw_line((x, 40), (x, 120), width=0.5)
        for y in (40, 80, 120):
            p.draw_line((40, y), (280, y), width=0.5)
        p.insert_text((50, 64), "H1", fontsize=10)
        p.insert_text((170, 64), "H2", fontsize=10)
        p.insert_text((50, 104), "V1", fontsize=10)
        p.insert_text((170, 104), "V2", fontsize=10)

        # Bottom borderless 2x2 table.
        p.insert_text((50, 224), "B1", fontsize=10)
        p.insert_text((200, 224), "B2", fontsize=10)
        p.insert_text((50, 264), "D1", fontsize=10)
        p.insert_text((200, 264), "D2", fontsize=10)
        b.save(pdf_path)

        with pdfplumber.open(str(pdf_path)) as doc:
            tables = extract_tables_on_page(
                doc.pages[0],
                [(38.0, 38.0, 282.0, 122.0), (38.0, 200.0, 282.0, 280.0)],
                extractor=AutoTableExtractor(tatr_extractor=_FakeTatr()),
            )
    assert len(tables) == 2
    assert tables[0].source == "auto_refined" and tables[0].confidence == 1.0
    assert [c.text for c in tables[0].cells] == ["H1", "H2", "V1", "V2"]
    assert tables[1].source == "auto_tatr" and tables[1].confidence < 1.0
    assert [c.text for c in tables[1].cells] == ["B1", "B2", "D1", "D2"]


def test_auto_raster_and_suspect_ocr_skip_pdfplumber():
    import pdfplumber

    class _FailPlumber:
        def extract_from_page(self, *_args, **_kwargs):
            raise AssertionError("pdfplumber must not be called on raster/suspect pages")

    class _FakeOcr:
        def recognize(self, image: Image.Image):
            cw, ch = float(image.width), float(image.height)
            return [
                OcrLine("OcrH1", (0.05 * cw, 0.1 * ch, 0.4 * cw, 0.4 * ch), 0.99),
                OcrLine("OcrH2", (0.55 * cw, 0.1 * ch, 0.9 * cw, 0.4 * ch), 0.99),
                OcrLine("OcrV1", (0.05 * cw, 0.6 * ch, 0.4 * cw, 0.9 * ch), 0.99),
                OcrLine("OcrV2", (0.55 * cw, 0.6 * ch, 0.9 * cw, 0.9 * ch), 0.99),
            ]

    class _FakeTatr:
        def extract(self, image: Image.Image, table_bbox_pt, tokens=()):
            cw, ch = float(image.width), float(image.height)
            objs = [
                ("table", (0.0, 0.0, cw, ch), 0.92),
                ("table row", (0.0, 0.0, cw, ch * 0.5), 0.9),
                ("table row", (0.0, ch * 0.5, cw, ch), 0.9),
                ("table column", (0.0, 0.0, cw * 0.5, ch), 0.9),
                ("table column", (cw * 0.5, 0.0, cw, ch), 0.9),
                ("table column header", (0.0, 0.0, cw, ch * 0.5), 0.9),
            ]
            return objects_to_table(objs, image.size, table_bbox_pt, tokens)

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "decoy.pdf"
        b = TestPdfBuilder()
        p = b.new_page(width=400, height=300)
        for x in (40, 160, 280):
            p.draw_line((x, 50), (x, 150), width=0.5)
        for y in (50, 100, 150):
            p.draw_line((40, y), (280, y), width=0.5)
        p.insert_text((50, 74), "BAD_TEXT_LAYER", fontsize=10)
        b.save(pdf_path)

        ext = AutoTableExtractor(
            pdfplumber_extractor=_FailPlumber(),
            tatr_extractor=_FakeTatr(),
            ocr_recognizer=_FakeOcr(),
        )
        with pdfplumber.open(str(pdf_path)) as doc:
            tables = extract_tables_on_page(
                doc.pages[0],
                [(38.0, 48.0, 282.0, 152.0)],
                is_raster=True,
                extractor=ext,
            )
    tbl = tables[0]
    assert tbl.source == "auto_tatr"
    assert [c.text for c in tbl.cells] == ["OcrH1", "OcrH2", "OcrV1", "OcrV2"]
    assert all("BAD_TEXT_LAYER" not in c.text for c in tbl.cells)


def test_explicit_pdfplumber_and_tatr_backends_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "explicit.pdf"
        b = TestPdfBuilder()
        p = b.new_page(width=400, height=300)
        # 3x3 grid (8 lines >= MIN_CANDIDATE_DRAWINGS).
        for x in (40, 120, 200, 280):
            p.draw_line((x, 50), (x, 170), width=0.5)
        for y in (50, 90, 130, 170):
            p.draw_line((40, y), (280, y), width=0.5)
        p.insert_text((50, 74), "K1", fontsize=10)
        p.insert_text((130, 74), "K2", fontsize=10)
        p.insert_text((210, 74), "K3", fontsize=10)
        p.insert_text((50, 114), "D1", fontsize=10)
        p.insert_text((130, 114), "D2", fontsize=10)
        p.insert_text((210, 114), "D3", fontsize=10)
        p.insert_text((50, 154), "E1", fontsize=10)
        p.insert_text((130, 154), "E2", fontsize=10)
        p.insert_text((210, 154), "E3", fontsize=10)
        b.save(pdf_path)

        import pdf.figures as figures_mod

        orig_detect = figures_mod.detect_figures
        figures_mod.detect_figures = lambda _p, pages, **_kw: {
            pno: [FigureBox((38.0, 48.0, 282.0, 172.0), "table", "layout_table", 0.95)] for pno in pages
        }
        try:
            ir_pl = ir_mod.extract_ir(pdf_path, figures=True, ocr=False, tables=True, table_backend="pdfplumber")
            assert ir_pl["tables"][0]["metadata"]["source"] == "pdfplumber"

            reset_cache()
            tatr_ext = get("table", "tatr-onnx")
            tatr_ext.ensure_loaded = lambda: False
            tatr_ext.detect_objects = lambda crop: [
                ("table", (0.0, 0.0, float(crop.width), float(crop.height)), 0.93),
                ("table row", (0.0, 0.0, float(crop.width), crop.height * 0.5), 0.9),
                ("table row", (0.0, crop.height * 0.5, float(crop.width), float(crop.height)), 0.9),
                ("table column", (0.0, 0.0, crop.width * 0.5, float(crop.height)), 0.9),
                ("table column", (crop.width * 0.5, 0.0, float(crop.width), float(crop.height)), 0.9),
                ("table column header", (0.0, 0.0, float(crop.width), crop.height * 0.5), 0.9),
            ]
            ir_tatr = ir_mod.extract_ir(pdf_path, figures=True, ocr=False, tables=True, table_backend="tatr-onnx")
            assert ir_tatr["tables"][0]["metadata"]["source"] == "tatr_onnx"
        finally:
            figures_mod.detect_figures = orig_detect
            reset_cache()


def test_table_ir_top_level_fields_and_empty_table_warning():
    from pdf.tables import table_to_ir_dict

    tbl = ExtractedTable(
        bbox=(10.0, 20.0, 110.0, 80.0),
        num_rows=1,
        num_cols=2,
        cells=(
            TableCell(0, 0, 1, 1, "A", (10.0, 20.0, 60.0, 80.0), True),
            TableCell(0, 1, 1, 1, "B", (60.0, 20.0, 110.0, 80.0), True),
        ),
        source="auto_refined",
        confidence=0.92,
    )
    ir_tbl = table_to_ir_dict("tbl_0001", "elem_0001", 1, "doc.pdf", tbl)
    assert ir_tbl["source"] == "auto_refined"
    assert ir_tbl["confidence"] == 0.92
    assert ir_tbl["metadata"]["source"] == "auto_refined"
    assert ir_tbl["metadata"]["confidence"] == 0.92
    assert all(c["confidence"] == 1.0 for c in ir_tbl["cells"])
    tbl_custom_conf = ExtractedTable(
        bbox=(10.0, 20.0, 110.0, 80.0),
        num_rows=1,
        num_cols=2,
        cells=(
            TableCell(0, 0, 1, 1, "A", (10.0, 20.0, 60.0, 80.0), True, confidence=0.875),
            TableCell(0, 1, 1, 1, "B", (60.0, 20.0, 110.0, 80.0), True, confidence=0.9125),
        ),
        source="tatr_onnx",
        confidence=0.9,
    )
    ir_custom = table_to_ir_dict("tbl_0002", "elem_0002", 1, "doc.pdf", tbl_custom_conf)
    assert [c["confidence"] for c in ir_custom["cells"]] == [0.875, 0.9125]
    assert isinstance(get("table", "tatr-onnx"), TableExtractor)
    assert DiagramExtractor is not None

    class _EmptyExtractor:
        name = "empty_tbl"
        last_stats = None

        def extract(self, image: Image.Image, table_bbox_pt, tokens=()):
            del image, tokens
            return ExtractedTable(table_bbox_pt, 0, 0, (), "empty_tbl", 0.0)

        def extract_from_page(self, page, table_boxes_pt, **_kw):
            del page
            warn = {"page": 1, "code": "backend_note", "severity": "info", "reason": "custom_stats_warning", "fallback": None}
            self.last_stats = RunStats(pages=1, in_process=True, warnings=(warn,))
            return [ExtractedTable(b, 0, 0, (), "empty_tbl", 0.0) for b in table_boxes_pt]

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "empty_tbl.pdf"
        b = TestPdfBuilder()
        p = b.new_page(width=400, height=300)
        for i in range(8):
            p.draw_rect((40 + i * 25, 50, 60 + i * 25, 70))
        p.insert_text((40, 200), "Body text", fontsize=10)
        b.save(pdf_path)

        import pdf.figures as figures_mod

        orig_detect = figures_mod.detect_figures
        figures_mod.detect_figures = lambda _p, pages, **_kw: {
            pno: [FigureBox((38.0, 48.0, 250.0, 120.0), "table", "layout_table", 0.88)] for pno in pages
        }
        try:
            ir = ir_mod.extract_ir(
                pdf_path,
                figures=True,
                ocr=False,
                tables=True,
                diagrams=False,
                diagram_backend="vector_first",
                table_backend=_EmptyExtractor(),
            )
        finally:
            figures_mod.detect_figures = orig_detect

    assert ir["tables"] == []
    assert ir["diagrams"] == []
    assert any(w.get("code") == "table_parse_empty" and w.get("page") == 1 for w in ir["metadata"]["warnings"])
    assert any(w.get("code") == "backend_note" for w in ir["metadata"]["warnings"])


def test_table_to_html_preserves_full_rowspan_tr_and_br_whitespace():
    from pdf.eval.tables import parse_table_html, teds_score
    from pdf.models import ExtractedTable, TableCell

    tbl_1col = ExtractedTable(
        bbox=(0.0, 0.0, 100.0, 100.0),
        num_rows=2,
        num_cols=1,
        cells=(TableCell(0, 0, 2, 1, "A", (0.0, 0.0, 100.0, 100.0)),),
        source="tatr_onnx",
        confidence=0.9,
    )
    html_1col = table_to_html(tbl_1col)
    assert html_1col == '<table><tr><td rowspan="2">A</td></tr><tr></tr></table>'
    assert teds_score(html_1col, '<table><tr><td rowspan="2">A</td></tr><tr></tr></table>', structure_only=True) == 1.0

    tbl_2col = ExtractedTable(
        bbox=(0.0, 0.0, 100.0, 100.0),
        num_rows=2,
        num_cols=2,
        cells=(
            TableCell(0, 0, 2, 1, "A", (0.0, 0.0, 50.0, 100.0)),
            TableCell(0, 1, 2, 1, "B", (50.0, 0.0, 100.0, 100.0)),
        ),
        source="tatr_onnx",
        confidence=0.9,
    )
    html_2col = table_to_html(tbl_2col)
    assert html_2col.count("<tr>") == 2
    assert (
        teds_score(
            html_2col,
            '<table><tr><td rowspan="2">A</td><td rowspan="2">B</td></tr><tr></tr></table>',
            structure_only=True,
        )
        == 1.0
    )

    root = parse_table_html("<table><tr><td>hello<br>world<br/>again</td></tr></table>")
    assert root.children[0].children[0].text == "hello world again"
    assert (
        teds_score(
            "<table><tr><td>hello<br>world</td></tr></table>",
            "<table><tr><td>hello world</td></tr></table>",
            structure_only=False,
        )
        == 1.0
    )


def test_table_to_html_thead_tbody_and_teds_parity():
    from pdf.eval.tables import parse_table_html, teds_score

    tbl = ExtractedTable(
        bbox=(0.0, 0.0, 100.0, 90.0),
        num_rows=3,
        num_cols=2,
        cells=(
            TableCell(0, 0, 1, 1, "Key", (0.0, 0.0, 50.0, 30.0), is_header=True, confidence=0.95),
            TableCell(0, 1, 1, 1, "Val", (50.0, 0.0, 100.0, 30.0), is_header=True, confidence=0.95),
            TableCell(1, 0, 1, 1, "A", (0.0, 30.0, 50.0, 60.0), is_header=False, confidence=0.9),
            TableCell(1, 1, 1, 1, "1", (50.0, 30.0, 100.0, 60.0), is_header=False, confidence=0.9),
            TableCell(2, 0, 1, 1, "B", (0.0, 60.0, 50.0, 90.0), is_header=False, confidence=0.9),
            TableCell(2, 1, 1, 1, "2", (50.0, 60.0, 100.0, 90.0), is_header=False, confidence=0.9),
        ),
        source="tatr_onnx",
        confidence=0.94,
    )
    out_html = table_to_html(tbl)
    assert out_html == (
        "<table><thead><tr><th>Key</th><th>Val</th></tr></thead>"
        "<tbody><tr><td>A</td><td>1</td></tr><tr><td>B</td><td>2</td></tr></tbody></table>"
    )
    root = parse_table_html(out_html)
    assert len(root.children) == 3
    assert [c.text for c in root.children[0].children] == ["Key", "Val"]
    assert teds_score(out_html, "<table><tr><td>Key</td><td>Val</td></tr><tr><td>A</td><td>1</td></tr><tr><td>B</td><td>2</td></tr></table>") == 1.0


def test_grid_occupancy_invariant_and_cell_confidence():
    import pdfplumber

    objects = [
        ("table", (0.0, 0.0, 200.0, 120.0), 0.96),
        ("table row", (0.0, 0.0, 200.0, 40.0), 0.92),
        ("table row", (0.0, 40.0, 200.0, 80.0), 0.88),
        ("table row", (0.0, 80.0, 200.0, 120.0), 0.84),
        ("table column", (0.0, 0.0, 100.0, 120.0), 0.90),
        ("table column", (100.0, 0.0, 200.0, 120.0), 0.86),
        ("table column header", (0.0, 0.0, 200.0, 40.0), 0.91),
        ("table spanning cell", (0.0, 0.0, 200.0, 80.0), 0.94),
        # Overlapping lower-confidence spanning cell must be skipped without double-occupying slots.
        ("table spanning cell", (100.0, 40.0, 200.0, 120.0), 0.72),
    ]
    tbl = objects_to_table(objects, (200, 120), (0.0, 0.0, 200.0, 120.0), ())
    assert tbl.num_rows == 3 and tbl.num_cols == 2
    occupied: set[tuple[int, int]] = set()
    for cell in tbl.cells:
        assert 0.0 < cell.confidence <= 1.0
        assert cell.row + cell.rowspan <= tbl.num_rows
        assert cell.col + cell.colspan <= tbl.num_cols
        for r in range(cell.row, cell.row + cell.rowspan):
            for c in range(cell.col, cell.col + cell.colspan):
                assert (r, c) not in occupied
                occupied.add((r, c))
    assert len(occupied) == tbl.num_rows * tbl.num_cols

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "pl_conf.pdf"
        b = TestPdfBuilder()
        p = b.new_page(width=300, height=200)
        for x in (40, 120, 200):
            p.draw_line((x, 40), (x, 120), width=0.5)
        for y in (40, 80, 120):
            p.draw_line((40, y), (200, y), width=0.5)
        p.insert_text((50, 64), "H1", fontsize=10)
        p.insert_text((130, 64), "H2", fontsize=10)
        p.insert_text((50, 104), "V1", fontsize=10)
        p.insert_text((130, 104), "V2", fontsize=10)
        b.save(pdf_path)
        with pdfplumber.open(str(pdf_path)) as doc:
            pl_tbl = extract_tables_on_page(doc.pages[0], [(38.0, 38.0, 202.0, 122.0)], extractor="pdfplumber")[0]
    assert all(c.confidence == 1.0 for c in pl_tbl.cells)


def test_token_aware_snapping_and_multicell_split():
    # Token "LineTwo" straddles the nominal y=50 midpoint (row0 ends at 46, row1 starts at 54) -> boundary nudges above/below it.
    objects = [
        ("table", (0.0, 0.0, 200.0, 100.0), 0.95),
        ("table row", (0.0, 0.0, 200.0, 46.0), 0.94),
        ("table row", (0.0, 54.0, 200.0, 100.0), 0.92),
        ("table column", (0.0, 0.0, 100.0, 100.0), 0.93),
        ("table column", (100.0, 0.0, 200.0, 100.0), 0.93),
        ("table column header", (0.0, 0.0, 200.0, 46.0), 0.93),
    ]
    tokens = [
        ("HdrA", (10.0, 10.0, 50.0, 24.0)),
        ("HdrB", (120.0, 10.0, 160.0, 24.0)),
        ("LineTwo", (10.0, 45.0, 70.0, 62.0)),
        ("Right1 Right2", (20.0, 72.0, 175.0, 88.0)),
    ]
    tbl = objects_to_table(objects, (200, 100), (0.0, 0.0, 200.0, 100.0), tokens)
    by_rc = {(c.row, c.col): c for c in tbl.cells}
    assert by_rc[(1, 0)].text == "LineTwo Right1"
    assert by_rc[(1, 1)].text == "Right2"


def test_auto_token_coverage_gate_and_1x1_callout():
    import pdfplumber

    class _BadCoveragePlumber:
        def extract_from_page(self, _page, table_boxes_pt, **_kw):
            b = table_boxes_pt[0]
            return [
                ExtractedTable(
                    bbox=b,
                    num_rows=2,
                    num_cols=2,
                    cells=(
                        TableCell(0, 0, 1, 1, "OnlyOne", (b[0], b[1], b[0] + 40.0, b[1] + 20.0), True),
                        TableCell(0, 1, 1, 1, "", (b[0] + 40.0, b[1], b[2], b[1] + 20.0), True),
                        TableCell(1, 0, 1, 1, "", (b[0], b[1] + 20.0, b[0] + 40.0, b[3]), False),
                        TableCell(1, 1, 1, 1, "", (b[0] + 40.0, b[1] + 20.0, b[2], b[3]), False),
                    ),
                    source="pdfplumber",
                    confidence=1.0,
                )
            ]

    class _FakeTatr:
        def __init__(self, split_grid: bool):
            self.split_grid = split_grid

        def detect_objects(self, crop: Image.Image):
            cw, ch = float(crop.width), float(crop.height)
            if not self.split_grid:
                return [("table", (0.0, 0.0, cw, ch), 0.85)]
            return [
                ("table", (0.0, 0.0, cw, ch), 0.92),
                ("table row", (0.0, 0.0, cw, ch * 0.5), 0.90),
                ("table row", (0.0, ch * 0.5, cw, ch), 0.90),
                ("table column", (0.0, 0.0, cw * 0.5, ch), 0.90),
                ("table column", (cw * 0.5, 0.0, cw, ch), 0.90),
                ("table column header", (0.0, 0.0, cw, ch * 0.5), 0.90),
            ]

        def extract(self, image: Image.Image, table_bbox_pt, tokens=()):
            return objects_to_table(self.detect_objects(image), image.size, table_bbox_pt, tokens)

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "gate.pdf"
        b = TestPdfBuilder()
        p1 = b.new_page(width=400, height=300)
        p1.insert_text((50, 64), "OnlyOne", fontsize=10)
        p1.insert_text((170, 64), "Miss1", fontsize=10)
        p1.insert_text((50, 114), "Miss2", fontsize=10)
        p1.insert_text((170, 114), "Miss3", fontsize=10)

        p2 = b.new_page(width=400, height=300)
        p2.draw_rect((40, 50, 260, 110), width=0.5)
        p2.insert_text((55, 82), "Important note callout", fontsize=10)
        b.save(pdf_path)

        with pdfplumber.open(str(pdf_path)) as doc:
            gate_tbl = extract_tables_on_page(
                doc.pages[0],
                [(40.0, 45.0, 250.0, 130.0)],
                extractor=AutoTableExtractor(
                    pdfplumber_extractor=_BadCoveragePlumber(),
                    tatr_extractor=_FakeTatr(split_grid=True),
                ),
            )[0]
            assert gate_tbl.source == "auto_tatr"
            assert [c.text for c in gate_tbl.cells] == ["OnlyOne", "Miss1", "Miss2", "Miss3"]

            callout_tbl = extract_tables_on_page(
                doc.pages[1],
                [(38.0, 48.0, 262.0, 112.0)],
                extractor=AutoTableExtractor(tatr_extractor=_FakeTatr(split_grid=False)),
            )[0]
            assert callout_tbl.num_rows == 1 and callout_tbl.num_cols == 1
            assert callout_tbl.cells[0].is_header is False
            assert "Important note callout" in callout_tbl.cells[0].text
            assert 0.0 < callout_tbl.confidence < 1.0


def test_link_table_continuations_positive_and_negative():
    from pdf.tables import link_table_continuations

    def _mk_tbl(tid, eid, page, bbox, ncols, hdr_texts, is_hdr=True):
        cells = [
            {"row": 0, "col": c, "rowspan": 1, "colspan": 1, "text": hdr_texts[c], "is_header": is_hdr}
            for c in range(ncols)
        ]
        return {
            "id": tid,
            "page": page,
            "bbox": list(bbox),
            "num_rows": 2,
            "num_cols": ncols,
            "continuation_of": None,
            "cells": cells,
            "metadata": {"element_id": eid},
        }

    # Positive 1: repeated header row at bottom->top margin; pure (input unchanged).
    t1 = _mk_tbl("tbl_0001", "e1", 1, (40, 500, 500, 740), 2, ["Param", "Value"], True)
    t2 = _mk_tbl("tbl_0002", "e2", 2, (40, 40, 500, 260), 2, ["Param", "Value"], True)
    # Positive 2: headerless continuation on page 3.
    t2b = _mk_tbl("tbl_0002", "e2", 2, (40, 520, 500, 740), 2, ["Param", "Value"], True)
    t3 = _mk_tbl("tbl_0003", "e3", 3, (40, 40, 500, 220), 2, ["foo", "bar"], False)
    elems_pos = [
        {"id": "e1", "page": 1, "type": "table", "bbox": [40, 500, 500, 740]},
        {"id": "f1", "page": 1, "type": "footer", "bbox": [40, 760, 500, 780]},
        {"id": "h2", "page": 2, "type": "header", "bbox": [40, 15, 500, 30]},
        {"id": "e2", "page": 2, "type": "table", "bbox": [40, 40, 500, 260]},
    ]
    linked = link_table_continuations([t1, t2], elems_pos)
    assert t2["continuation_of"] is None
    assert linked[0]["continuation_of"] is None
    assert linked[1]["continuation_of"] == "tbl_0001"

    elems_hdrless = [
        {"id": "e2", "page": 2, "type": "table", "bbox": [40, 520, 500, 740]},
        {"id": "e3", "page": 3, "type": "table", "bbox": [40, 40, 500, 220]},
    ]
    linked_hdrless = link_table_continuations([t2b, t3], elems_hdrless)
    assert linked_hdrless[1]["continuation_of"] == "tbl_0002"

    # Negative 1: mismatched num_cols.
    t_cols = _mk_tbl("tbl_0002", "e2", 2, (40, 40, 500, 260), 3, ["Param", "Value", "Unit"], True)
    assert link_table_continuations([t1, t_cols], elems_pos)[1]["continuation_of"] is None

    # Negative 2: intervening heading or body text between tables.
    elems_heading = [
        {"id": "e1", "page": 1, "type": "table", "bbox": [40, 500, 500, 740]},
        {"id": "hd", "page": 2, "type": "heading", "bbox": [40, 20, 300, 36]},
        {"id": "e2", "page": 2, "type": "table", "bbox": [40, 40, 500, 260]},
    ]
    assert link_table_continuations([t1, t2], elems_heading)[1]["continuation_of"] is None

    # Negative 3: different header row texts.
    t_diff_hdr = _mk_tbl("tbl_0002", "e2", 2, (40, 40, 500, 260), 2, ["Other", "Cols"], True)
    assert link_table_continuations([t1, t_diff_hdr], elems_pos)[1]["continuation_of"] is None

    # Negative 4: non-consecutive pages (page 1 -> page 3).
    t_p3 = _mk_tbl("tbl_0002", "e2", 3, (40, 40, 500, 260), 2, ["Param", "Value"], True)
    assert link_table_continuations([t1, t_p3], elems_pos)[1]["continuation_of"] is None

    # Negative 5: apply_structure() heading (type="text" with metadata role="heading" / heading_level).
    elems_role_heading = [
        {"id": "e1", "page": 1, "type": "table", "bbox": [40, 500, 500, 740]},
        {"id": "hd_role", "page": 1, "type": "text", "bbox": [300, 100, 520, 125], "metadata": {"role": "heading"}},
        {"id": "e2", "page": 2, "type": "table", "bbox": [40, 40, 500, 260]},
    ]
    assert link_table_continuations([t1, t2], elems_role_heading)[1]["continuation_of"] is None

    elems_level_heading = [
        {"id": "e1", "page": 1, "type": "table", "bbox": [40, 500, 500, 740]},
        {"id": "hd_lvl", "page": 1, "type": "text", "bbox": [300, 100, 520, 125], "metadata": {"heading_level": 2}},
        {"id": "e2", "page": 2, "type": "table", "bbox": [40, 40, 500, 260]},
    ]
    assert link_table_continuations([t1, t2], elems_level_heading)[1]["continuation_of"] is None


def test_table_to_html_never_splits_rowspan_across_thead_tbody():
    # Header cell spanning into a body row must not be split across <thead>/<tbody>.
    tbl_split = ExtractedTable(
        bbox=(0.0, 0.0, 100.0, 60.0),
        num_rows=2,
        num_cols=2,
        cells=(
            TableCell(0, 0, 2, 1, "Metric", (0.0, 0.0, 50.0, 60.0), is_header=True),
            TableCell(0, 1, 1, 1, "Values", (50.0, 0.0, 100.0, 30.0), is_header=True),
            TableCell(1, 1, 1, 1, "10", (50.0, 30.0, 100.0, 60.0), is_header=False),
        ),
        source="tatr_onnx",
        confidence=0.9,
    )
    html_split = table_to_html(tbl_split)
    assert "<thead>" not in html_split and "<tbody>" not in html_split
    assert html_split == '<table><tr><th rowspan="2">Metric</th><th>Values</th></tr><tr><td>10</td></tr></table>'
    assert teds_score(html_split, '<table><tr><td rowspan="2">Metric</td><td>Values</td></tr><tr><td>10</td></tr></table>') == 1.0

    # Row 0 stays in <thead> while row 1 rowspan=2 moves with row 2 into <tbody>.
    tbl_partial = ExtractedTable(
        bbox=(0.0, 0.0, 100.0, 90.0),
        num_rows=3,
        num_cols=2,
        cells=(
            TableCell(0, 0, 1, 1, "H0", (0.0, 0.0, 50.0, 30.0), is_header=True),
            TableCell(0, 1, 1, 1, "H1", (50.0, 0.0, 100.0, 30.0), is_header=True),
            TableCell(1, 0, 2, 1, "Sub", (0.0, 30.0, 50.0, 90.0), is_header=True),
            TableCell(1, 1, 1, 1, "S1", (50.0, 30.0, 100.0, 60.0), is_header=True),
            TableCell(2, 1, 1, 1, "10", (50.0, 60.0, 100.0, 90.0), is_header=False),
        ),
        source="tatr_onnx",
        confidence=0.9,
    )
    html_partial = table_to_html(tbl_partial)
    assert html_partial == (
        "<table><thead><tr><th>H0</th><th>H1</th></tr></thead>"
        '<tbody><tr><th rowspan="2">Sub</th><th>S1</th></tr><tr><td>10</td></tr></tbody></table>'
    )

    # All-header multi-row group stays inside <thead> cleanly.
    tbl_hdr2 = ExtractedTable(
        bbox=(0.0, 0.0, 100.0, 90.0),
        num_rows=3,
        num_cols=2,
        cells=(
            TableCell(0, 0, 2, 1, "Metric", (0.0, 0.0, 50.0, 60.0), is_header=True),
            TableCell(0, 1, 1, 1, "Group", (50.0, 0.0, 100.0, 30.0), is_header=True),
            TableCell(1, 1, 1, 1, "Sub", (50.0, 30.0, 100.0, 60.0), is_header=True),
            TableCell(2, 0, 1, 1, "A", (0.0, 60.0, 50.0, 90.0), is_header=False),
            TableCell(2, 1, 1, 1, "1", (50.0, 60.0, 100.0, 90.0), is_header=False),
        ),
        source="tatr_onnx",
        confidence=0.9,
    )
    assert table_to_html(tbl_hdr2) == (
        '<table><thead><tr><th rowspan="2">Metric</th><th>Group</th></tr><tr><th>Sub</th></tr></thead>'
        "<tbody><tr><td>A</td><td>1</td></tr></tbody></table>"
    )


def test_auto_rejects_partial_pdfplumber_table_against_layout_box():
    import pdfplumber

    class _FourRowTatr:
        def __init__(self):
            self.extract_calls = 0

        def _objs(self, cw: float, ch: float):
            return [
                ("table", (0.0, 0.0, cw, ch), 0.94),
                ("table row", (0.0, 0.0, cw, ch * 0.25), 0.92),
                ("table row", (0.0, ch * 0.25, cw, ch * 0.50), 0.91),
                ("table row", (0.0, ch * 0.50, cw, ch * 0.75), 0.90),
                ("table row", (0.0, ch * 0.75, cw, ch), 0.89),
                ("table column", (0.0, 0.0, cw * 0.5, ch), 0.93),
                ("table column", (cw * 0.5, 0.0, cw, ch), 0.93),
                ("table column header", (0.0, 0.0, cw, ch * 0.25), 0.92),
            ]

        def detect_objects(self, crop: Image.Image):
            return self._objs(float(crop.width), float(crop.height))

        def extract(self, image: Image.Image, table_bbox_pt, tokens=()):
            self.extract_calls += 1
            return objects_to_table(self._objs(float(image.width), float(image.height)), image.size, table_bbox_pt, tokens)

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "partial_ruled.pdf"
        b = TestPdfBuilder()
        p = b.new_page(width=400, height=300)
        # Only top 2 of 4 rows are ruled (IoU = 0.50); bottom 2 rows are unruled.
        for x in (40, 160, 280):
            p.draw_line((x, 40), (x, 120), width=0.5)
        for y in (40, 80, 120):
            p.draw_line((40, y), (280, y), width=0.5)
        p.insert_text((55, 64), "H1", fontsize=10)
        p.insert_text((175, 64), "H2", fontsize=10)
        p.insert_text((55, 104), "R1", fontsize=10)
        p.insert_text((175, 104), "V1", fontsize=10)
        p.insert_text((55, 144), "R2", fontsize=10)
        p.insert_text((175, 144), "V2", fontsize=10)
        p.insert_text((55, 184), "R3", fontsize=10)
        p.insert_text((175, 184), "V3", fontsize=10)
        b.save(pdf_path)

        tatr = _FourRowTatr()
        with pdfplumber.open(str(pdf_path)) as doc:
            tbl = extract_tables_on_page(
                doc.pages[0],
                [(40.0, 40.0, 280.0, 200.0)],
                extractor=AutoTableExtractor(tatr_extractor=tatr),
            )[0]

    assert tatr.extract_calls == 1
    assert tbl.source == "auto_tatr"
    assert tbl.num_rows == 4 and tbl.num_cols == 2
    assert [c.text for c in tbl.cells] == ["H1", "H2", "R1", "V1", "R2", "V2", "R3", "V3"]


def test_auto_token_coverage_handles_tight_border_merged_words():
    import pdfplumber
    from pdf.backends.table_auto import _token_coverage

    class _NoFallbackTatr:
        def detect_objects(self, crop: Image.Image):
            cw, ch = float(crop.width), float(crop.height)
            return [
                ("table", (0.0, 0.0, cw, ch), 0.95),
                ("table row", (0.0, 0.0, cw, ch * 0.5), 0.92),
                ("table row", (0.0, ch * 0.5, cw, ch), 0.92),
                ("table column header", (0.0, 0.0, cw, ch * 0.5), 0.92),
            ]

        def extract(self, *_args, **_kwargs):
            raise AssertionError("TATR extract() fallback must not be called for tight-border words")

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "tight_border.pdf"
        b = TestPdfBuilder()
        p = b.new_page(width=300, height=200)
        for x in (40, 100, 180):
            p.draw_line((x, 40), (x, 120), width=0.5)
        for y in (40, 80, 120):
            p.draw_line((40, y), (180, y), width=0.5)
        # Gap across x=100 is < 2pt so extract_words merges ColA+ColB, while pdfplumber splits them.
        p.insert_text((77, 64), "ColA", fontsize=10)
        p.insert_text((100.7, 64), "ColB", fontsize=10)
        p.insert_text((55, 104), "Val1", fontsize=10)
        p.insert_text((115, 104), "Val2", fontsize=10)
        b.save(pdf_path)

        with pdfplumber.open(str(pdf_path)) as doc:
            tbl = extract_tables_on_page(
                doc.pages[0],
                [(38.0, 38.0, 182.0, 122.0)],
                extractor=AutoTableExtractor(tatr_extractor=_NoFallbackTatr()),
            )[0]

    assert tbl.source == "auto_refined"
    assert [c.text for c in tbl.cells] == ["ColA", "ColB", "Val1", "Val2"]

    # Multi-line cells where top line tokens fused across vertical cell border also reach 1.0 coverage.
    multi_tbl = ExtractedTable(
        bbox=(40.0, 40.0, 180.0, 120.0),
        num_rows=1,
        num_cols=2,
        cells=(
            TableCell(0, 0, 1, 1, "ColA foo", (40.0, 40.0, 100.0, 120.0), True),
            TableCell(0, 1, 1, 1, "ColB bar", (100.0, 40.0, 180.0, 120.0), True),
        ),
        source="pdfplumber",
        confidence=1.0,
    )
    assert _token_coverage(
        multi_tbl,
        [("ColAColB", (77.0, 50.0, 123.0, 62.0)), ("foo", (77.0, 75.0, 95.0, 87.0)), ("bar", (105.0, 75.0, 125.0, 87.0))],
    ) == 1.0


def test_tatr_1d_clustering_and_snap_intervals_boundaries():
    from pdf.backends.tatr_onnx import _cluster_1d_tokens, _snap_intervals

    clustered = _cluster_1d_tokens(
        [
            (-10.0, -2.0, 5.0, 10.0),
            (105.0, 10.0, 120.0, 20.0),
            (-3.0, 15.0, 20.0, 25.0),
            (40.0, 30.0, 40.0, 40.0),
            (85.0, 50.0, 110.0, 60.0),
        ],
        axis=0,
        limit=100.0,
    )
    assert clustered == [(0.0, 20.0, 0.6), (85.0, 100.0, 0.6)]

    for raw, lim in [
        ([(0.0, 100.0, 0.9), (105.0, 120.0, 0.8)], 100.0),
        ([(0.0, 99.5, 0.9), (99.8, 100.0, 0.9)], 100.0),
        ([(0.0, 60.0, 0.9), (40.0, 100.0, 0.9)], 100.0),
        ([(0.0, 1.0, 0.9), (1.0, 2.0, 0.9), (2.0, 3.0, 0.9)], 1.5),
    ]:
        snapped = _snap_intervals(raw, lim)
        assert snapped
        assert snapped[0][0] == 0.0 and snapped[-1][1] == lim
        for idx, (lo, hi) in enumerate(snapped):
            assert 0.0 <= lo < hi <= lim
            if idx > 0:
                assert snapped[idx - 1][1] <= lo


def test_tatr_projected_header_filter_and_multiline_row_collapse():
    # Reject projected row header when tokens span >=2 separated columns; collapse over-segmented multi-line body row.
    objects = [
        ("table", (0.0, 0.0, 300.0, 180.0), 0.99),
        ("table column", (0.0, 0.0, 100.0, 180.0), 0.95),
        ("table column", (100.0, 0.0, 200.0, 180.0), 0.95),
        ("table column", (200.0, 0.0, 300.0, 180.0), 0.95),
        ("table row", (0.0, 0.0, 300.0, 40.0), 0.96),
        ("table row", (0.0, 40.0, 300.0, 80.0), 0.94),
        ("table row", (0.0, 80.0, 300.0, 120.0), 0.94),
        ("table row", (0.0, 120.0, 300.0, 180.0), 0.95),
        ("table column header", (0.0, 0.0, 300.0, 40.0), 0.96),
        ("table spanning cell", (0.0, 40.0, 100.0, 120.0), 0.92),
        ("table spanning cell", (200.0, 40.0, 300.0, 120.0), 0.91),
        ("table projected row header", (0.0, 120.0, 300.0, 180.0), 0.88),
    ]
    tokens = [
        ("Study", (10.0, 10.0, 60.0, 28.0)),
        ("Notes", (110.0, 10.0, 160.0, 28.0)),
        ("Result", (210.0, 10.0, 260.0, 28.0)),
        ("S1", (15.0, 60.0, 45.0, 85.0)),
        ("LineA", (110.0, 48.0, 160.0, 68.0)),
        ("LineB", (110.0, 88.0, 160.0, 108.0)),
        ("Pass", (215.0, 60.0, 260.0, 85.0)),
        ("S2", (15.0, 140.0, 45.0, 160.0)),
        ("Mid", (115.0, 140.0, 155.0, 160.0)),
        ("Fail", (215.0, 140.0, 255.0, 160.0)),
    ]
    tbl = objects_to_table(objects, (300, 180), (0.0, 0.0, 300.0, 180.0), tokens)
    assert tbl.num_rows == 3 and tbl.num_cols == 3
    by_rc = {(c.row, c.col): c for c in tbl.cells}
    assert by_rc[(1, 0)].rowspan == 1 and by_rc[(1, 0)].text == "S1"
    assert by_rc[(1, 1)].text == "LineA LineB"
    assert by_rc[(2, 0)].colspan == 1 and by_rc[(2, 0)].text == "S2"
    assert by_rc[(2, 1)].text == "Mid"
    assert by_rc[(2, 2)].text == "Fail"


def test_tatr_undersegmented_splits_dot_leaders_and_vertical_headers():
    # Column 1 splits via body gutter; row 1 splits via aligned Y-bands; dot-leader splits across cols; vertical header sorts bottom-to-top.
    objects = [
        ("table", (0.0, 0.0, 300.0, 220.0), 0.99),
        ("table column", (0.0, 0.0, 100.0, 220.0), 0.95),
        ("table column", (100.0, 0.0, 300.0, 220.0), 0.95),
        ("table row", (0.0, 0.0, 300.0, 70.0), 0.96),
        ("table row", (0.0, 70.0, 300.0, 220.0), 0.94),
        ("table column header", (0.0, 0.0, 300.0, 70.0), 0.95),
    ]
    tokens = [
        ("generation", (20.0, 5.0, 40.0, 35.0)),
        ("Key", (20.0, 38.0, 40.0, 65.0)),
        ("Pts", (130.0, 20.0, 160.0, 42.0)),
        ("Team", (230.0, 20.0, 275.0, 42.0)),
        ("Alpha", (15.0, 80.0, 70.0, 100.0)),
        ("7½.....Coastal Carolina", (125.0, 80.0, 290.0, 100.0)),
        ("Beta", (15.0, 130.0, 65.0, 150.0)),
        ("3", (135.0, 130.0, 150.0, 150.0)),
        ("Miami", (230.0, 130.0, 280.0, 150.0)),
        ("Gamma", (15.0, 180.0, 70.0, 200.0)),
        ("5", (135.0, 180.0, 150.0, 200.0)),
        ("Duke", (230.0, 180.0, 275.0, 200.0)),
    ]
    tbl = objects_to_table(objects, (300, 220), (0.0, 0.0, 300.0, 220.0), tokens)
    assert tbl.num_rows == 4 and tbl.num_cols == 3
    by_rc = {(c.row, c.col): c for c in tbl.cells}
    assert by_rc[(0, 0)].text == "Key generation"
    assert by_rc[(1, 1)].text == "7½....."
    assert by_rc[(1, 2)].text == "Coastal Carolina"
    assert by_rc[(2, 1)].text == "3" and by_rc[(2, 2)].text == "Miami"
    assert by_rc[(3, 1)].text == "5" and by_rc[(3, 2)].text == "Duke"


def test_tatr_rotated_90_orientation_calibration_and_auto_1xn():
    import pdfplumber
    from pdf.backends.tatr_onnx import TatrOnnxTableExtractor

    # 90deg CCW rotated 2x2 table in a (120x200) crop; vertical tokens trigger CW 90deg rotation before TATR.
    class _RotTatr(TatrOnnxTableExtractor):
        def __init__(self):
            super().__init__()
            self.seen_sizes = []

        def ensure_loaded(self) -> bool:
            return False

        def detect_objects(self, crop: Image.Image):
            self.seen_sizes.append(crop.size)
            cw, ch = float(crop.width), float(crop.height)
            return [
                ("table", (0.0, 0.0, cw, ch), 0.9999),
                ("table row", (0.0, 0.0, cw, ch * 0.5), 0.99),
                ("table row", (0.0, ch * 0.5, cw, ch), 0.99),
                ("table column", (0.0, 0.0, cw * 0.5, ch), 0.99),
                ("table column", (cw * 0.5, 0.0, cw, ch), 0.99),
                ("table column header", (0.0, 0.0, cw, ch * 0.5), 0.99),
            ]

    ext = _RotTatr()
    img = Image.new("RGB", (120, 200), (255, 255, 255))
    # Vertical tokens in original (120x200) -> after CW 90deg in (200x120): (200-y1, x0, 200-y0, x1).
    rot_tokens = [
        ("HdrA", (15.0, 140.0, 45.0, 190.0)),
        ("HdrB", (15.0, 10.0, 45.0, 70.0)),
        ("Val1", (75.0, 140.0, 105.0, 190.0)),
        ("Val2", (75.0, 10.0, 105.0, 70.0)),
        ("Extra", (75.0, 145.0, 105.0, 185.0)),
    ]
    tbl = ext.extract(img, (0.0, 0.0, 120.0, 200.0), rot_tokens)
    assert ext.seen_sizes == [(200, 120)]
    assert tbl.num_rows == 2 and tbl.num_cols == 2
    assert tbl.confidence <= 0.96 and all(c.confidence <= 0.96 for c in tbl.cells)
    by_rc = {(c.row, c.col): c for c in tbl.cells}
    assert by_rc[(0, 0)].text == "HdrA" and by_rc[(0, 1)].text == "HdrB"
    assert by_rc[(0, 0)].bbox == (0.0, 100.0, 60.0, 200.0)

    # Low-confidence grid scores strictly lower than clean high-confidence grid.
    degraded = objects_to_table(
        [
            ("table", (0.0, 0.0, 100.0, 80.0), 0.99),
            ("table row", (0.0, 0.0, 100.0, 40.0), 0.62),
            ("table row", (0.0, 40.0, 100.0, 80.0), 0.62),
            ("table column", (0.0, 0.0, 50.0, 80.0), 0.64),
            ("table column", (50.0, 0.0, 100.0, 80.0), 0.64),
        ],
        (100, 80),
        (0.0, 0.0, 100.0, 80.0),
        (),
    )
    assert degraded.confidence < tbl.confidence

    # AutoTableExtractor 1xN (1x2) falls back to TATR when TATR recovers >=2x2.
    class _Mock1x2Plumber:
        def extract_from_page(self, _page, table_boxes_pt, **_kw):
            b = table_boxes_pt[0]
            return [
                ExtractedTable(
                    bbox=b,
                    num_rows=1,
                    num_cols=2,
                    cells=(
                        TableCell(0, 0, 1, 1, "H1 R1", (b[0], b[1], (b[0] + b[2]) * 0.5, b[3]), True),
                        TableCell(0, 1, 1, 1, "H2 V1", ((b[0] + b[2]) * 0.5, b[1], b[2], b[3]), True),
                    ),
                    source="pdfplumber",
                    confidence=1.0,
                )
            ]

    class _Mock2x2Tatr:
        def _objs(self, cw: float, ch: float):
            return [
                ("table", (0.0, 0.0, cw, ch), 0.93),
                ("table row", (0.0, 0.0, cw, ch * 0.5), 0.91),
                ("table row", (0.0, ch * 0.5, cw, ch), 0.91),
                ("table column", (0.0, 0.0, cw * 0.5, ch), 0.91),
                ("table column", (cw * 0.5, 0.0, cw, ch), 0.91),
                ("table column header", (0.0, 0.0, cw, ch * 0.5), 0.91),
            ]

        def detect_objects(self, crop: Image.Image):
            return self._objs(float(crop.width), float(crop.height))

        def extract(self, image: Image.Image, table_bbox_pt, tokens=()):
            return objects_to_table(self._objs(float(image.width), float(image.height)), image.size, table_bbox_pt, tokens)

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "one_row.pdf"
        b = TestPdfBuilder()
        p = b.new_page(width=300, height=200)
        p.insert_text((50, 60), "H1", fontsize=10)
        p.insert_text((170, 60), "H2", fontsize=10)
        p.insert_text((50, 100), "R1", fontsize=10)
        p.insert_text((170, 100), "V1", fontsize=10)
        b.save(pdf_path)
        with pdfplumber.open(str(pdf_path)) as doc:
            out = extract_tables_on_page(
                doc.pages[0],
                [(35.0, 40.0, 240.0, 125.0)],
                extractor=AutoTableExtractor(pdfplumber_extractor=_Mock1x2Plumber(), tatr_extractor=_Mock2x2Tatr()),
            )[0]
    assert out.source == "auto_tatr" and out.num_rows == 2 and out.num_cols == 2
    assert [c.text for c in out.cells] == ["H1", "H2", "R1", "V1"]


def test_pdfplumber_clipped_open_rowspan_slots_merge_and_backfill():
    from pdf.backends.table_pdfplumber import _convert_pdfplumber_table
    from pdf.models import Word

    class _BoxObj:
        def __init__(self, bbox, cells=None):
            self.bbox = bbox
            self.cells = cells

    class _FakePlumberTbl:
        bbox = (40.0, 50.0, 260.0, 170.0)
        columns = [_BoxObj((40.0, 50.0, 120.0, 170.0)), _BoxObj((120.0, 50.0, 260.0, 170.0))]
        rows = [
            _BoxObj((40.0, 50.0, 260.0, 80.0), cells=[(40.0, 50.0, 120.0, 80.0), (120.0, 50.0, 260.0, 80.0)]),
            _BoxObj((40.0, 80.0, 260.0, 110.0), cells=[None, (120.0, 80.0, 260.0, 110.0)]),
            _BoxObj((40.0, 110.0, 260.0, 140.0), cells=[None, (120.0, 110.0, 260.0, 140.0)]),
            _BoxObj((40.0, 140.0, 260.0, 170.0), cells=[None, (120.0, 140.0, 260.0, 170.0)]),
        ]

        def extract(self, **_kw):
            return [
                ["H1", "H2"],
                [None, "R1"],
                [None, "R2"],
                [None, "R3"],
            ]

    class _FakePage:
        bbox = (0.0, 0.0, 300.0, 250.0)
        # Rule between row 0 (H1) and row 1 (open slots 1..3), none between rows 1..3.
        lines = [{"x0": 40.0, "top": 80.0, "x1": 260.0, "bottom": 80.0}]
        rects = ()
        curves = ()

    words = [
        Word("H1", (50.0, 60.0, 70.0, 70.0), 10.0, False),
        Word("GroupX", (50.0, 115.0, 100.0, 125.0), 10.0, False),
    ]
    tbl = _convert_pdfplumber_table(_FakePlumberTbl(), (40.0, 50.0, 260.0, 170.0), page=_FakePage(), words=words)
    by_rc = {(c.row, c.col): c for c in tbl.cells}
    assert (1, 0) in by_rc
    assert by_rc[(1, 0)].rowspan == 3
    assert by_rc[(1, 0)].text == "GroupX"
    assert by_rc[(0, 0)].rowspan == 1
    assert by_rc[(0, 0)].text == "H1"


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
