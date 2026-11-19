import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pdfplumber  # noqa: E402

import pdf.ir as ir_mod  # noqa: E402
from pdf._pdf_io import page_bbox  # noqa: E402
from pdf._test_pdf import TestPdfBuilder  # noqa: E402
from pdf.assemble import assemble_page, filter_overlapping_figures, tag_running_headers_footers  # noqa: E402
from pdf.models import FigureBox, PageProfile, RasterPage, TextBlock  # noqa: E402
from pdf.text import extract_words  # noqa: E402


def _make_synthetic_builder():
    b = TestPdfBuilder()
    page = b.new_page(width=600, height=800)
    page.insert_text((50, 50), "Intro paragraph before diagram", fontsize=12)
    page.insert_text((80, 140), "NodeA", fontsize=10)
    page.insert_text((200, 140), "NodeB", fontsize=10)
    page.insert_text((50, 320), "ColA row1\nColA row2", fontsize=10)
    page.insert_text((220, 320), "ColB row1\nColB row2", fontsize=10)
    page.insert_text((50, 520), "Conclusion paragraph after table", fontsize=12)
    for i in range(8):
        page.draw_rect((60 + i * 30, 600, 80 + i * 30, 620))
    return b


def test_figure_and_table_children_ordering_and_no_body_leak():
    b = _make_synthetic_builder()
    with pdfplumber.open(io.BytesIO(b.to_bytes())) as doc:
        words = extract_words(doc.pages[0])
    profile = PageProfile(1, "native", len(words), 0.1, 0.0, False)
    fig = FigureBox((60.0, 110.0, 280.0, 220.0), "figure", "v1_group", 0.95)
    tbl = FigureBox((40.0, 290.0, 340.0, 400.0), "table", "layout_table", 0.95)
    items = assemble_page(profile, words, (0.0, 0.0, 600.0, 800.0), [fig, tbl])

    kinds = [it.kind for it in items]
    assert kinds == ["text", "figure", "text", "text", "table", "text", "text", "text"]

    fig_idx = kinds.index("figure")
    assert items[fig_idx + 1].parent_index == fig_idx and items[fig_idx + 1].level == 2
    assert items[fig_idx + 2].parent_index == fig_idx and items[fig_idx + 2].level == 2
    assert {items[fig_idx + 1].text, items[fig_idx + 2].text} == {"NodeA", "NodeB"}

    tbl_idx = kinds.index("table")
    assert items[tbl_idx + 1].parent_index == tbl_idx and items[tbl_idx + 1].level == 2
    assert items[tbl_idx + 2].parent_index == tbl_idx and items[tbl_idx + 2].level == 2
    table_text = items[tbl_idx + 1].text + "\n" + items[tbl_idx + 2].text
    assert "ColA row1" in table_text and "ColB row2" in table_text

    body_texts = [it.text for it in items if it.parent_index is None and it.kind == "text"]
    joined_body = "\n".join(body_texts)
    for inner in ("NodeA", "NodeB", "ColA", "ColB"):
        assert inner not in joined_body, (inner, body_texts)
    assert "Intro paragraph" in body_texts[0]
    assert "Conclusion paragraph" in body_texts[-1]


def test_overlapping_table_drop_and_innermost_box_assignment():
    fig = FigureBox((100.0, 100.0, 400.0, 300.0), "figure", "v1_group", 0.95)
    subsumed_tbl = FigureBox((120.0, 120.0, 300.0, 260.0), "table", "layout_table", 0.95)
    partial_tbl = FigureBox((350.0, 100.0, 550.0, 300.0), "table", "layout_table", 0.95)

    kept = filter_overlapping_figures([fig, subsumed_tbl, partial_tbl])
    assert kept == [fig, partial_tbl]

    b = TestPdfBuilder()
    page = b.new_page(width=600, height=400)
    page.insert_text((365, 180), "SharedWord", fontsize=10)
    with pdfplumber.open(io.BytesIO(b.to_bytes())) as doc:
        words = extract_words(doc.pages[0])

    # partial_tbl area (40,000) < fig area (60,000), so SharedWord belongs solely to partial_tbl.
    profile = PageProfile(1, "native", len(words), 0.05, 0.0, False)
    items = assemble_page(profile, words, (0.0, 0.0, 600.0, 400.0), [fig, subsumed_tbl, partial_tbl])
    children = [it for it in items if it.parent_index is not None]
    assert len(children) == 1
    assert items[children[0].parent_index].kind == "table"
    assert children[0].text == "SharedWord"


def test_extract_ir_end_to_end_with_mocked_figures():
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "assembled.pdf"
        _make_synthetic_builder().save(pdf_path)

        fig = FigureBox((60.0, 110.0, 280.0, 220.0), "figure", "v1_group", 0.95)
        tbl = FigureBox((40.0, 290.0, 340.0, 400.0), "table", "layout_table", 0.95)

        import pdf.figures as figures_mod

        orig_detect = figures_mod.detect_figures
        try:
            figures_mod.detect_figures = lambda _p, pages, **_kw: {p: [fig, tbl] for p in pages}
            ir = ir_mod.extract_ir(pdf_path, figures=True, ocr=False)
        finally:
            figures_mod.detect_figures = orig_detect

    assert json.loads(json.dumps(ir)) == ir
    assert ir["tables"] == []
    elems = ir["elements"]
    assert [e["reading_order"] for e in elems] == list(range(len(elems)))
    for e in elems:
        assert e["source_ref"] == {"file": "assembled.pdf", "page": 1, "bbox": e["bbox"]}
        assert isinstance(e["confidence"], float)
        assert "table_id" not in e
        if e["type"] == "text":
            assert "font_size" in e["metadata"] and "bold" in e["metadata"]


def _save_pdf(path, with_diagram_page):
    b = TestPdfBuilder()
    b.new_page(width=600, height=800).insert_text((50, 50), "Text only page", fontsize=12)
    if with_diagram_page:
        page = b.new_page(width=600, height=800)
        page.insert_text((50, 50), "Page with a diagram", fontsize=12)
        for i in range(8):
            page.draw_rect((60 + i * 30, 200, 80 + i * 30, 220))
    b.save(path)


def test_extract_ir_detects_figures_only_on_candidate_pages():
    import pdf.figures as figures_mod

    seen = []
    orig_detect = figures_mod.detect_figures
    figures_mod.detect_figures = lambda _p, pages, **_kw: seen.append(list(pages)) or {}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            mixed, text_only = Path(tmp) / "mixed.pdf", Path(tmp) / "text_only.pdf"
            _save_pdf(mixed, with_diagram_page=True)
            _save_pdf(text_only, with_diagram_page=False)
            ir = ir_mod.extract_ir(mixed, figures=True, ocr=False)
            assert seen == [[2]]
            ir_mod.extract_ir(text_only, figures=True, ocr=False)
            assert seen == [[2]]
    finally:
        figures_mod.detect_figures = orig_detect

    assert [p["figure_candidate"] for p in ir["metadata"]["pages"]] == [False, True]


def test_autosar_dev_pages_figure_assembly_integration():
    layered = ROOT / "datasets" / "AUTOSAR" / "AUTOSAR_CP_EXP_LayeredSoftwareArchitecture.pdf"
    if not layered.exists():
        print("SKIP AUTOSAR PDF not present")
        return

    from pdf.figures import detect_figures

    dev_figure_pages = [13, 17, 21, 25, 29, 33, 37, 41]
    with pdfplumber.open(str(layered)) as doc:
        figs_by_page = detect_figures(layered, dev_figure_pages)
        assert any(any(b.kind == "figure" for b in boxes) for boxes in figs_by_page.values())
        for pno in dev_figure_pages:
            page = doc.pages[pno - 1]
            words = extract_words(page)
            profile = PageProfile(pno, "native", len(words), 0.1, 0.0, False)
            items = assemble_page(profile, words, page_bbox(page), figs_by_page[pno])
            kept_figs = [it.bbox for it in items if it.kind == "figure"]
            body_items = [it for it in items if it.parent_index is None and it.kind == "text"]
            for w in words:
                cx = (w.bbox[0] + w.bbox[2]) * 0.5
                cy = (w.bbox[1] + w.bbox[3]) * 0.5
                inside_fig = any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in kept_figs)
                if inside_fig:
                    assert any(it.parent_index is not None and w.text in it.text for it in items), (pno, w.text)
            assert len(body_items) >= 1


def test_assemble_page_hybrid_suspect_ocr_fallback():
    b = TestPdfBuilder()
    page = b.new_page(width=600, height=800)
    page.insert_text((72, 72), "garbled layer text", fontsize=10)
    with pdfplumber.open(io.BytesIO(b.to_bytes())) as doc:
        words = extract_words(doc.pages[0])

    profile = PageProfile(1, "hybrid", len(words), 0.005, 0.9, True)
    rp = RasterPage(
        blocks=(TextBlock("OCR clean text", (70.0, 60.0, 220.0, 80.0), 0.0, False, "ocr", 0.92, label="text"),),
        figures=(),
        reading_order=(0,),
    )
    ocr_items = assemble_page(profile, words, (0.0, 0.0, 600.0, 800.0), raster_page=rp)
    assert len(ocr_items) == 1
    assert ocr_items[0].text == "OCR clean text" and ocr_items[0].source == "ocr"

    layer_items = assemble_page(profile, words, (0.0, 0.0, 600.0, 800.0), raster_page=None)
    assert len(layer_items) == 1
    assert "garbled" in layer_items[0].text and layer_items[0].source == "text_layer"


def test_running_header_footer_tagging_multipage():
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "multipage.pdf"
        b = TestPdfBuilder()
        for pno in range(1, 5):
            page = b.new_page(width=600, height=800)
            page.insert_text((50, 40), f"Acme Spec   Page {pno}", fontsize=10)
            page.insert_text((50, 200), "Repeated body note", fontsize=12)
            page.insert_text((50, 320), f"Unique body paragraph {pno}", fontsize=12)
            footer_text = str(pno) if pno <= 3 else "Final endnote"
            page.insert_text((280, 770), footer_text, fontsize=10)
        b.save(pdf_path)

        ir = ir_mod.extract_ir(pdf_path, figures=False, ocr=False)

    elems = ir["elements"]
    headers = [e for e in elems if "Acme Spec" in e["text"]]
    assert len(headers) == 4
    assert all(e["metadata"].get("role") == "page_header" for e in headers)

    footers = [e for e in elems if e["text"] in ("1", "2", "3")]
    assert len(footers) == 3
    assert all(e["metadata"].get("role") == "page_footer" for e in footers)

    endnote = [e for e in elems if e["text"] == "Final endnote"][0]
    assert endnote["metadata"].get("role") not in ("page_header", "page_footer")

    body_repeated = [e for e in elems if "Repeated body note" in e["text"]]
    assert len(body_repeated) == 4
    assert all(e["metadata"].get("role") not in ("page_header", "page_footer") for e in body_repeated)

    for pno in range(1, 5):
        ros = [e["reading_order"] for e in elems if e["page"] == pno]
        assert ros == list(range(len(ros)))

    with tempfile.TemporaryDirectory() as tmp:
        two_page = Path(tmp) / "twopage.pdf"
        b2 = TestPdfBuilder()
        for _ in range(2):
            p = b2.new_page(width=600, height=800)
            p.insert_text((50, 40), "Short Doc Header", fontsize=10)
            p.insert_text((50, 200), "Body content", fontsize=12)
        b2.save(two_page)
        ir2 = ir_mod.extract_ir(two_page, figures=False, ocr=False)
        assert all(e["metadata"].get("role") not in ("page_header", "page_footer") for e in ir2["elements"])

    direct_elems = [
        {"type": "text", "page": 1, "level": 1, "bbox": [10, 20, 100, 40], "text": "Hdr", "metadata": {"layout_label": "page_header"}},
        {"type": "text", "page": 1, "level": 2, "bbox": [10, 20, 100, 40], "text": "Nested", "metadata": {}},
    ]
    tag_running_headers_footers(direct_elems, {1: 800.0})
    assert direct_elems[0]["metadata"]["role"] == "page_header"
    assert "role" not in direct_elems[1]["metadata"]


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
