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
from pdf.assemble import (  # noqa: E402
    assemble_page,
    extract_plain_text,
    filter_overlapping_figures,
    merge_hybrid_region_ocr,
    tag_running_headers_footers,
)
from pdf.models import FigureBox, PageProfile, RasterPage, SemanticRegion, TextBlock, Word  # noqa: E402
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

        fig = FigureBox((60.0, 110.0, 280.0, 220.0), "figure", "v1_group", 0.83)
        tbl = FigureBox((40.0, 290.0, 340.0, 400.0), "table", "layout_table", 0.79)

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
    fig_elems = [e for e in elems if e["type"] == "figure"]
    tbl_elems = [e for e in elems if e["type"] == "table"]
    assert fig_elems[0]["confidence"] == 0.83 and tbl_elems[0]["confidence"] == 0.79
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
    assert 0.0 <= ocr_items[0].confidence <= 0.85

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


def test_filter_overlapping_figures_clips_grouper_against_clean_table():
    grouper_fig = FigureBox((100.0, 100.0, 400.0, 300.0), "figure", "v2_vector_vision", 0.90)
    clean_tbl = FigureBox((280.0, 100.0, 480.0, 300.0), "table", "layout_table", 0.85)
    kept = filter_overlapping_figures([grouper_fig, clean_tbl])
    assert len(kept) == 2
    assert kept[0].kind == "figure" and kept[0].bbox == (100.0, 100.0, 280.0, 300.0)
    assert kept[1] == clean_tbl

    # True full containment (>= 0.85) drops spurious table, and non-grouper figure drops >= 0.50 overlap.
    tiny_tbl = FigureBox((120.0, 120.0, 260.0, 260.0), "table", "layout_table", 0.80)
    assert filter_overlapping_figures([grouper_fig, tiny_tbl]) == [grouper_fig]
    raw_fig = FigureBox((100.0, 100.0, 400.0, 300.0), "figure", "layout_picture", 0.90)
    assert filter_overlapping_figures([raw_fig, clean_tbl]) == [raw_fig]


def test_assign_words_to_boxes_ioa_and_smallest_box_tie_break():
    words = [
        Word("MostlyIn", (85.0, 100.0, 125.0, 115.0), 10.0, False),   # 20/40 = 0.50 in box_big
        Word("MostlyOut", (80.0, 100.0, 120.0, 115.0), 10.0, False),  # 15/40 = 0.375 < 0.50
        Word("TieWord", (110.0, 110.0, 150.0, 125.0), 10.0, False),   # 1.0 in both -> smaller box_small wins
    ]
    box_big = FigureBox((105.0, 90.0, 250.0, 250.0), "figure", "v2_vector_vision", 0.9)       # area 23200
    box_small = FigureBox((110.0, 110.0, 400.0, 140.0), "table", "layout_table", 0.9)         # area 8700, overlap 0.48
    profile = PageProfile(1, "native", len(words), 0.1, 0.0, False)

    items = assemble_page(profile, words, (0.0, 0.0, 600.0, 800.0), [box_big, box_small])
    body = [it.text for it in items if it.parent_index is None and it.kind == "text"]
    assert body == ["MostlyOut"]

    fig_idx = next(i for i, it in enumerate(items) if it.kind == "figure")
    tbl_idx = next(i for i, it in enumerate(items) if it.kind == "table")
    fig_children = [it.text for it in items if it.parent_index == fig_idx]
    tbl_children = [it.text for it in items if it.parent_index == tbl_idx]
    assert fig_children == ["MostlyIn"]
    assert tbl_children == ["TieWord"]


def test_merge_hybrid_region_ocr_dedup():
    native_words = [
        Word("Hello", (50.0, 100.0, 95.0, 115.0), 10.0, False),
        Word("World", (100.0, 100.0, 150.0, 115.0), 10.0, False),
        Word("Body", (50.0, 200.0, 90.0, 215.0), 10.0, False),
    ]
    ocr_blocks = [
        TextBlock("Hello World", (48.0, 99.0, 152.0, 116.0), 10.0, False, "ocr_high_dpi", 0.92),
        TextBlock("Raster Label", (220.0, 300.0, 340.0, 320.0), 11.0, False, "ocr_high_dpi", 0.89),
        TextBlock("Totally Different", (48.0, 99.0, 152.0, 116.0), 10.0, False, "ocr_high_dpi", 0.88),
    ]
    kept_words, kept_ocr = merge_hybrid_region_ocr(native_words, ocr_blocks)
    assert kept_words == native_words
    assert [b.text for b in kept_ocr] == ["Raster Label", "Totally Different"]


def test_tag_running_headers_footers_even_odd_roman_and_toc():
    elems = []
    page_heights = {p: 800.0 for p in range(1, 7)}
    romans = ["i", "ii", "iii", "iv", "v", "vi"]
    for p in range(1, 7):
        hdr = "Even Spec Header" if p % 2 == 0 else "Odd Chapter Header"
        elems.append({"type": "text", "page": p, "level": 1, "bbox": [50, 30, 300, 45], "text": hdr, "metadata": {}})
        elems.append({"type": "text", "page": p, "level": 1, "bbox": [260, 760, 340, 775], "text": f"Preface {romans[p - 1]}", "metadata": {}})
    # Protected ToC title sitting inside the top header band across 4 pages must not be tagged.
    for p in range(1, 5):
        elems.append({"type": "text", "page": p, "level": 1, "bbox": [50, 50, 250, 65], "text": "1 Introduction", "metadata": {}})

    tag_running_headers_footers(elems, page_heights, toc_titles=["1 Introduction"])

    even_hdrs = [e for e in elems if e["text"] == "Even Spec Header"]
    odd_hdrs = [e for e in elems if e["text"] == "Odd Chapter Header"]
    footers = [e for e in elems if e["text"].startswith("Preface ")]
    protected = [e for e in elems if e["text"] == "1 Introduction"]

    assert len(even_hdrs) == 3 and all(e["metadata"].get("role") == "page_header" for e in even_hdrs)
    assert len(odd_hdrs) == 3 and all(e["metadata"].get("role") == "page_header" for e in odd_hdrs)
    assert len(footers) == 6 and all(e["metadata"].get("role") == "page_footer" for e in footers)
    assert all("role" not in e["metadata"] for e in protected)

    # On a 5-page doc, even-page header on pages 2, 4 (2/5 < 50% globally) still matches via even parity.
    five_elems = [
        {"type": "text", "page": 2, "level": 1, "bbox": [50, 30, 220, 45], "text": "Even Only Header", "metadata": {}},
        {"type": "text", "page": 4, "level": 1, "bbox": [50, 30, 220, 45], "text": "Even Only Header", "metadata": {}},
    ]
    tag_running_headers_footers(five_elems, {p: 800.0 for p in range(1, 6)})
    assert all(e["metadata"].get("role") == "page_header" for e in five_elems)

    # Below 4 pages, single-parity header on 1 of 3 pages must not be tagged.
    short_elems = [
        {"type": "text", "page": 1, "level": 1, "bbox": [50, 30, 200, 45], "text": "Odd Only", "metadata": {}},
        {"type": "text", "page": 2, "level": 1, "bbox": [50, 30, 200, 45], "text": "Even Only", "metadata": {}},
        {"type": "text", "page": 3, "level": 1, "bbox": [50, 30, 200, 45], "text": "Odd Only", "metadata": {}},
    ]
    tag_running_headers_footers(short_elems, {1: 800.0, 2: 800.0, 3: 800.0})
    assert "role" not in short_elems[1]["metadata"]


def test_extract_ir_semantic_regions_and_toc_titles_e2e():
    import pdf.figures as figures_mod

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "step3_sem_toc.pdf"
        b = TestPdfBuilder()
        for pno in range(1, 5):
            page = b.new_page(width=600, height=800)
            page.insert_text((50, 30), f"Acme Running Header {pno}", fontsize=10)
            page.insert_text((50, 58), "1 Introduction", fontsize=11)
            if pno == 1:
                page.insert_text((50, 120), "Unstyled Section Overview", fontsize=11)
                page.insert_text((60, 260), "Hardware block layout", fontsize=11)
                for i in range(8):
                    page.draw_rect((65 + i * 25, 160, 80 + i * 25, 220))
            else:
                page.insert_text((50, 200), f"Body paragraph content on page {pno}.", fontsize=11)
        b.save(pdf_path)

        def _fake_detect(_p, pages, **kw):
            sem_out = kw.get("semantic_regions_out")
            if sem_out is not None and 1 in pages:
                sem_out[1] = [
                    SemanticRegion((45.0, 108.0, 320.0, 136.0), "section_header", 0.91),
                    SemanticRegion((55.0, 248.0, 280.0, 274.0), "caption", 0.89),
                ]
            return {1: [FigureBox((60.0, 150.0, 280.0, 235.0), "figure", "v2_vector_vision", 0.93)]}

        orig_detect = figures_mod.detect_figures
        orig_toc = ir_mod.extract_toc
        figures_mod.detect_figures = _fake_detect
        ir_mod.extract_toc = lambda *_a, **_kw: [(1, "1 Introduction", 1)]
        try:
            ir = ir_mod.extract_ir(pdf_path, figures=True, ocr=False, structure=True)
        finally:
            figures_mod.detect_figures = orig_detect
            ir_mod.extract_toc = orig_toc

    elems = ir["elements"]
    running = [e for e in elems if "Acme Running Header" in e["text"]]
    assert len(running) == 4 and all(e["metadata"].get("role") == "page_header" for e in running)

    protected = [e for e in elems if e["text"] == "1 Introduction"]
    assert len(protected) == 4
    assert all(e["metadata"].get("role") != "page_header" for e in protected)
    p1_intro = next(e for e in protected if e["page"] == 1)
    assert p1_intro["metadata"].get("role") == "heading"
    assert p1_intro["metadata"].get("heading_level") == 1

    unstyled_sec = next(e for e in elems if e["text"] == "Unstyled Section Overview")
    assert unstyled_sec["metadata"].get("layout_label") == "section_header"
    assert unstyled_sec["metadata"].get("role") == "heading"

    cap_elem = next(e for e in elems if e["text"] == "Hardware block layout")
    fig_elem = next(e for e in elems if e["type"] == "figure")
    assert cap_elem["metadata"].get("layout_label") == "caption"
    assert cap_elem["metadata"].get("role") == "caption"
    assert cap_elem["metadata"].get("caption_of") == fig_elem["id"]
    assert fig_elem["metadata"].get("caption_id") == cap_elem["id"]


def test_extract_ir_hybrid_partial_raster_regions_and_table_continuations_e2e():
    from PIL import Image
    import pdf.figures as figures_mod
    from pdf.backends.base import RunStats

    buf = io.BytesIO()
    Image.new("RGB", (400, 240), color=(240, 240, 240)).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    def _draw_2x2_table(page, y0: float):
        for x in (40, 220, 480):
            page.draw_line((x, y0), (x, y0 + 180), width=0.5)
        for y in (y0, y0 + 60, y0 + 180):
            page.draw_line((40, y), (480, y), width=0.5)

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "hybrid_step3.pdf"
        b = TestPdfBuilder()

        p1 = b.new_page(width=600, height=800)
        p1.insert_text((50, 60), "Top native paragraph on hybrid page", fontsize=12)
        p1.insert_image((60, 180, 460, 420), stream=png_bytes)
        _draw_2x2_table(p1, 540.0)
        p1.insert_text((55, 575), "Param", fontsize=10)
        p1.insert_text((235, 575), "Value", fontsize=10)
        p1.insert_text((55, 650), "Alpha", fontsize=10)
        p1.insert_text((235, 650), "100", fontsize=10)

        p2 = b.new_page(width=600, height=800)
        _draw_2x2_table(p2, 50.0)
        p2.insert_text((55, 85), "Param", fontsize=10)
        p2.insert_text((235, 85), "Value", fontsize=10)
        p2.insert_text((55, 160), "Beta", fontsize=10)
        p2.insert_text((235, 160), "200", fontsize=10)
        b.save(pdf_path)

        class _FakeHybridReader:
            name = "fake_hybrid_reader"
            last_stats = None

            def __init__(self) -> None:
                self.seen_regions = None

            def read(self, _path, _pages, **_kw):
                raise AssertionError("read_raster_pages should not be called for non-suspect hybrid page")

            def read_raster_regions(self, _path, regions_by_page, **_kw):
                self.seen_regions = dict(regions_by_page)
                self.last_stats = RunStats(
                    pages=len(regions_by_page),
                    in_process=True,
                    warnings=({"page": 1, "code": "region_ocr_note", "severity": "info", "reason": "test", "fallback": None},),
                )
                return {
                    1: [
                        TextBlock(
                            "Top native paragraph on hybrid page",
                            (48.0, 48.0, 265.0, 64.0),
                            12.0,
                            False,
                            "ocr_high_dpi",
                            0.90,
                            label="text",
                        ),
                        TextBlock(
                            "Embedded raster callout text",
                            (80.0, 260.0, 320.0, 285.0),
                            11.0,
                            False,
                            "ocr_high_dpi",
                            0.94,
                            label="text",
                        ),
                    ]
                }

        fake_reader = _FakeHybridReader()
        orig_detect = figures_mod.detect_figures
        figures_mod.detect_figures = lambda _p, _pages, **_kw: {
            1: [FigureBox((38.0, 538.0, 482.0, 722.0), "table", "layout_table", 0.95)],
            2: [FigureBox((38.0, 48.0, 482.0, 232.0), "table", "layout_table", 0.95)],
        }
        try:
            ir = ir_mod.extract_ir(
                pdf_path,
                figures=True,
                ocr=True,
                tables=True,
                table_backend="pdfplumber",
                reader=fake_reader,
                structure=True,
            )
        finally:
            figures_mod.detect_figures = orig_detect

    assert fake_reader.seen_regions == {1: ((60.0, 180.0, 460.0, 420.0),)}
    assert any(w.get("code") == "region_ocr_note" for w in ir["metadata"]["warnings"])

    p1_top_elems = [e for e in ir["elements"] if e["page"] == 1 and e["level"] == 1]
    assert [e["type"] for e in p1_top_elems] == ["text", "text", "table"]
    assert p1_top_elems[0]["text"] == "Top native paragraph on hybrid page"
    assert p1_top_elems[0]["source"] == "text_layer"
    assert p1_top_elems[1]["text"] == "Embedded raster callout text"
    assert p1_top_elems[1]["source"] == "ocr_high_dpi"
    assert p1_top_elems[1]["confidence"] == 0.94

    assert len(ir["tables"]) == 2
    assert ir["tables"][0]["id"] == "tbl_0001" and ir["tables"][0]["continuation_of"] is None
    assert ir["tables"][1]["id"] == "tbl_0002" and ir["tables"][1]["continuation_of"] == "tbl_0001"


def test_tag_running_headers_footers_preserves_appendix_letters():
    page_heights = {p: 800.0 for p in range(1, 7)}
    elems = [
        {"type": "text", "page": 1, "level": 1, "bbox": [50, 30, 200, 45], "text": "Appendix C", "metadata": {}},
        {"type": "text", "page": 2, "level": 1, "bbox": [50, 30, 200, 45], "text": "Appendix C", "metadata": {}},
        {"type": "text", "page": 3, "level": 1, "bbox": [50, 30, 200, 45], "text": "Appendix D", "metadata": {}},
        {"type": "text", "page": 4, "level": 1, "bbox": [50, 30, 200, 45], "text": "Appendix D", "metadata": {}},
        {"type": "text", "page": 5, "level": 1, "bbox": [50, 30, 200, 45], "text": "Section I", "metadata": {}},
        {"type": "text", "page": 6, "level": 1, "bbox": [50, 30, 200, 45], "text": "Table V", "metadata": {}},
    ]
    tag_running_headers_footers(elems, page_heights)
    assert all("role" not in e["metadata"] for e in elems), elems


def test_ir_low_confidence_warning_emitted():
    from PIL import Image

    class _FakeReader:
        last_stats = None

        def read(self, pdf_path, pages, **kwargs):
            del pdf_path, kwargs
            return {
                p: RasterPage(
                    blocks=(TextBlock("Blurry text", (50.0, 60.0, 220.0, 80.0), 0.0, False, "ocr", 0.62, label="text"),),
                    figures=(),
                    reading_order=(0,),
                )
                for p in pages
            }

    buf = io.BytesIO()
    Image.new("RGB", (600, 800), (255, 255, 255)).save(buf, format="PNG")
    b = TestPdfBuilder()
    b.new_page(width=600, height=800).insert_image((0, 0, 600, 800), stream=buf.getvalue())
    b.new_page(width=600, height=800).insert_text((72, 120), "Clean native body text " * 4, fontsize=11)
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "low_conf.pdf"
        b.save(pdf_path)
        ir = ir_mod.extract_ir(pdf_path, figures=False, ocr=True, reader=_FakeReader(), structure=False)
    low_warnings = [w for w in ir["metadata"]["warnings"] if w.get("code") == "low_confidence"]
    assert len(low_warnings) == 1 and low_warnings[0]["page"] == 1


def test_extract_plain_text_tables_reading_order_and_furniture():
    ir = {
        "metadata": {"pages": [{"number": 1}, {"number": 2}, {"number": 3}]},
        "elements": [
            {"id": "e1", "type": "text", "page": 1, "level": 1, "reading_order": 0, "text": "Header Line", "metadata": {"role": "page_header"}},
            {"id": "e2", "type": "text", "page": 1, "level": 1, "reading_order": 1, "text": "Intro paragraph", "metadata": {}},
            {"id": "e3", "type": "table", "page": 1, "level": 1, "reading_order": 2, "text": "[Table tbl_0001: 2x2]", "metadata": {"table_id": "tbl_0001"}},
            {"id": "e3_c1", "type": "text", "page": 1, "level": 2, "parent_id": "e3", "reading_order": 3, "text": "Duplicate child", "metadata": {}},
            {"id": "e4", "type": "figure", "page": 1, "level": 1, "reading_order": 4, "text": "", "metadata": {}},
            {"id": "e4_c1", "type": "text", "page": 1, "level": 2, "parent_id": "e4", "reading_order": 5, "text": "Diagram node", "metadata": {}},
            {"id": "e5", "type": "text", "page": 1, "level": 1, "reading_order": 6, "text": "Closing paragraph", "metadata": {}},
            {"id": "e6", "type": "text", "page": 1, "level": 1, "reading_order": 7, "text": "Page 1", "metadata": {"layout_label": "page_footer"}},
            {"id": "e7", "type": "table", "page": 2, "level": 1, "reading_order": 0, "text": "", "metadata": {}},
            {"id": "e7_c2", "type": "text", "page": 2, "level": 2, "parent_id": "e7", "reading_order": 2, "text": "Fallback row 2", "metadata": {}},
            {"id": "e7_c1", "type": "text", "page": 2, "level": 2, "parent_id": "e7", "reading_order": 1, "text": "Fallback row 1", "metadata": {}},
        ],
        "tables": [
            {
                "id": "tbl_0001",
                "page": 1,
                "num_rows": 2,
                "num_cols": 2,
                "cells": [
                    {"row": 1, "col": 1, "text": "100"},
                    {"row": 0, "col": 1, "text": "Value"},
                    {"row": 0, "col": 0, "text": "Key"},
                    {"row": 1, "col": 0, "text": "Alpha"},
                ],
            }
        ],
    }
    body = extract_plain_text(ir, include_furniture=False)
    assert body == {
        1: "Intro paragraph\nKey Value\nAlpha 100\nClosing paragraph",
        2: "Fallback row 1\nFallback row 2",
        3: "",
    }
    raw = extract_plain_text(ir, include_furniture=True)
    assert raw[1] == "Header Line\nIntro paragraph\nKey Value\nAlpha 100\nClosing paragraph\nPage 1"
    assert ir["elements"][2]["text"] == "[Table tbl_0001: 2x2]"


def test_tag_running_headers_footers_unicode_toc_protection():
    page_heights = {p: 800.0 for p in range(1, 5)}
    elems = []
    for p in range(1, 5):
        elems.append({"type": "text", "page": p, "level": 1, "bbox": [50, 30, 260, 45], "text": "第１章　総則", "metadata": {}})
        elems.append({"type": "text", "page": p, "level": 1, "bbox": [280, 760, 310, 775], "text": str(p), "metadata": {}})

    tag_running_headers_footers(elems, page_heights, toc_titles=["第1章 総則", "1", "（２）"])
    headers = [e for e in elems if e["text"] == "第１章　総則"]
    footers = [e for e in elems if e["text"].isdigit()]
    assert all("role" not in e["metadata"] for e in headers)
    assert all(e["metadata"].get("role") == "page_footer" for e in footers)


def test_hardened_running_header_narrow_margin_and_protected_layout_label():
    # Narrow-margin body text at cy/page_h ~ 0.066 with fs == body_font must NOT be tagged page_header.
    elems = []
    page_heights = {p: 800.0 for p in range(1, 5)}
    for p in range(1, 5):
        elems.append(
            {
                "type": "text",
                "page": p,
                "level": 1,
                "bbox": [50.0, 48.0, 400.0, 60.0],
                "text": "General Terms and Conditions",
                "metadata": {"font_size": 10.0},
            }
        )
        elems.append(
            {
                "type": "text",
                "page": p,
                "level": 1,
                "bbox": [50.0, 120.0, 500.0, 220.0],
                "text": "Main body paragraph text establishing the 10pt body font size across pages. " * 3,
                "metadata": {"font_size": 10.0},
            }
        )
    tag_running_headers_footers(elems, page_heights)
    top_elems = [e for e in elems if e["text"] == "General Terms and Conditions"]
    assert all("role" not in e["metadata"] for e in top_elems), top_elems

    # layout_label=page_header at margin band with fs <= body_font * 1.02 overrides protected_titles.
    elems_lbl = [
        {
            "type": "text",
            "page": 1,
            "level": 1,
            "bbox": [50.0, 20.0, 250.0, 32.0],
            "text": "1 Introduction",
            "metadata": {"font_size": 9.5, "layout_label": "page_header"},
        },
        {
            "type": "text",
            "page": 1,
            "level": 1,
            "bbox": [50.0, 120.0, 500.0, 200.0],
            "text": "Body text content of sufficient length. " * 3,
            "metadata": {"font_size": 10.0},
        },
    ]
    tag_running_headers_footers(elems_lbl, {1: 800.0}, toc_titles=["1 Introduction"])
    assert elems_lbl[0]["metadata"].get("role") == "page_header"


def test_trailing_badge_stripping_and_negative_unit_suffix():
    profile = PageProfile(1, "native", 8, 0.0, 0.0, False)
    # Two consecutive small trailing badges inside vector boxes on one line.
    badge_words = [
        Word("1.", (50.0, 100.0, 62.0, 112.0), 12.0, True),
        Word("Coverage", (66.0, 100.0, 125.0, 112.0), 12.0, True),
        Word("Overview", (129.0, 100.0, 185.0, 112.0), 12.0, True),
        Word("Contract", (200.0, 102.0, 240.0, 110.0), 7.5, False),
        Word("Notice", (248.0, 102.0, 285.0, 110.0), 7.5, False),
    ]
    boxes = [(197.0, 100.0, 243.0, 112.0), (245.0, 100.0, 288.0, 112.0)]
    items = assemble_page(profile, badge_words, (0.0, 0.0, 600.0, 800.0), vector_boxes=boxes)
    assert len(items) == 1
    assert items[0].text == "1. Coverage Overview Contract Notice"
    assert items[0].heading_clean_text == "1. Coverage Overview"

    # Negative: unit suffix "(kg)" without a vector box and with normal word gap must NOT be stripped.
    unit_words = [
        Word("Section", (50.0, 100.0, 96.0, 112.0), 12.0, True),
        Word("3", (100.0, 100.0, 108.0, 112.0), 12.0, True),
        Word("(kg)", (112.0, 102.0, 132.0, 110.0), 8.0, False),
    ]
    unit_items = assemble_page(profile, unit_words, (0.0, 0.0, 600.0, 800.0))
    assert len(unit_items) == 1
    assert unit_items[0].heading_clean_text is None


def test_text_layer_margin_font_lock_and_short_doc_page_number_footer():
    import pdf.figures as figures_mod

    # 1. Direct tag_running_headers_footers lock clears both role and layout_label on out-of-margin or oversized text_layer.
    direct = [
        {
            "id": "hdr_ok",
            "type": "text",
            "page": 1,
            "level": 1,
            "source": "text_layer",
            "bbox": [50.0, 20.0, 240.0, 34.0],
            "text": "Valid Top Margin Header",
            "metadata": {"font_size": 9.5, "layout_label": "page_header"},
        },
        {
            "id": "mid_fp",
            "type": "text",
            "page": 1,
            "level": 1,
            "source": "text_layer",
            "bbox": [50.0, 350.0, 400.0, 370.0],
            "text": "Mid-page body sentence mislabeled as header.",
            "metadata": {"font_size": 10.0, "layout_label": "page_header", "role": "page_header"},
        },
        {
            "id": "big_fp",
            "type": "text",
            "page": 1,
            "level": 1,
            "source": "text_layer",
            "bbox": [50.0, 25.0, 350.0, 45.0],
            "text": "Large Document Title",
            "metadata": {"font_size": 18.0, "layout_label": "page_header"},
        },
        {
            "id": "body",
            "type": "text",
            "page": 1,
            "level": 1,
            "source": "text_layer",
            "bbox": [50.0, 150.0, 500.0, 280.0],
            "text": "Main body text establishing 10pt font size cleanly. " * 4,
            "metadata": {"font_size": 10.0},
        },
    ]
    tag_running_headers_footers(direct, {1: 800.0})
    assert direct[0]["metadata"].get("role") == "page_header"
    assert direct[0]["metadata"].get("layout_label") == "page_header"
    assert "role" not in direct[1]["metadata"] and "layout_label" not in direct[1]["metadata"]
    assert "role" not in direct[2]["metadata"] and "layout_label" not in direct[2]["metadata"]

    # 2. End-to-end 2-page PDF: mid-page page_header SemanticRegion is blocked, and isolated bottom page numbers
    #    on < 3 page docs are tagged page_footer without touching footnotes.
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "short_lock.pdf"
        b = TestPdfBuilder()
        p1 = b.new_page(width=600, height=800)
        p1.insert_text((50, 24), "Small Header", fontsize=9)
        p1.insert_text((50, 350), "Important middle body clause that must survive.", fontsize=11)
        for i in range(8):
            p1.draw_rect((60 + i * 30, 500, 75 + i * 30, 520))
        p1.insert_text((280, 760), "1", fontsize=10)

        p2 = b.new_page(width=600, height=800)
        p2.insert_text((50, 120), "Second page body paragraph of sufficient length. " * 2, fontsize=11)
        p2.insert_text((50, 730), "1 Footnote explanation text at bottom.", fontsize=9)
        p2.insert_text((270, 760), "Page 2 of 2", fontsize=10)
        b.save(pdf_path)

        def _fake_detect(_p, _pages, **kw):
            sem_out = kw.get("semantic_regions_out")
            if sem_out is not None:
                sem_out[1] = [
                    SemanticRegion((40.0, 15.0, 200.0, 40.0), "page_header", 0.94),
                    SemanticRegion((40.0, 340.0, 480.0, 380.0), "page_header", 0.95),
                ]
            return {}

        orig_detect = figures_mod.detect_figures
        figures_mod.detect_figures = _fake_detect
        try:
            ir = ir_mod.extract_ir(pdf_path, figures=True, ocr=False, structure=True)
        finally:
            figures_mod.detect_figures = orig_detect

    elems = ir["elements"]
    mid_elem = next(e for e in elems if "Important middle body clause" in e["text"])
    assert mid_elem["metadata"].get("layout_label") is None
    assert mid_elem["metadata"].get("role") is None
    plain = extract_plain_text(ir, include_furniture=False)
    assert "Important middle body clause" in plain[1]

    hdr_elem = next(e for e in elems if e["text"] == "Small Header")
    assert hdr_elem["metadata"].get("layout_label") == "page_header"
    assert hdr_elem["metadata"].get("role") == "page_header"

    ftr1 = next(e for e in elems if e["page"] == 1 and e["text"] == "1")
    ftr2 = next(e for e in elems if e["page"] == 2 and e["text"] == "Page 2 of 2")
    fn2 = next(e for e in elems if e["page"] == 2 and "Footnote explanation" in e["text"])
    assert ftr1["metadata"].get("role") == "page_footer"
    assert ftr2["metadata"].get("role") == "page_footer"
    assert fn2["metadata"].get("role") is None

    # Negative: bottom number sharing the same row with another element must NOT be tagged page_footer.
    non_iso = [
        {"type": "text", "page": 1, "level": 1, "bbox": [50.0, 755.0, 200.0, 770.0], "text": "Footnote text", "metadata": {"font_size": 10.0}},
        {"type": "text", "page": 1, "level": 1, "bbox": [520.0, 755.0, 540.0, 770.0], "text": "1", "metadata": {"font_size": 10.0}},
    ]
    tag_running_headers_footers(non_iso, {1: 800.0})
    assert non_iso[1]["metadata"].get("role") is None


def test_extract_ir_semantic_layout_modes_and_validation():
    import pdf.figures as figures_mod
    from pdf.backends import PageLayout, RunStats

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "modes.pdf"
        b = TestPdfBuilder()
        p1 = b.new_page(width=600, height=800)
        p1.insert_text((50, 50), "Plain text page", fontsize=12)
        p1.insert_text((50, 120), "x = 1", fontsize=10)
        p1.insert_text((50, 150), "y = 2", fontsize=10)

        p2 = b.new_page(width=600, height=800)
        p2.insert_text((50, 50), "Diagram candidate page", fontsize=12)
        for i in range(8):
            p2.draw_rect((60 + i * 30, 200, 80 + i * 30, 220))
        b.save(pdf_path)

        try:
            ir_mod.extract_ir(pdf_path, semantic_layout="invalid")  # type: ignore[arg-type]
            raise AssertionError("expected ValueError for invalid semantic_layout")
        except ValueError:
            pass

        class _ModeBackend:
            name = "mode_mock"
            revision = "m1"
            cache_dpi = 150
            last_stats: RunStats | None = None

            def __init__(self) -> None:
                self.analyzed_pages: list[int] = []

            def analyze(self, _pdf_path, pages):
                self.analyzed_pages.extend(pages)
                self.last_stats = RunStats(infer_sec=0.01, pages=len(pages), in_process=True)
                out = {}
                for p in pages:
                    if p == 1:
                        out[p] = PageLayout(
                            items=[
                                {"order": 0, "label": "code", "bbox": [40.0, 110.0, 220.0, 180.0], "text": "", "score": 0.92},
                                {"order": 1, "label": "table", "bbox": [40.0, 200.0, 500.0, 320.0], "text": "", "score": 0.85},
                            ],
                            frame=(600.0, 800.0),
                        )
                    else:
                        out[p] = PageLayout(
                            items=[
                                {"order": 0, "label": "picture", "bbox": [50.0, 180.0, 350.0, 320.0], "text": "", "score": 0.90},
                            ],
                            frame=(600.0, 800.0),
                        )
                return out

        class _NoNonCandGrouper:
            name = "no_non_cand"
            source = "v2_vector_vision"

            def __init__(self) -> None:
                self.calls = 0

            def group(self, bgr, layout, min_area=0.015):
                del bgr, layout, min_area
                self.calls += 1
                from pdf.models import GroupedFigure

                return [GroupedFigure(bbox=(50.0, 180.0, 350.0, 320.0), score=0.90, source="v2_vector_vision")]

        orig_root = figures_mod._REPO_ROOT
        try:
            # auto: only page 2 (figure_candidate=True) runs layout.
            figures_mod._REPO_ROOT = Path(tmp) / "auto_root"
            b_auto = _ModeBackend()
            g_auto = _NoNonCandGrouper()
            ir_auto = ir_mod.extract_ir(
                pdf_path, figures=True, ocr=False, layout=b_auto, grouper=g_auto, semantic_layout="auto"
            )
            assert b_auto.analyzed_pages == [2]
            assert g_auto.calls == 1
            p1_auto = [e for e in ir_auto["elements"] if e["page"] == 1]
            assert all(e["metadata"].get("layout_label") is None for e in p1_auto)

            # always: both page 1 and page 2 run layout, grouper only runs on page 2,
            # non-table_candidate page 1 does NOT emit layout_table, and page 1 code lines soft-merge.
            figures_mod._REPO_ROOT = Path(tmp) / "always_root"
            b_always = _ModeBackend()
            g_always = _NoNonCandGrouper()
            ir_always = ir_mod.extract_ir(
                pdf_path, figures=True, ocr=False, layout=b_always, grouper=g_always, semantic_layout="always"
            )
            assert sorted(b_always.analyzed_pages) == [1, 2]
            assert g_always.calls == 1
            p1_always = [e for e in ir_always["elements"] if e["page"] == 1]
            assert not any(e["type"] == "table" for e in p1_always)
            code_elems = [e for e in p1_always if e["metadata"].get("layout_label") == "code"]
            assert len(code_elems) == 1
            assert code_elems[0]["text"] == "x = 1\ny = 2"

            # off: semantic_regions are not collected or projected even on candidate pages.
            figures_mod._REPO_ROOT = Path(tmp) / "off_root"
            b_off = _ModeBackend()
            g_off = _NoNonCandGrouper()
            seen_sem_outs = []
            orig_detect = figures_mod.detect_figures

            def _spy_detect(p, pages, **kw):
                seen_sem_outs.append(kw.get("semantic_regions_out"))
                return orig_detect(p, pages, **kw)

            figures_mod.detect_figures = _spy_detect
            try:
                ir_off = ir_mod.extract_ir(
                    pdf_path,
                    figures=True,
                    ocr=False,
                    formulas=True,
                    layout=b_off,
                    grouper=g_off,
                    semantic_layout="off",
                )
            finally:
                figures_mod.detect_figures = orig_detect

            assert seen_sem_outs == [None]
            assert b_off.analyzed_pages == [2]
            assert ir_off["formulas"] == []
            assert all(e["metadata"].get("layout_label") is None for e in ir_off["elements"])
        finally:
            figures_mod._REPO_ROOT = orig_root


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
