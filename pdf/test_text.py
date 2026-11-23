import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pdfplumber  # noqa: E402

import pdf  # noqa: E402
from pdf import extract_ir  # noqa: E402
from pdf._pdf_io import extract_words_raw  # noqa: E402
from pdf._test_pdf import TestPdfBuilder  # noqa: E402
from pdf.models import Word  # noqa: E402
from pdf.text import extract_raw_line_segments, extract_words, order_blocks, xy_cut_order  # noqa: E402


def _make_two_column_pdf(path: Path) -> None:
    b = TestPdfBuilder()
    p1 = b.new_page(width=595, height=842)
    p1.insert_text((60, 60), "TITLE BANNER ACROSS BOTH COLUMNS", fontsize=16)
    p1.insert_text((60, 110), "L1 left column first paragraph\nL2 left column second line", fontsize=11)
    p1.insert_text((60, 170), "L3 left column bottom paragraph", fontsize=11)
    p1.insert_text((340, 110), "R1 right column first paragraph\nR2 right column second line", fontsize=11)
    p1.insert_text((340, 170), "R3 right column bottom paragraph", fontsize=11)

    p2 = b.new_page(width=595, height=842)
    p2.insert_text((60, 80), "Page two single column content", fontsize=12)
    b.save(path)


def test_no_docling_or_torch_on_native_text_ir():
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "two_col.pdf"
        _make_two_column_pdf(pdf_path)
        ir = extract_ir(pdf_path, figures=False, ocr=False)
        assert "docling" not in sys.modules
        assert "torch" not in sys.modules
        assert set(ir) == {"doc_id", "source_file", "doc_type", "metadata", "elements", "tables", "diagrams"}
        assert ir["doc_type"] == "pdf"
        assert ir["tables"] == [] and ir["diagrams"] == []
        assert json.loads(json.dumps(ir)) == ir


def test_two_column_column_major_reading_order():
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "two_col.pdf"
        _make_two_column_pdf(pdf_path)
        ir = extract_ir(pdf_path, figures=False, ocr=False)
        p1_texts = [e["text"] for e in ir["elements"] if e["page"] == 1]
        joined = "\n".join(p1_texts)
        positions = [joined.index(marker) for marker in ("TITLE", "L1", "L3", "R1", "R3")]
        assert positions == sorted(positions), (positions, p1_texts)

        p1_ros = [e["reading_order"] for e in ir["elements"] if e["page"] == 1]
        p2_ros = [e["reading_order"] for e in ir["elements"] if e["page"] == 2]
        assert p1_ros == list(range(len(p1_ros)))
        assert p2_ros == list(range(len(p2_ros)))
        assert all(e["metadata"]["font_size"] > 0 for e in ir["elements"])


def test_word_parity_and_speed_on_layered_pdf():
    layered = ROOT / "datasets" / "AUTOSAR" / "AUTOSAR_CP_EXP_LayeredSoftwareArchitecture.pdf"
    if not layered.exists():
        print("SKIP Layered PDF not present")
        return

    t0 = time.perf_counter()
    ir = extract_ir(layered, figures=False, ocr=False)
    elapsed = time.perf_counter() - t0
    assert "docling" not in sys.modules
    assert "torch" not in sys.modules
    assert "pymupdf" not in sys.modules

    with pdfplumber.open(str(layered)) as doc:
        ms_per_page = (elapsed / len(doc.pages)) * 1000.0
        print(f"  Layered {len(doc.pages)} pages: {elapsed:.3f}s total ({ms_per_page:.2f} ms/page)")
        assert ms_per_page < 200.0, f"too slow: {ms_per_page:.2f} ms/page"

        ir_words_by_page: dict[int, Counter[str]] = {}
        for elem in ir["elements"]:
            assert elem["type"] == "text"
            ir_words_by_page.setdefault(elem["page"], Counter()).update(elem["text"].split())

        for page_idx, page in enumerate(doc.pages, start=1):
            expected = Counter(w.text for w in extract_words(page))
            actual = ir_words_by_page.get(page_idx, Counter())
            assert actual == expected, (page_idx, expected - actual, actual - expected)
            raw_chars = Counter(ch for w in extract_words_raw(page, with_style=True) for ch in w["text"])
            actual_chars = Counter(ch for w_text, n in actual.items() for ch in w_text * n)
            assert actual_chars == raw_chars, (page_idx, raw_chars - actual_chars, actual_chars - raw_chars)
        p9_texts = [e["text"] for e in ir["elements"] if e["page"] == 9]
        assert "page id: toc01" in p9_texts, p9_texts


def test_two_column_mid_page_spanning_banner():
    items = [
        ("UL1", (45.0, 80.0, 260.0, 120.0)),
        ("UL2", (45.0, 130.0, 260.0, 180.0)),
        ("UR1", (320.0, 80.0, 540.0, 120.0)),
        ("UR2", (320.0, 130.0, 540.0, 180.0)),
        ("MID_BANNER", (45.0, 195.0, 540.0, 230.0)),
        ("BRIDGE_NOTE", (275.0, 170.0, 305.0, 260.0)),
        ("LL1", (45.0, 245.0, 260.0, 290.0)),
        ("LL2", (45.0, 300.0, 260.0, 350.0)),
        ("LR1", (320.0, 245.0, 540.0, 290.0)),
        ("LR2", (320.0, 300.0, 540.0, 350.0)),
    ]
    ordered = [name for name, _ in xy_cut_order(items, lambda it: it[1], page_rect=(0.0, 0.0, 595.0, 842.0))]
    expected_upper = ["UL1", "UL2", "UR1", "UR2"]
    expected_lower = ["LL1", "LL2", "LR1", "LR2"]
    mid_idx = ordered.index("MID_BANNER")
    assert [x for x in ordered[:mid_idx] if x != "BRIDGE_NOTE"] == expected_upper, ordered
    assert [x for x in ordered[mid_idx + 1 :] if x != "BRIDGE_NOTE"] == expected_lower, ordered

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "mid_banner.pdf"
        b = TestPdfBuilder()
        p = b.new_page(width=595, height=842)
        p.insert_text((50, 80), "UL1 upper left top", fontsize=11)
        p.insert_text((50, 140), "UL2 upper left low", fontsize=11)
        p.insert_text((320, 80), "UR1 upper right top", fontsize=11)
        p.insert_text((320, 140), "UR2 upper right low", fontsize=11)
        p.insert_text((50, 220), "MID_BANNER FULL WIDTH SPANNING SECTION ACROSS PAGE", fontsize=13)
        p.insert_text((50, 280), "LL1 lower left top", fontsize=11)
        p.insert_text((50, 340), "LL2 lower left low", fontsize=11)
        p.insert_text((320, 280), "LR1 lower right top", fontsize=11)
        p.insert_text((320, 340), "LR2 lower right low", fontsize=11)
        b.save(pdf_path)
        ir = extract_ir(pdf_path, figures=False, ocr=False)
        joined = "\n".join(e["text"] for e in ir["elements"])
        markers = ("UL1", "UL2", "UR1", "UR2", "MID_BANNER", "LL1", "LL2", "LR1", "LR2")
        positions = [joined.index(m) for m in markers]
        assert positions == sorted(positions), (positions, joined)


def test_three_column_layout():
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "three_col.pdf"
        b = TestPdfBuilder()
        p = b.new_page(width=595, height=842)
        p.insert_text((40, 80), "C1_TOP col one top", fontsize=11)
        p.insert_text((40, 160), "C1_BOT col one low", fontsize=11)
        p.insert_text((220, 80), "C2_TOP col two top", fontsize=11)
        p.insert_text((220, 160), "C2_BOT col two low", fontsize=11)
        p.insert_text((400, 80), "C3_TOP col three top", fontsize=11)
        p.insert_text((400, 160), "C3_BOT col three low", fontsize=11)
        b.save(pdf_path)
        ir = extract_ir(pdf_path, figures=False, ocr=False)
        joined = "\n".join(e["text"] for e in ir["elements"])
        markers = ("C1_TOP", "C1_BOT", "C2_TOP", "C2_BOT", "C3_TOP", "C3_BOT")
        positions = [joined.index(m) for m in markers]
        assert positions == sorted(positions), (positions, joined)


def test_vertical_90_270_text_no_horizontal_interleaving():
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "rotated.pdf"
        b = TestPdfBuilder()
        p = b.new_page(width=595, height=842)
        p.insert_text((100, 100), "HorizontalBodyLine", fontsize=11)
        p._ops.append(b"BT /F1 11.00 Tf 0 1 -1 0 60.00 500.00 Tm (AlphaNinety) Tj ET")
        p._ops.append(b"BT /F1 11.00 Tf 0 -1 1 0 490.00 650.00 Tm (ColLeftVert) Tj ET")
        p._ops.append(b"BT /F1 11.00 Tf 0 -1 1 0 505.00 650.00 Tm (ColRightVert) Tj ET")
        b.save(pdf_path)

        expected = Counter(["HorizontalBodyLine", "AlphaNinety", "ColLeftVert", "ColRightVert"])

        ir = extract_ir(pdf_path, figures=False, ocr=False)
        actual = Counter()
        for e in ir["elements"]:
            actual.update(e["text"].split())
        assert actual == expected, (expected - actual, actual - expected)
        horiz_blocks = [e["text"] for e in ir["elements"] if "HorizontalBodyLine" in e["text"]]
        assert len(horiz_blocks) == 1 and horiz_blocks[0] == "HorizontalBodyLine"

    synthetic_words = [
        Word("HorizA", (100.0, 200.0, 160.0, 212.0), 11.0, False, upright=True),
        Word("HorizB", (165.0, 200.0, 225.0, 212.0), 11.0, False, upright=True),
        Word("A", (420.0, 195.0, 428.0, 203.0), 8.0, False, upright=False),
        Word("B", (420.0, 204.0, 428.0, 212.0), 8.0, False, upright=False),
        Word("C", (420.0, 213.0, 428.0, 221.0), 8.0, False, upright=False),
        Word("X", (434.0, 195.0, 442.0, 203.0), 8.0, False, upright=False),
        Word("Y", (434.0, 204.0, 442.0, 212.0), 8.0, False, upright=False),
        Word("Z", (434.0, 213.0, 442.0, 221.0), 8.0, False, upright=False),
        Word("TallBoxLabel", (460.0, 190.0, 470.0, 270.0), 10.0, False, upright=True),
    ]
    blocks = order_blocks(synthetic_words, (0.0, 0.0, 595.0, 842.0))
    texts = [b.text for b in blocks]
    assert "HorizA HorizB" in texts, texts
    assert "A B C" in texts, texts
    assert "X Y Z" in texts, texts
    assert "TallBoxLabel" in texts, texts


def test_bullet_list_preservation():
    lines = [
        (["•", "First", "bullet", "item"], 100.0),
        (["continues", "on", "next", "line"], 114.0),
        (["•", "Second", "bullet", "item"], 128.0),
        (["-", "Dash", "item", "one"], 142.0),
        (["*", "Star", "item", "two"], 156.0),
        (["1)", "Numbered", "paren"], 170.0),
        (["2.", "Numbered", "dot"], 184.0),
        (["[REQ_001]", "Requirement", "alpha"], 198.0),
        (["[REQ_002]", "Requirement", "beta"], 212.0),
    ]
    words: list[Word] = []
    for tokens, y0 in lines:
        x = 60.0
        for tok in tokens:
            w = max(8.0, len(tok) * 6.0)
            words.append(Word(tok, (x, y0, x + w, y0 + 10.0), 10.0, False))
            x += w + 4.0

    blocks = order_blocks(words, (0.0, 0.0, 595.0, 842.0))
    texts = [b.text for b in blocks]
    assert len(texts) == 8, texts
    assert texts[0] == "• First bullet item\ncontinues on next line"
    assert texts[1] == "• Second bullet item"
    assert texts[2] == "- Dash item one"
    assert texts[3] == "* Star item two"
    assert texts[4] == "1) Numbered paren"
    assert texts[5] == "2. Numbered dot"
    assert texts[6] == "[REQ_001] Requirement alpha"
    assert texts[7] == "[REQ_002] Requirement beta"


def test_scale_free_tolerances_and_overlapping_stress():
    boxes = [(50.0 + (i % 7) * 0.5, 50.0 + i * 0.25, 450.0, 62.0 + i * 0.25) for i in range(2500)]
    ordered = xy_cut_order(boxes, lambda b: b)
    assert len(ordered) == 2500
    assert Counter(ordered) == Counter(boxes)

    identical = [(50.0, 100.0, 450.0, 112.0) for _ in range(2200)]
    ordered_id = xy_cut_order(identical, lambda b: b, page_rect=(0.0, 0.0, 0.0, 0.0), font_size=0.0)
    assert len(ordered_id) == 2200

    zero_words = [
        Word("Left", (50.0, 100.0, 90.0, 112.0), 0.0, False),
        Word("Right", (320.0, 100.0, 360.0, 112.0), 0.0, False),
    ]
    blocks_no_rect = order_blocks(zero_words, page_rect=None)
    assert [b.text for b in blocks_no_rect] == ["Left", "Right"]

    scaled_words = [
        Word("UL", (100.0, 200.0, 180.0, 224.0), 20.0, False),
        Word("UR", (640.0, 200.0, 720.0, 224.0), 20.0, False),
    ]
    scaled_blocks = order_blocks(scaled_words, page_rect=(0.0, 0.0, 1190.0, 1684.0))
    assert [b.text for b in scaled_blocks] == ["UL", "UR"]


def test_rotated_words_upright_column_digits_multiline_vert_and_xy_cut_top_overlap():
    # Bug #7: 90 CCW (btt) and 270 CW (ttb) multi-word labels + /Rotate 90/270 pages.
    with tempfile.TemporaryDirectory() as tmp:
        b = TestPdfBuilder()
        p = b.new_page(width=595, height=842)
        p._ops.append(b"BT /F1 8.00 Tf 0 1 -1 0 60.00 500.00 Tm (data required) Tj ET")
        p._ops.append(b"BT /F1 8.00 Tf 0 -1 1 0 490.00 650.00 Tm (Complex Drivers) Tj ET")
        pdf_path = Path(tmp) / "rot_words.pdf"
        b.save(pdf_path)
        ir = extract_ir(pdf_path, figures=False, ocr=False)
        texts = [e["text"] for e in ir["elements"]]
        assert "data required" in texts, texts
        assert "Complex Drivers" in texts, texts

        for rot, expected_txt in ((90, "RotatedPage90"), (270, "RotatedPage270")):
            b_rot = TestPdfBuilder()
            pr = b_rot.new_page(width=600, height=800)
            pr.insert_text((100, 150), expected_txt, fontsize=11)
            raw = b_rot.to_bytes().replace(
                b"/MediaBox [0 0 600.00 800.00]",
                f"/MediaBox [0 0 600.00 800.00] /Rotate {rot}".encode("ascii"),
            )
            rot_path = Path(tmp) / f"page_rot_{rot}.pdf"
            rot_path.write_bytes(raw)
            ir_rot = extract_ir(rot_path, figures=False, ocr=False)
            assert [e["text"] for e in ir_rot["elements"]] == [expected_txt]

    # Bug #8: upright stacked column digits stay horizontal; adjacent vertical rotated lines stay vertical.
    stacked_words = [
        Word("1", (50.0, 100.0, 55.0, 110.0), 10.0, False, upright=True),
        Word("Alpha", (73.0, 100.0, 110.0, 110.0), 10.0, False, upright=True),
        Word("2", (50.0, 118.0, 55.0, 128.0), 10.0, False, upright=True),
        Word("Beta", (73.0, 118.0, 105.0, 128.0), 10.0, False, upright=True),
        Word("3", (50.0, 136.0, 55.0, 146.0), 10.0, False, upright=True),
        Word("Gamma", (73.0, 136.0, 115.0, 146.0), 10.0, False, upright=True),
    ]
    stacked_blocks = order_blocks(stacked_words, (0.0, 0.0, 595.0, 842.0))
    assert [b.text for b in stacked_blocks] == ["1 Alpha\n2 Beta\n3 Gamma"]

    vert_adj = [
        Word("Layered", (50.0, 100.0, 60.0, 160.0), 10.0, False, upright=False),
        Word("Software", (64.0, 100.0, 74.0, 170.0), 10.0, False, upright=False),
    ]
    vert_blocks = order_blocks(vert_adj, (0.0, 0.0, 595.0, 842.0))
    assert [b.text for b in vert_blocks] == ["Layered", "Software"]

    # Bug #9: wide top block overlapping parallel column must not split the column.
    xy_items = [
        ("LeftCol", (50.0, 105.0, 180.0, 300.0)),
        ("RightTop", (200.0, 100.0, 500.0, 140.0)),
        ("RightBot", (200.0, 160.0, 500.0, 300.0)),
        ("BotBanner", (50.0, 320.0, 500.0, 350.0)),
    ]
    assert [n for n, _ in xy_cut_order(xy_items, lambda it: it[1])] == [
        "LeftCol",
        "RightTop",
        "RightBot",
        "BotBanner",
    ]


def test_xy_cut_centered_gutter_footer_and_split_header():
    two_col_with_footer = [
        ("L1", (50.0, 90.0, 270.0, 140.0)),
        ("L2", (50.0, 160.0, 270.0, 240.0)),
        ("L3", (50.0, 260.0, 270.0, 340.0)),
        ("R1", (320.0, 90.0, 540.0, 140.0)),
        ("R2", (320.0, 160.0, 540.0, 240.0)),
        ("R3", (320.0, 260.0, 540.0, 340.0)),
        ("PageNum", (288.0, 380.0, 302.0, 394.0)),
    ]
    assert [n for n, _ in xy_cut_order(two_col_with_footer, lambda it: it[1])] == [
        "L1",
        "L2",
        "L3",
        "R1",
        "R2",
        "R3",
        "PageNum",
    ]

    single_col_split_hdr = [
        ("Header", (90.0, 80.0, 340.0, 96.0)),
        ("PageNum", (480.0, 80.0, 505.0, 96.0)),
        ("ShortIntro", (90.0, 120.0, 130.0, 136.0)),
        ("Formula", (160.0, 150.0, 420.0, 210.0)),
        ("Body", (90.0, 225.0, 505.0, 290.0)),
    ]
    assert [n for n, _ in xy_cut_order(single_col_split_hdr, lambda it: it[1])] == [
        "Header",
        "PageNum",
        "ShortIntro",
        "Formula",
        "Body",
    ]


def test_unicode_hanging_indent_and_vertical_outdent():
    # Hanging-indent markers across Latin, CJK (fullwidth/katakana/circled), and Arabic-Indic scripts.
    words = [
        Word("(a)", (50.0, 80.0, 66.0, 90.0), 10.0, False),
        Word("Latin", (92.0, 80.0, 122.0, 90.0), 10.0, False),
        Word("item", (126.0, 80.0, 152.0, 90.0), 10.0, False),
        Word("wraps", (92.0, 94.0, 125.0, 104.0), 10.0, False),
        Word("cleanly", (129.0, 94.0, 168.0, 104.0), 10.0, False),
        Word("iv)", (50.0, 110.0, 64.0, 120.0), 10.0, False),
        Word("Roman", (92.0, 110.0, 128.0, 120.0), 10.0, False),
        Word("item", (132.0, 110.0, 158.0, 120.0), 10.0, False),
        Word("ア．", (50.0, 126.0, 68.0, 136.0), 10.0, False),
        Word("カタカナ項目", (92.0, 126.0, 162.0, 136.0), 10.0, False),
        Word("①", (50.0, 142.0, 60.0, 152.0), 10.0, False),
        Word("丸数字項目", (92.0, 142.0, 155.0, 152.0), 10.0, False),
        Word("١.", (50.0, 158.0, 62.0, 168.0), 10.0, False),
        Word("Arabic", (92.0, 158.0, 128.0, 168.0), 10.0, False),
        Word("digit", (132.0, 158.0, 158.0, 168.0), 10.0, False),
    ]
    blocks = order_blocks(words, (0.0, 0.0, 595.0, 842.0))
    assert [b.text for b in blocks] == [
        "(a) Latin item\nwraps cleanly",
        "iv) Roman item",
        "ア． カタカナ項目",
        "① 丸数字項目",
        "١. Arabic digit",
    ], [b.text for b in blocks]

    # Raw segments before order_blocks merging remain unmerged for router table_candidate.
    raw_lines = extract_raw_line_segments(words, page_rect=(0.0, 0.0, 595.0, 842.0))
    assert len(raw_lines[0]) == 2 and len(raw_lines[2]) == 2

    # Non-ASCII word markers (not in _LIST_MARKER_RE) split cleanly via hanging indent + outdent.
    clause_words = [
        Word("第1条", (50.0, 200.0, 76.0, 210.0), 10.0, False),
        Word("第一項の本文", (100.0, 200.0, 175.0, 210.0), 10.0, False),
        Word("二行目の本文", (100.0, 214.0, 175.0, 224.0), 10.0, False),
        Word("第2条", (50.0, 228.0, 76.0, 238.0), 10.0, False),
        Word("第二項の本文", (100.0, 228.0, 175.0, 238.0), 10.0, False),
        Word("継続行の本文", (100.0, 242.0, 175.0, 252.0), 10.0, False),
    ]
    clause_blocks = order_blocks(clause_words, (0.0, 0.0, 595.0, 842.0))
    assert [b.text for b in clause_blocks] == [
        "第1条 第一項の本文\n二行目の本文",
        "第2条 第二項の本文\n継続行の本文",
    ], [b.text for b in clause_blocks]


def test_leader_and_right_aligned_merging():
    # Character leader, stroke leader, and space-separated right-aligned page numbers across >= 3 lines.
    char_leader_words = [
        Word("Chapter", (50.0, 100.0, 95.0, 110.0), 10.0, False),
        Word("1", (99.0, 100.0, 106.0, 110.0), 10.0, False),
        Word("...........", (130.0, 100.0, 440.0, 110.0), 10.0, False),
        Word("12", (470.0, 100.0, 484.0, 110.0), 10.0, False),
    ]
    assert [b.text for b in order_blocks(char_leader_words, (0.0, 0.0, 595.0, 842.0))] == [
        "Chapter 1 ........... 12"
    ]

    stroke_words = [
        Word("Appendix", (50.0, 130.0, 102.0, 140.0), 10.0, False),
        Word("A", (106.0, 130.0, 115.0, 140.0), 10.0, False),
        Word("99", (470.0, 130.0, 484.0, 140.0), 10.0, False),
    ]
    stroke_blocks = order_blocks(
        stroke_words,
        (0.0, 0.0, 595.0, 842.0),
        horiz_rules=[(120.0, 138.0, 465.0, 138.8)],
    )
    assert [b.text for b in stroke_blocks] == ["Appendix A 99"]

    toc_words = [
        Word("第1編", (43.0, 200.0, 75.0, 210.0), 10.0, False),
        Word("ご契約のしおり", (79.0, 200.0, 155.0, 210.0), 10.0, False),
        Word("4", (468.0, 200.0, 475.0, 210.0), 10.0, False),
        Word("第2編", (43.0, 216.0, 75.0, 226.0), 10.0, False),
        Word("普通保険約款", (79.0, 216.0, 148.0, 226.0), 10.0, False),
        Word("18", (462.0, 216.0, 475.0, 226.0), 10.0, False),
        Word("第3編", (43.0, 232.0, 75.0, 242.0), 10.0, False),
        Word("特約条項", (79.0, 232.0, 132.0, 242.0), 10.0, False),
        Word("104", (456.0, 232.0, 475.0, 242.0), 10.0, False),
    ]
    toc_blocks = order_blocks(toc_words, (0.0, 0.0, 595.0, 842.0))
    assert [b.text for b in toc_blocks] == [
        "第1編 ご契約のしおり 4\n第2編 普通保険約款 18\n第3編 特約条項 104"
    ], [b.text for b in toc_blocks]


def test_inline_bold_paragraph_vs_bold_heading():
    # Line 1 has >= 50% bold inline terms mixed with non-bold text: must stay one paragraph.
    inline_words = [
        Word("Important", (50.0, 100.0, 108.0, 111.0), 11.0, True),
        Word("coverage", (112.0, 100.0, 165.0, 111.0), 11.0, True),
        Word("terms", (169.0, 100.0, 202.0, 111.0), 11.0, True),
        Word("apply", (206.0, 100.0, 236.0, 111.0), 11.0, False),
        Word("when", (240.0, 100.0, 268.0, 111.0), 11.0, False),
        Word("filing", (50.0, 115.0, 82.0, 126.0), 11.0, False),
        Word("a", (86.0, 115.0, 93.0, 126.0), 11.0, False),
        Word("claim", (97.0, 115.0, 128.0, 126.0), 11.0, False),
        Word("online.", (132.0, 115.0, 172.0, 126.0), 11.0, False),
    ]
    inline_blocks = order_blocks(inline_words, (0.0, 0.0, 595.0, 842.0))
    assert len(inline_blocks) == 1, [b.text for b in inline_blocks]
    assert inline_blocks[0].text == "Important coverage terms apply when\nfiling a claim online."

    # Truly bold 100% heading at same font size above body text: must stay separate.
    heading_words = [
        Word("Section", (50.0, 180.0, 95.0, 191.0), 11.0, True),
        Word("Overview", (99.0, 180.0, 152.0, 191.0), 11.0, True),
        Word("This", (50.0, 195.0, 74.0, 206.0), 11.0, False),
        Word("policy", (78.0, 195.0, 112.0, 206.0), 11.0, False),
        Word("covers", (116.0, 195.0, 154.0, 206.0), 11.0, False),
        Word("damages.", (158.0, 195.0, 210.0, 206.0), 11.0, False),
    ]
    heading_blocks = order_blocks(heading_words, (0.0, 0.0, 595.0, 842.0))
    assert [b.text for b in heading_blocks] == ["Section Overview", "This policy covers damages."]


def test_negative_counterexamples_two_col_prose_first_line_indent_and_borderless_table():
    # 1. Real 2-col prose with short ending line in left column and justified right column must not merge.
    prose_2col = [
        Word("Left", (50.0, 100.0, 76.0, 110.0), 10.0, False),
        Word("column", (80.0, 100.0, 122.0, 110.0), 10.0, False),
        Word("continues", (126.0, 100.0, 182.0, 110.0), 10.0, False),
        Word("here", (186.0, 100.0, 215.0, 110.0), 10.0, False),
        Word("Right", (320.0, 100.0, 352.0, 110.0), 10.0, False),
        Word("column", (356.0, 100.0, 398.0, 110.0), 10.0, False),
        Word("full", (402.0, 100.0, 425.0, 110.0), 10.0, False),
        Word("justified", (429.0, 100.0, 478.0, 110.0), 10.0, False),
        Word("line", (482.0, 100.0, 540.0, 110.0), 10.0, False),
        Word("end.", (50.0, 114.0, 72.0, 124.0), 10.0, False),
        Word("Second", (320.0, 114.0, 362.0, 124.0), 10.0, False),
        Word("right", (366.0, 114.0, 394.0, 124.0), 10.0, False),
        Word("justified", (398.0, 114.0, 448.0, 124.0), 10.0, False),
        Word("line", (452.0, 114.0, 540.0, 124.0), 10.0, False),
        Word("New", (50.0, 128.0, 76.0, 138.0), 10.0, False),
        Word("left", (80.0, 128.0, 102.0, 138.0), 10.0, False),
        Word("prose", (106.0, 128.0, 140.0, 138.0), 10.0, False),
        Word("sentence", (144.0, 128.0, 200.0, 138.0), 10.0, False),
        Word("done", (510.0, 128.0, 540.0, 138.0), 10.0, False),
        Word("fin", (520.0, 142.0, 540.0, 152.0), 10.0, False),
        Word("end", (520.0, 156.0, 540.0, 166.0), 10.0, False),
    ]
    prose_blocks = order_blocks(prose_2col, (0.0, 0.0, 595.0, 842.0))
    for b in prose_blocks:
        assert not ("end." in b.text and "Second" in b.text), prose_blocks
        assert not ("sentence" in b.text and "done" in b.text), prose_blocks

    # 2. Normal paragraph with first-line indent must stay merged into one paragraph.
    first_indent = [
        Word("First", (70.0, 200.0, 98.0, 210.0), 10.0, False),
        Word("indented", (102.0, 200.0, 150.0, 210.0), 10.0, False),
        Word("line.", (154.0, 200.0, 180.0, 210.0), 10.0, False),
        Word("Second", (50.0, 214.0, 90.0, 224.0), 10.0, False),
        Word("unindented", (94.0, 214.0, 154.0, 224.0), 10.0, False),
        Word("line.", (158.0, 214.0, 184.0, 224.0), 10.0, False),
        Word("Third", (50.0, 228.0, 80.0, 238.0), 10.0, False),
        Word("line.", (84.0, 228.0, 110.0, 238.0), 10.0, False),
    ]
    assert [b.text for b in order_blocks(first_indent, (0.0, 0.0, 595.0, 842.0))] == [
        "First indented line.\nSecond unindented line.\nThird line."
    ]

    # 3. 2-column borderless table (non-marker IDs) must not merge horizontally in order_blocks.
    table_2col = [
        Word("ID1", (50.0, 300.0, 70.0, 310.0), 10.0, False),
        Word("Description", (180.0, 300.0, 245.0, 310.0), 10.0, False),
        Word("alpha", (249.0, 300.0, 280.0, 310.0), 10.0, False),
        Word("ID2", (50.0, 316.0, 70.0, 326.0), 10.0, False),
        Word("Description", (180.0, 316.0, 245.0, 326.0), 10.0, False),
        Word("beta", (249.0, 316.0, 276.0, 326.0), 10.0, False),
        Word("ID3", (50.0, 332.0, 70.0, 342.0), 10.0, False),
        Word("Description", (180.0, 332.0, 245.0, 342.0), 10.0, False),
        Word("gamma", (249.0, 332.0, 286.0, 342.0), 10.0, False),
    ]
    tbl_blocks = order_blocks(table_2col, (0.0, 0.0, 595.0, 842.0))
    assert not any("ID1" in b.text and "Description" in b.text for b in tbl_blocks), tbl_blocks


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
