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
from pdf.text import extract_words, order_blocks, xy_cut_order  # noqa: E402


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
