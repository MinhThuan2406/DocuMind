import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pymupdf  # noqa: E402

import pdf  # noqa: E402
from pdf import extract_ir  # noqa: E402


def _make_two_column_pdf(path: Path) -> None:
    doc = pymupdf.open()
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text((60, 60), "TITLE BANNER ACROSS BOTH COLUMNS", fontsize=16)
    p1.insert_text((60, 110), "L1 left column first paragraph\nL2 left column second line", fontsize=11)
    p1.insert_text((60, 170), "L3 left column bottom paragraph", fontsize=11)
    p1.insert_text((340, 110), "R1 right column first paragraph\nR2 right column second line", fontsize=11)
    p1.insert_text((340, 170), "R3 right column bottom paragraph", fontsize=11)

    p2 = doc.new_page(width=595, height=842)
    p2.insert_text((60, 80), "Page two single column content", fontsize=12)
    doc.save(str(path))
    doc.close()


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

    doc = pymupdf.open(str(layered))
    ms_per_page = (elapsed / len(doc)) * 1000.0
    print(f"  Layered {len(doc)} pages: {elapsed:.3f}s total ({ms_per_page:.2f} ms/page)")
    assert ms_per_page < 50.0, f"too slow: {ms_per_page:.2f} ms/page"

    ir_words_by_page: dict[int, Counter[str]] = {}
    for elem in ir["elements"]:
        assert elem["type"] == "text"
        ir_words_by_page.setdefault(elem["page"], Counter()).update(elem["text"].split())

    for page_idx, page in enumerate(doc, start=1):
        expected = Counter(w[4] for w in page.get_text("words"))
        actual = ir_words_by_page.get(page_idx, Counter())
        assert actual == expected, (page_idx, expected - actual, actual - expected)
    doc.close()


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
