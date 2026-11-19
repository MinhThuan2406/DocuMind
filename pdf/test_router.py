import io
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pdfplumber  # noqa: E402
from PIL import Image  # noqa: E402

from pdf._test_pdf import TestPdfBuilder  # noqa: E402
from pdf.router import classify_page  # noqa: E402


def _png_bytes(width=400, height=600):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(240, 240, 240)).save(buf, format="PNG")
    return buf.getvalue()


def _classify_builder(builder: TestPdfBuilder, index: int = 0):
    with pdfplumber.open(io.BytesIO(builder.to_bytes())) as doc:
        return classify_page(doc.pages[index])


def test_synthetic_native_text_page():
    b = TestPdfBuilder()
    page = b.new_page(width=595, height=842)
    page.insert_text((72, 72), "AUTOSAR layered software architecture overview.\n" * 4, fontsize=12)
    profile = _classify_builder(b)
    assert profile.page == 1
    assert profile.kind == "native"
    assert profile.word_count >= 20
    assert not profile.suspect_ocr_layer
    assert profile.image_area_ratio == 0.0


def test_synthetic_raster_image_only_page():
    b = TestPdfBuilder()
    page = b.new_page(width=400, height=600)
    page.insert_image((0, 0, 400, 600), stream=_png_bytes(400, 600))
    profile = _classify_builder(b)
    assert profile.kind == "raster"
    assert profile.word_count == 0
    assert profile.image_area_ratio >= 0.99
    assert not profile.suspect_ocr_layer


def test_synthetic_mixed_hybrid_page():
    b = TestPdfBuilder()
    page = b.new_page(width=595, height=842)
    page.insert_image((50, 100, 300, 500), stream=_png_bytes(250, 400))
    page.insert_text((72, 72), "Figure 1", fontsize=11)
    profile = _classify_builder(b)
    assert profile.kind == "hybrid"
    assert profile.word_count == 2
    assert 0.1 < profile.image_area_ratio < 0.96


class _FakePage:
    def __init__(self, words, width=595.0, height=842.0):
        self.page_number = 1
        self.bbox = (0.0, 0.0, width, height)
        self.images = []
        self.lines = []
        self.rects = []
        self.curves = []
        self._words = [
            {"x0": w[0], "top": w[1], "x1": w[2], "bottom": w[3], "text": w[4]} for w in words
        ]

    def extract_words(self, **_kw):
        return self._words


def test_synthetic_suspect_ocr_layer_cases():
    p1 = _FakePage([
        (72.0, 60.0, 120.0, 74.0, "\ufffd\ufffd\ufffd"),
        (124.0, 60.0, 150.0, 74.0, "bad"),
        (154.0, 60.0, 175.0, 74.0, "ocr"),
        (179.0, 60.0, 205.0, 74.0, "text"),
        (209.0, 60.0, 240.0, 74.0, "layer"),
    ])
    prof1 = classify_page(p1)
    assert prof1.suspect_ocr_layer and prof1.kind == "hybrid"

    b = TestPdfBuilder()
    p2 = b.new_page(width=595, height=842)
    p2.insert_image((0, 0, 595, 842), stream=_png_bytes(595, 842))
    p2.insert_text((72, 72), "hidden tiny ocr text words", fontsize=2)
    prof2 = _classify_builder(b)
    assert prof2.suspect_ocr_layer and prof2.kind == "hybrid"


def test_figure_candidate_signals():
    b = TestPdfBuilder()
    text_only = b.new_page(width=595, height=842)
    text_only.insert_text((72, 72), "Plain paragraph without any graphics on the page.", fontsize=12)
    text_only.draw_line((50, 40), (545, 40))

    many_boxes = b.new_page(width=595, height=842)
    for i in range(8):
        many_boxes.draw_rect((60 + i * 60, 200, 100 + i * 60, 240))

    large_image = b.new_page(width=595, height=842)
    large_image.insert_image((50, 100, 300, 400), stream=_png_bytes(250, 300))

    logo_only = b.new_page(width=595, height=842)
    logo_only.insert_image((50, 20, 90, 50), stream=_png_bytes(40, 30))

    footer_graphics = b.new_page(width=595, height=842)
    for i in range(8):
        footer_graphics.draw_rect((60 + i * 60, 790, 100 + i * 60, 810))

    with pdfplumber.open(io.BytesIO(b.to_bytes())) as doc:
        assert not classify_page(doc.pages[0]).figure_candidate
        assert classify_page(doc.pages[1]).figure_candidate
        assert classify_page(doc.pages[2]).figure_candidate
        assert not classify_page(doc.pages[3]).figure_candidate
        assert not classify_page(doc.pages[4]).figure_candidate


def test_autosar_and_omnidocbench_datasets():
    autosar_dir = ROOT / "datasets" / "AUTOSAR"
    if not autosar_dir.exists():
        print("SKIP AUTOSAR datasets not present")
    else:
        for pdf_path in sorted(autosar_dir.glob("*.pdf")):
            with pdfplumber.open(str(pdf_path)) as doc:
                counts = Counter(classify_page(page).kind for page in doc.pages)
            print(f"  AUTOSAR {pdf_path.name}: {dict(counts)}")
            assert counts.get("raster", 0) == 0, (pdf_path.name, counts)

    omni_dir = ROOT / "datasets" / "OmniDocBench" / "images"
    if not omni_dir.exists():
        print("SKIP OmniDocBench images not present")
    else:
        pngs = sorted(omni_dir.glob("*.png"))[:30]
        assert len(pngs) == 30, len(pngs)
        counts = Counter()
        for png_path in pngs:
            with Image.open(png_path) as img:
                w, h = img.size
            b = TestPdfBuilder()
            pw, ph = w * 72.0 / 150.0, h * 72.0 / 150.0
            page = b.new_page(width=pw, height=ph)
            page.insert_image((0, 0, pw, ph), stream=png_path.read_bytes())
            counts[_classify_builder(b).kind] += 1
        print(f"  OmniDocBench 30 wrapped PNGs: {dict(counts)}")
        assert counts["raster"] == 30, counts


def test_no_heavy_ml_imports():
    assert "docling" not in sys.modules
    assert "torch" not in sys.modules
    assert "pymupdf" not in sys.modules


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
