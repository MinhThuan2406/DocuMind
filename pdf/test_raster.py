import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pymupdf  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import pdf.raster as raster_mod  # noqa: E402
from pdf import extract_ir  # noqa: E402
from pdf.layout.types import LayoutItem  # noqa: E402
from pdf.ocr.types import OcrLine  # noqa: E402


def _make_raster_pdf(path: Path, text: str = "RASTER OCR SAMPLE") -> None:
    img = Image.new("RGB", (600, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((40, 80), text, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    doc = pymupdf.open()
    page = doc.new_page(width=600, height=300)
    page.insert_image(page.rect, stream=buf.getvalue())
    doc.save(str(path))
    doc.close()


def test_importing_raster_does_not_load_docling_or_torch():
    assert "docling" not in sys.modules
    assert "torch" not in sys.modules


def test_ocr_disabled_skips_ml_and_marks_unreadable_page():
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "scan.pdf"
        _make_raster_pdf(pdf_path)
        ir = extract_ir(pdf_path, figures=False, ocr=False)
        assert "docling" not in sys.modules
        assert "torch" not in sys.modules
        assert ir["elements"] == []
        assert ir["metadata"]["unreadable_pages"] == [{"page": 1, "reason": "raster_page_ocr_disabled"}]


def test_read_raster_page_and_ir_emit_ocr_source_and_confidence():
    class _FakeDetector:
        def detect(self, image):
            w, h = float(image.width), float(image.height)
            sx, sy = w / 600.0, h / 300.0
            return [
                LayoutItem("section_header", (10 * sx, 10 * sy, 220 * sx, 45 * sy), 0.92),
                LayoutItem("text", (10 * sx, 50 * sy, 320 * sx, 95 * sy), 0.90),
                LayoutItem("picture", (10 * sx, 110 * sy, 250 * sx, 240 * sy), 0.94),
            ]

    class _FakeRecognizer:
        def recognize(self, image, regions=None):
            del regions
            w, h = float(image.width), float(image.height)
            sx, sy = w / 600.0, h / 300.0
            return [
                OcrLine("Heading", (15 * sx, 15 * sy, 180 * sx, 40 * sy), 0.91),
                OcrLine("Body OCR line", (15 * sx, 55 * sy, 290 * sx, 88 * sy), 0.93),
            ]

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "scan.pdf"
        _make_raster_pdf(pdf_path)
        orig_det = raster_mod.get_layout_detector
        orig_rec = raster_mod._get_recognizer
        try:
            raster_mod.get_layout_detector = lambda _backend, _device: (_FakeDetector(), "fake")
            raster_mod._get_recognizer = lambda _backend: _FakeRecognizer()
            ir = extract_ir(pdf_path, figures=False, ocr=True)
        finally:
            raster_mod.get_layout_detector = orig_det
            raster_mod._get_recognizer = orig_rec

    assert json.loads(json.dumps(ir)) == ir
    assert ir["metadata"]["unreadable_pages"] == []
    elems = ir["elements"]
    assert len(elems) == 3
    assert [e["type"] for e in elems] == ["text", "text", "figure"]
    assert elems[0]["metadata"]["layout_label"] == "section_header"
    assert elems[1]["metadata"]["layout_label"] == "text"
    assert elems[0]["confidence"] == 0.91 and elems[0]["source"] == "ocr"
    assert elems[1]["confidence"] == 0.93 and elems[1]["source"] == "ocr"
    assert elems[2]["source"] == "layout_picture"
    assert "docling" not in sys.modules


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
