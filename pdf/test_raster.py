import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pymupdf  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import pdf.raster as raster_mod  # noqa: E402
from pdf import extract_ir  # noqa: E402


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


def test_ocr_disabled_skips_docling_and_marks_unreadable_page():
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "scan.pdf"
        _make_raster_pdf(pdf_path)
        ir = extract_ir(pdf_path, figures=False, ocr=False)
        assert "docling" not in sys.modules
        assert "torch" not in sys.modules
        assert ir["elements"] == []
        assert ir["metadata"]["unreadable_pages"] == [{"page": 1, "reason": "raster_page_ocr_disabled"}]


def test_read_raster_page_and_ir_emit_ocr_source_and_confidence():
    class _FakeBBox:
        def __init__(self, l, t, r, b):
            self.l, self.t, self.r, self.b = l, t, r, b

        def to_top_left_origin(self, _h):
            return self

    class _FakeConverter:
        def convert(self, _path):
            items = [
                (SimpleNamespace(label="section_header", text="Heading", prov=[SimpleNamespace(bbox=_FakeBBox(10, 10, 200, 40))]), 1),
                (SimpleNamespace(label="text", text="Body OCR line", prov=[SimpleNamespace(bbox=_FakeBBox(10, 50, 300, 90))]), 1),
                (SimpleNamespace(label="picture", text="", prov=[SimpleNamespace(bbox=_FakeBBox(10, 110, 250, 240))]), 1),
            ]
            doc = SimpleNamespace(
                pages={1: SimpleNamespace(size=SimpleNamespace(width=600.0, height=300.0))},
                iterate_items=lambda: items,
            )
            conf = SimpleNamespace(pages={1: SimpleNamespace(ocr_score=0.9123)})
            return SimpleNamespace(document=doc, confidence=conf)

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "scan.pdf"
        _make_raster_pdf(pdf_path)
        orig_builder = raster_mod._build_ocr_converter
        try:
            raster_mod._build_ocr_converter = lambda _device: _FakeConverter()
            ir = extract_ir(pdf_path, figures=False, ocr=True)
        finally:
            raster_mod._build_ocr_converter = orig_builder

    assert json.loads(json.dumps(ir)) == ir
    assert ir["metadata"]["unreadable_pages"] == []
    elems = ir["elements"]
    assert len(elems) == 3
    assert [e["type"] for e in elems] == ["text", "text", "figure"]
    assert elems[0]["metadata"]["docling_label"] == "section_header"
    assert elems[1]["metadata"]["docling_label"] == "text"
    for e in elems:
        assert e["confidence"] == 0.9123
        if e["type"] == "text":
            assert e["source"] == "ocr"


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
