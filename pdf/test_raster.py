import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pymupdf  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from pdf import extract_ir  # noqa: E402
from pdf.backends import ComposedReader, register  # noqa: E402
from pdf.layout import mineru_middle_json_to_raster_pages  # noqa: E402
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
        register("detector", "fake_det", lambda **_kw: _FakeDetector())
        register("ocr", "fake_ocr", lambda **_kw: _FakeRecognizer())
        ir = extract_ir(pdf_path, figures=False, ocr=True, layout="fake_det", ocr_backend="fake_ocr")
        assert isinstance(ComposedReader(_FakeDetector(), _FakeRecognizer()).read(pdf_path, [1])[1].blocks, tuple)

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


def test_mineru_middle_json_to_raster_pages_keeps_headers_and_ocr_source():
    middle_json = {
        "pages": [
            {
                "page_idx": 0,
                "blocks": [
                    {
                        "type": "header",
                        "index": 0,
                        "bbox": [0.1, 0.05, 0.5, 0.08],
                        "content": [{"type": "text", "content": "Section Header", "styles": []}],
                    },
                    {
                        "type": "page_number",
                        "index": 1,
                        "bbox": [0.8, 0.05, 0.9, 0.08],
                        "content": [{"type": "text", "content": "25", "styles": []}],
                    },
                    {
                        "type": "text",
                        "index": 2,
                        "bbox": [0.1, 0.12, 0.9, 0.25],
                        "content": [
                            {"type": "text", "content": "Solve", "styles": []},
                            {"type": "equation_inline", "content": "x^2 = 1"},
                        ],
                    },
                    {
                        "type": "table",
                        "index": 3,
                        "bbox": [0.1, 0.30, 0.9, 0.60],
                        "content": [
                            {"type": "table_caption", "content": [{"type": "text", "content": "Table 1", "styles": []}]},
                            {"type": "table_body", "content": "<table><tr><td>A</td><td>1</td></tr></table>"},
                        ],
                    },
                    {
                        "type": "image",
                        "index": 4,
                        "bbox": [0.1, 0.65, 0.9, 0.90],
                        "content": [
                            {"type": "image_body", "image_path": "fig.png"},
                            {"type": "image_caption", "content": [{"type": "text", "content": "Figure 1", "styles": []}]},
                        ],
                    },
                ],
            }
        ]
    }
    pages = mineru_middle_json_to_raster_pages(middle_json, {1: (600.0, 800.0)})
    rp = pages[1]
    assert len(rp.figures) == 2
    assert [f.kind for f in rp.figures] == ["table", "figure"]
    texts = [b.text for b in rp.blocks]
    assert "Section Header" in texts
    assert "25" in texts
    assert any("x^2 = 1" in t for t in texts)
    assert "Table 1" in texts
    assert any("<td>A</td>" in t for t in texts)
    assert "Figure 1" in texts
    assert all(b.source == "ocr" and isinstance(b.confidence, float) for b in rp.blocks)


def test_extract_ir_batch_calls_read_raster_pages_once_for_mineru():
    import pdf.backends.mineru as mp_backend_mod

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "scan2.pdf"
        doc = pymupdf.open()
        for label in ("Page One", "Page Two"):
            img = Image.new("RGB", (600, 300), color=(255, 255, 255))
            ImageDraw.Draw(img).text((40, 80), label, fill=(0, 0, 0))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            p = doc.new_page(width=600, height=300)
            p.insert_image(p.rect, stream=buf.getvalue())
        doc.save(str(pdf_path))
        doc.close()

        calls = 0
        fake_mj = {
            "pages": [
                {"page_idx": 0, "blocks": [{"type": "text", "index": 0, "bbox": [0.1, 0.1, 0.5, 0.3], "content": [{"type": "text", "content": "P1"}]}]},
                {"page_idx": 1, "blocks": [{"type": "text", "index": 0, "bbox": [0.1, 0.1, 0.5, 0.3], "content": [{"type": "text", "content": "P2"}]}]},
            ]
        }

        def _fake_run(path, **kwargs):
            nonlocal calls
            del path, kwargs
            calls += 1
            return fake_mj, {"pages_processed": 2}

        orig = mp_backend_mod.run_mineru_pdf
        try:
            mp_backend_mod.run_mineru_pdf = _fake_run
            ir = extract_ir(pdf_path, figures=False, ocr=True, reader="mineru-pipeline")
        finally:
            mp_backend_mod.run_mineru_pdf = orig

        assert calls == 1
        assert [e["text"] for e in ir["elements"]] == ["P1", "P2"]
        assert all(e["source"] == "ocr" and isinstance(e["confidence"], float) for e in ir["elements"])

        ir_off = extract_ir(pdf_path, figures=False, ocr=False, reader="mineru-pipeline")
        assert ir_off["elements"] == []
        assert len(ir_off["metadata"]["unreadable_pages"]) == 2


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
