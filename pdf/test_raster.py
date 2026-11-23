import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw  # noqa: E402

from pdf import extract_ir  # noqa: E402
from pdf._test_pdf import TestPdfBuilder  # noqa: E402
from pdf.assemble import assemble_page  # noqa: E402
from pdf.backends import ComposedReader, register  # noqa: E402
from pdf.backends.mineru import mineru_middle_json_to_raster_pages  # noqa: E402
from pdf.layout.types import LayoutItem  # noqa: E402
from pdf.models import FigureBox, PageProfile, RasterPage, TextBlock  # noqa: E402
from pdf.ocr.types import OcrLine  # noqa: E402


def _make_raster_pdf(path: Path, text: str = "RASTER OCR SAMPLE") -> None:
    img = Image.new("RGB", (600, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((40, 80), text, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    b = TestPdfBuilder()
    page = b.new_page(width=600, height=300)
    page.insert_image((0, 0, 600, 300), stream=buf.getvalue())
    b.save(path)


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

        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(str(pdf_path))
        try:
            shared_res = ComposedReader(_FakeDetector(), _FakeRecognizer()).read(pdf_path, [1], pdfium_doc=doc)
            assert len(doc) == 1 and shared_res[1].figures[0].confidence == 0.94
        finally:
            doc.close()

    assert json.loads(json.dumps(ir)) == ir
    assert ir["metadata"]["unreadable_pages"] == []
    elems = ir["elements"]
    assert len(elems) == 3
    assert [e["type"] for e in elems] == ["text", "text", "figure"]
    assert elems[0]["metadata"]["layout_label"] == "section_header"
    assert elems[1]["metadata"]["layout_label"] == "text"
    assert elems[0]["confidence"] == 0.91 and elems[0]["source"] == "ocr"
    assert elems[1]["confidence"] == 0.93 and elems[1]["source"] == "ocr"
    assert elems[2]["source"] == "layout_picture" and elems[2]["confidence"] == 0.94
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
        b = TestPdfBuilder()
        for label in ("Page One", "Page Two"):
            img = Image.new("RGB", (600, 300), color=(255, 255, 255))
            ImageDraw.Draw(img).text((40, 80), label, fill=(0, 0, 0))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            p = b.new_page(width=600, height=300)
            p.insert_image((0, 0, 600, 300), stream=buf.getvalue())
        b.save(pdf_path)

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


def test_hybrid_suspect_ocr_layer_fallback_vs_ocr_disabled():
    class _Det:
        def detect(self, image):
            w, h = float(image.width), float(image.height)
            sx, sy = w / 595.0, h / 842.0
            return [LayoutItem("text", (50 * sx, 60 * sy, 320 * sx, 110 * sy), 0.92)]

    class _Ocr:
        def recognize(self, image, regions=None):
            del regions
            w, h = float(image.width), float(image.height)
            sx, sy = w / 595.0, h / 842.0
            return [OcrLine("Clean OCR fallback", (55 * sx, 65 * sy, 300 * sx, 100 * sy), 0.94)]

    img = Image.new("RGB", (595, 842), color=(245, 245, 245))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "hybrid_suspect.pdf"
        b = TestPdfBuilder()
        page = b.new_page(width=595, height=842)
        page.insert_image((0, 0, 595, 842), stream=buf.getvalue())
        page.insert_text((72, 72), "hidden tiny ocr text words", fontsize=2)
        b.save(pdf_path)

        register("detector", "fake_hyb_det", lambda **_kw: _Det())
        register("ocr", "fake_hyb_ocr", lambda **_kw: _Ocr())

        import pdf.figures as figures_mod

        orig_detect = figures_mod.detect_figures
        figures_mod.detect_figures = lambda *_a, **_kw: (_ for _ in ()).throw(
            AssertionError("detect_figures should not run on suspect_ocr_layer page when ocr=True")
        )
        try:
            ir_on = extract_ir(pdf_path, figures=True, ocr=True, layout="fake_hyb_det", ocr_backend="fake_hyb_ocr")
        finally:
            figures_mod.detect_figures = orig_detect

        assert ir_on["metadata"]["unreadable_pages"] == []
        assert ir_on["metadata"]["warnings"] == [
            {
                "page": 1,
                "code": "suspect_ocr_layer",
                "severity": "warning",
                "reason": "suspect_ocr_layer",
                "fallback": "ocr",
            }
        ]
        assert [e["text"] for e in ir_on["elements"]] == ["Clean OCR fallback"]
        assert ir_on["elements"][0]["source"] == "ocr"

        ir_off = extract_ir(pdf_path, figures=False, ocr=False)
        assert ir_off["metadata"]["unreadable_pages"] == []
        assert ir_off["metadata"]["warnings"] == [
            {
                "page": 1,
                "code": "suspect_ocr_layer",
                "severity": "warning",
                "reason": "suspect_ocr_layer",
                "fallback": None,
            }
        ]
        assert any("hidden" in e["text"] for e in ir_off["elements"])
        assert all(e["source"] == "text_layer" for e in ir_off["elements"])


def test_raster_table_child_nesting_and_empty_table():
    class _Det:
        def detect(self, image):
            w, h = float(image.width), float(image.height)
            sx, sy = w / 600.0, h / 300.0
            return [
                LayoutItem("page_header", (20 * sx, 10 * sy, 220 * sx, 28 * sy), 0.91),
                LayoutItem("table", (20 * sx, 50 * sy, 320 * sx, 200 * sy), 0.93),
                LayoutItem("page_footer", (20 * sx, 270 * sy, 220 * sx, 290 * sy), 0.90),
            ]

    class _Ocr:
        def recognize(self, image, regions=None):
            del regions
            w, h = float(image.width), float(image.height)
            sx, sy = w / 600.0, h / 300.0
            return [
                OcrLine("Header Line", (25 * sx, 12 * sy, 180 * sx, 26 * sy), 0.92),
                OcrLine("ColA 100", (30 * sx, 70 * sy, 200 * sx, 95 * sy), 0.89),
                OcrLine("Footer 1", (25 * sx, 272 * sy, 150 * sx, 288 * sy), 0.90),
            ]

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "raster_tbl.pdf"
        _make_raster_pdf(pdf_path)
        register("detector", "fake_tbl_det", lambda **_kw: _Det())
        register("ocr", "fake_tbl_ocr", lambda **_kw: _Ocr())
        ir = extract_ir(pdf_path, figures=False, ocr=True, layout="fake_tbl_det", ocr_backend="fake_tbl_ocr")

    elems = ir["elements"]
    assert [e["reading_order"] for e in elems] == list(range(len(elems)))
    assert [e["type"] for e in elems] == ["text", "table", "text", "text"]

    hdr, tbl, tbl_child, ftr = elems
    assert hdr["level"] == 1 and hdr["parent_id"] is None
    assert hdr["metadata"]["layout_label"] == "page_header"
    assert hdr["metadata"]["role"] == "page_header"

    assert tbl["level"] == 1 and tbl["parent_id"] is None and tbl["text"] == ""
    assert tbl_child["level"] == 2 and tbl_child["parent_id"] == tbl["id"]
    assert tbl_child["metadata"]["layout_label"] == "table"
    assert tbl_child["text"] == "ColA 100"

    assert ftr["level"] == 1 and ftr["parent_id"] is None
    assert ftr["metadata"]["layout_label"] == "page_footer"
    assert ftr["metadata"]["role"] == "page_footer"

    empty_tbl = FigureBox((50.0, 50.0, 300.0, 180.0), "table", "layout_table", 0.95)
    rp_empty = RasterPage(blocks=(), figures=(empty_tbl,), reading_order=())
    items_empty = assemble_page(PageProfile(1, "raster", 0, 0.0, 1.0, False), [], (0.0, 0.0, 600.0, 300.0), raster_page=rp_empty)
    assert len(items_empty) == 1 and items_empty[0].kind == "table" and items_empty[0].level == 1

    # Overlapping table TextBlock slightly shifted outside box center still nests under the table.
    shifted_tbl_blk = TextBlock("Row", (200.0, 60.0, 360.0, 160.0), 0.0, False, "ocr", 0.9, label="table")
    rp_shifted = RasterPage(blocks=(shifted_tbl_blk,), figures=(empty_tbl,), reading_order=(0,))
    items_shifted = assemble_page(
        PageProfile(1, "raster", 0, 0.0, 1.0, False), [], (0.0, 0.0, 600.0, 300.0), raster_page=rp_shifted
    )
    assert len(items_shifted) == 2
    assert items_shifted[0].kind == "table" and items_shifted[0].level == 1
    assert items_shifted[1].level == 2 and items_shifted[1].parent_index == 0 and items_shifted[1].label == "table"


def test_zero_page_pdf_extract_ir_structure_true_without_handle_leak():
    import pypdfium2 as pdfium

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "empty.pdf"
        doc = pdfium.PdfDocument.new()
        doc.save(str(pdf_path))
        doc.close()

        ir = extract_ir(pdf_path, figures=True, ocr=True, tables=True, structure=True)
        assert ir["metadata"]["pages"] == []
        assert ir["metadata"]["warnings"] == []
        assert ir["metadata"]["outline"] == []
        assert ir["elements"] == []
        assert ir["tables"] == []


def test_pass2_high_dpi_refinement_accept_rollback_and_font_size():
    class _Det:
        def detect(self, image):
            sx, sy = image.width / 600.0, image.height / 300.0
            return [
                LayoutItem("text", (20 * sx, 20 * sy, 260 * sx, 60 * sy), 0.92),
                LayoutItem("text", (20 * sx, 80 * sy, 260 * sx, 120 * sy), 0.91),
                LayoutItem("text", (20 * sx, 140 * sy, 260 * sx, 180 * sy), 0.90),
                LayoutItem("picture", (300 * sx, 20 * sy, 560 * sx, 200 * sy), 0.95),
            ]

    class _Ocr:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def recognize(self, image, regions=None):
            del regions
            self.calls.append(image.size)
            if len(self.calls) == 1:
                sx, sy = image.width / 600.0, image.height / 300.0
                return [
                    OcrLine("Pass1 low conf", (25 * sx, 28 * sy, 240 * sx, 48 * sy), 0.62),
                    OcrLine("Pass1 rollback", (25 * sx, 88 * sy, 240 * sx, 108 * sy), 0.70),
                ]
            if len(self.calls) == 2:
                return [OcrLine("Pass2 improved", (10.0, 10.0, image.width - 10.0, image.height - 10.0), 0.91)]
            if len(self.calls) == 3:
                return [OcrLine("Pass2 worse", (10.0, 10.0, image.width - 10.0, image.height - 10.0), 0.55)]
            return [OcrLine("Pass2 recovered", (10.0, 10.0, image.width - 10.0, image.height - 10.0), 0.84)]

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "pass2.pdf"
        _make_raster_pdf(pdf_path)
        ocr = _Ocr()
        reader = ComposedReader(_Det(), ocr, dpi=150, high_dpi=275)
        rp = reader.read(pdf_path, [1])[1]

    assert len(ocr.calls) == 4
    assert [b.text for b in rp.blocks] == ["Pass2 improved", "Pass1 rollback", "Pass2 recovered"]
    assert [b.confidence for b in rp.blocks] == [0.91, 0.70, 0.84]
    assert [b.source for b in rp.blocks] == ["ocr_high_dpi", "ocr", "ocr_high_dpi"]
    assert all(b.font_size > 0.0 for b in rp.blocks)
    assert abs(rp.blocks[1].font_size - round(20.0 * 0.85, 2)) <= 0.1
    assert reader.last_stats is not None and reader.last_stats.warnings == ()


def test_orphan_ocr_lines_cluster_into_blocks_and_respect_pictures():
    class _Det:
        def detect(self, image):
            sx, sy = image.width / 600.0, image.height / 300.0
            return [LayoutItem("picture", (260 * sx, 40 * sy, 330 * sx, 220 * sy), 0.95)]

    class _Ocr:
        def recognize(self, image, regions=None):
            del regions
            sx, sy = image.width / 600.0, image.height / 300.0
            return [
                OcrLine("Left line one", (30 * sx, 40 * sy, 220 * sx, 56 * sy), 0.90),
                OcrLine("Left line two", (30 * sx, 60 * sy, 210 * sx, 76 * sy), 0.92),
                OcrLine("Left line three", (30 * sx, 80 * sy, 200 * sx, 96 * sy), 0.94),
                OcrLine("Right col one", (360 * sx, 40 * sy, 560 * sx, 56 * sy), 0.88),
                OcrLine("Right col two", (360 * sx, 60 * sy, 540 * sx, 76 * sy), 0.90),
                OcrLine("Inside picture", (270 * sx, 100 * sy, 320 * sx, 120 * sy), 0.85),
            ]

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "orphans.pdf"
        _make_raster_pdf(pdf_path)
        reader = ComposedReader(_Det(), _Ocr())
        rp = reader.read(pdf_path, [1])[1]

    assert len(rp.figures) == 1 and rp.figures[0].kind == "figure"
    assert len(rp.blocks) == 2
    assert rp.blocks[0].text == "Left line one Left line two Left line three"
    assert rp.blocks[1].text == "Right col one Right col two"
    assert all(b.source == "ocr" and b.label == "text" for b in rp.blocks)
    assert all(abs(b.font_size - round(16.0 * 0.85, 2)) <= 0.15 for b in rp.blocks)


def test_read_raster_regions_alignment_sorting_and_handle_cleanup():
    from pdf.raster import read_raster_regions

    class _Det:
        def detect(self, image):
            del image
            return []

    class _Ocr:
        def recognize(self, image, regions=None):
            del regions
            w, h = float(image.width), float(image.height)
            return [
                OcrLine("Line A", (0.1 * w, 0.1 * h, 0.8 * w, 0.3 * h), 0.92),
                OcrLine("Line B", (0.1 * w, 0.35 * h, 0.7 * w, 0.55 * h), 0.94),
            ]

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "regions.pdf"
        _make_raster_pdf(pdf_path)
        reader = ComposedReader(_Det(), _Ocr())
        region = (100.0, 50.0, 300.0, 150.0)
        res = read_raster_regions(pdf_path, {1: [region]}, reader=reader, dpi=275)

        assert list(res.keys()) == [1]
        assert len(res[1]) == 1
        blk = res[1][0]
        assert blk.text == "Line A Line B"
        assert blk.source == "ocr_high_dpi"
        assert blk.font_size > 0.0
        exp_box = (120.0, 60.0, 260.0, 105.0)
        for got, exp in zip(blk.bbox, exp_box):
            assert abs(got - exp) <= 0.5, (blk.bbox, exp_box)

        try:
            reader.read_raster_regions(pdf_path, {2: [region]})
            raise AssertionError("expected ValueError for out-of-range page")
        except ValueError:
            pass
        res_150 = read_raster_regions(pdf_path, {1: [region]}, reader=reader, dpi=150)
        assert len(res_150[1]) == 1 and res_150[1][0].source == "ocr"
        assert reader.read_raster_regions(pdf_path, {}) == {}
        assert reader.read_raster_regions(pdf_path, {1: []}) == {1: []}
        pdf_path.unlink()


def test_read_raster_regions_cropbox_and_rotate_alignment():
    import pdfplumber
    from pdf._pdf_io import page_bbox
    from pdf.raster import read_raster_regions

    class _Det:
        def detect(self, image):
            del image
            return []

    class _CropVerifyOcr:
        def recognize(self, image, regions=None):
            del regions
            w, h = float(image.width), float(image.height)
            return [OcrLine("CROPPED REGION", (0.0, 0.0, w, h), 0.95)]

    with tempfile.TemporaryDirectory() as tmp:
        for rot in (0, 90, 180, 270):
            b = TestPdfBuilder()
            pg = b.new_page(width=600, height=800)
            pg.insert_text((100, 150), "Anchor", fontsize=11)
            raw = b.to_bytes().replace(
                b"/MediaBox [0 0 600.00 800.00]",
                f"/MediaBox [0 0 600.00 800.00] /CropBox [40 60 540 760] /Rotate {rot}".encode("ascii"),
            )
            pdf_path = Path(tmp) / f"crop_{rot}.pdf"
            pdf_path.write_bytes(raw)
            with pdfplumber.open(str(pdf_path)) as doc:
                pb = page_bbox(doc.pages[0])
            region = (pb[0] + 60.0, pb[1] + 80.0, pb[0] + 260.0, pb[1] + 220.0)
            reader = ComposedReader(_Det(), _CropVerifyOcr())
            for pboxes in (None, {1: pb}):
                res = read_raster_regions(pdf_path, {1: [region]}, reader=reader, dpi=275, page_boxes=pboxes)
                assert len(res[1]) == 1
                blk = res[1][0]
                assert blk.source == "ocr_high_dpi"
                for got, exp in zip(blk.bbox, region):
                    assert abs(got - exp) <= 0.5, (rot, pboxes, blk.bbox, region)


def test_orphan_merge_blocks_headings_lists_and_narrow_gutters():
    from pdf.backends.reader_composed import _cluster_orphan_lines

    orphans = [
        OcrLine("1.2 Section Heading", (40.0, 50.0, 220.0, 64.0), 0.95),
        OcrLine("Body paragraph line.", (40.0, 70.0, 240.0, 81.5), 0.94),
        OcrLine("Second body line.", (40.0, 85.0, 230.0, 96.5), 0.94),
        OcrLine("- Bullet item one", (40.0, 100.0, 190.0, 111.5), 0.93),
        OcrLine("Left Col", (40.0, 140.0, 120.0, 152.0), 0.95),
        OcrLine("Right Col", (136.0, 140.0, 220.0, 152.0), 0.95),
    ]
    blocks = _cluster_orphan_lines(orphans, 1.0, 1.0)
    assert [b.text for b in blocks] == [
        "1.2 Section Heading",
        "Body paragraph line. Second body line.",
        "- Bullet item one",
        "Left Col",
        "Right Col",
    ], [b.text for b in blocks]


def test_orphan_inline_math_fragments_merge_without_crossing_gutters():
    from pdf.backends.reader_composed import _cluster_orphan_lines

    orphans = [
        OcrLine("A =", (120.0, 200.0, 160.0, 230.0), 0.88),
        OcrLine("f(x)", (173.0, 200.0, 205.0, 219.0), 0.94),
        OcrLine("dx", (172.0, 212.0, 212.0, 236.0), 0.91),
        OcrLine("+ B", (218.0, 202.0, 260.0, 230.0), 0.90),
        OcrLine("Col L", (120.0, 300.0, 180.0, 312.0), 0.95),
        OcrLine("Col R", (196.0, 300.0, 260.0, 312.0), 0.95),
    ]
    blocks = _cluster_orphan_lines(orphans, 1.0, 1.0)
    assert len(blocks) == 3, [b.text for b in blocks]
    assert "A =" in blocks[0].text and "+ B" in blocks[0].text
    assert [b.text for b in blocks[1:]] == ["Col L", "Col R"]


def test_reading_order_one_to_many_containment_and_no_gt_order_leak():
    from pdf.eval.gt import PT_PER_PX_150DPI
    from pdf.eval.reading_order import _match_elements_to_gt, kendall_tau_matched, kendall_tau_penalized

    scale = 1.0 / PT_PER_PX_150DPI
    gt_stack = [
        {"id": "g0", "order": 0, "bbox": [100.0 * scale, 50.0 * scale, 450.0 * scale, 80.0 * scale]},
        {"id": "g1", "order": 1, "bbox": [120.0 * scale, 100.0 * scale, 420.0 * scale, 130.0 * scale]},
        {"id": "g2", "order": 2, "bbox": [120.0 * scale, 138.0 * scale, 420.0 * scale, 168.0 * scale]},
        {"id": "g3", "order": 3, "bbox": [120.0 * scale, 176.0 * scale, 420.0 * scale, 206.0 * scale]},
    ]
    pred_stack = [
        {"type": "text", "level": 1, "bbox": [100.0, 50.0, 450.0, 80.0], "text": "Intro"},
        {"type": "text", "level": 1, "bbox": [115.0, 98.0, 425.0, 208.0], "text": "Eq1 Eq2 Eq3"},
    ]
    matched = _match_elements_to_gt(pred_stack, gt_stack)
    assert matched == ["g0", "g1", "g2", "g3"], matched
    assert kendall_tau_penalized(matched, ["g0", "g1", "g2", "g3"]) == 1.0

    # A single box spanning two columns must sort covered GTs geometrically (y, x), not by gt["order"].
    gt_two_col = [
        {"id": "l1", "order": 0, "bbox": [40.0 * scale, 100.0 * scale, 200.0 * scale, 130.0 * scale]},
        {"id": "l2", "order": 1, "bbox": [40.0 * scale, 150.0 * scale, 200.0 * scale, 180.0 * scale]},
        {"id": "r1", "order": 2, "bbox": [300.0 * scale, 100.0 * scale, 460.0 * scale, 130.0 * scale]},
        {"id": "r2", "order": 3, "bbox": [300.0 * scale, 150.0 * scale, 460.0 * scale, 180.0 * scale]},
    ]
    giant_box = [{"type": "text", "level": 1, "bbox": [35.0, 95.0, 465.0, 185.0], "text": "merged"}]
    giant_matched = _match_elements_to_gt(giant_box, gt_two_col)
    assert giant_matched == ["l1", "r1", "l2", "r2"], giant_matched
    assert kendall_tau_matched(giant_matched, ["l1", "l2", "r1", "r2"]) < 1.0


def test_rapidocr_lang_option_and_ir_forwarding():
    from pdf.backends import get as get_backend, reset_cache
    from pdf.backends.rapidocr import RapidOcrRecognizer
    from pdf.models import ExtractedDiagram, ExtractedTable, TableCell

    rec_default = RapidOcrRecognizer()
    rec_ja = RapidOcrRecognizer(lang="ja")
    assert rec_default.lang is None
    assert rec_ja.lang == "japan"
    assert rec_ja.ensure_loaded() is True
    assert str(rec_ja._reader.cfg.Rec.lang_type) == "japan"

    reset_cache()
    try:
        cached_default = get_backend("ocr", "rapidocr-onnx")
        cached_ja = get_backend("ocr", "rapidocr-onnx", lang="ja")
        assert cached_default is not cached_ja
        assert cached_ja.lang == "japan"

        seen_reader_langs: list[str | None] = []
        seen_table_langs: list[str | None] = []

        class _Det:
            def detect(self, image):
                w, h = float(image.width), float(image.height)
                return [
                    LayoutItem("text", (10.0, 10.0, w * 0.4, h * 0.2), 0.9),
                    LayoutItem("table", (10.0, h * 0.25, w * 0.8, h * 0.55), 0.9),
                    LayoutItem("picture", (10.0, h * 0.6, w * 0.8, h * 0.9), 0.9),
                ]

        class _Ocr:
            def __init__(self, lang=None):
                seen_reader_langs.append(lang)

            def recognize(self, image, regions=None):
                del image, regions
                return [OcrLine("日本語テスト", (12.0, 12.0, 120.0, 32.0), 0.95)]

        class _TblWithLang:
            last_stats = None

            def extract_from_page(self, _page, table_boxes_pt, *, ocr_lang=None, **_kw):
                seen_table_langs.append(ocr_lang)
                return [
                    ExtractedTable(
                        bbox=table_boxes_pt[0],
                        num_rows=1,
                        num_cols=1,
                        cells=(TableCell(0, 0, 1, 1, "表", None, False),),
                        source="auto_tatr",
                        confidence=0.9,
                    )
                ]

        class _LegacyDiag:
            last_stats = None

            def extract_from_page(self, _page, diagram_boxes_pt, *, words=None, is_raster=False, ocr_backend="rapidocr-onnx", device=None, dpi=150):
                del words, is_raster, ocr_backend, device, dpi
                return [ExtractedDiagram(bbox=diagram_boxes_pt[0], diagram_type="block_diagram")]

        register("detector", "fake_lang_det", lambda **_kw: _Det())
        register("ocr", "fake_lang_ocr", lambda **kw: _Ocr(lang=kw.get("lang")))

        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "ja_scan.pdf"
            _make_raster_pdf(pdf_path, text="JP")
            ir = extract_ir(
                pdf_path,
                figures=False,
                ocr=True,
                tables=True,
                diagrams=True,
                layout="fake_lang_det",
                ocr_backend="fake_lang_ocr",
                ocr_lang="ja",
                table_backend=_TblWithLang(),
                diagram_backend=_LegacyDiag(),
            )
            assert seen_reader_langs == ["ja"], seen_reader_langs
            assert seen_table_langs == ["ja"], seen_table_langs
            assert any(e["text"] == "日本語テスト" for e in ir["elements"])
    finally:
        reset_cache()


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
