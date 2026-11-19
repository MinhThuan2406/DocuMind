import io
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pdfplumber  # noqa: E402
import pypdfium2 as pdfium  # noqa: E402
from PIL import Image  # noqa: E402

from pdf._pdf_io import (  # noqa: E402
    extract_words_raw,
    normalize_page_bbox,
    normalize_point_or_box,
    page_bbox,
    page_box_to_pdfium_box,
    pdfium_box_to_page_box,
    pdfium_page_box,
    pt_to_px,
    px_to_pt,
    render_bgr,
)
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
    assert "invalid_char_ratio" in prof1.suspect_reasons

    b = TestPdfBuilder()
    p2 = b.new_page(width=595, height=842)
    p2.insert_image((0, 0, 595, 842), stream=_png_bytes(595, 842))
    p2.insert_text((72, 72), "hidden tiny ocr text words", fontsize=2)
    prof2 = _classify_builder(b)
    assert prof2.suspect_ocr_layer and prof2.kind == "hybrid"
    assert "suspect_ocr_image_cover" in prof2.suspect_reasons


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


def test_cid_and_pua_suspect_ocr_layer():
    cid_page = _FakePage([
        (72.0, 60.0, 110.0, 74.0, "(cid:12)"),
        (114.0, 60.0, 152.0, 74.0, "(cid:13)"),
        (156.0, 60.0, 180.0, 74.0, "bad"),
        (184.0, 60.0, 210.0, 74.0, "text"),
        (214.0, 60.0, 245.0, 74.0, "layer"),
    ])
    prof_cid = classify_page(cid_page)
    assert prof_cid.suspect_ocr_layer and prof_cid.kind == "hybrid"
    assert "cid_or_pua_glyphs" in prof_cid.suspect_reasons

    pua_page = _FakePage([
        (72.0, 60.0, 110.0, 74.0, "\ue001\ue002\ue003"),
        (114.0, 60.0, 155.0, 74.0, "garbled"),
        (159.0, 60.0, 185.0, 74.0, "ocr"),
        (189.0, 60.0, 215.0, 74.0, "text"),
        (219.0, 60.0, 250.0, 74.0, "page"),
    ])
    prof_pua = classify_page(pua_page)
    assert prof_pua.suspect_ocr_layer and prof_pua.kind == "hybrid"
    assert "cid_or_pua_glyphs" in prof_pua.suspect_reasons

    wingdings = _FakePage([])
    wingdings._words = [
        {"x0": 72.0, "top": 60.0, "x1": 82.0, "bottom": 72.0, "text": "\uf0fb", "fontname": "BCDFEE+Wingdings-Regular"},
        {"x0": 90.0, "top": 60.0, "x1": 140.0, "bottom": 72.0, "text": "Supported", "fontname": "ArialMT"},
        {"x0": 145.0, "top": 60.0, "x1": 185.0, "bottom": 72.0, "text": "feature", "fontname": "ArialMT"},
        {"x0": 190.0, "top": 60.0, "x1": 225.0, "bottom": 72.0, "text": "table", "fontname": "ArialMT"},
        {"x0": 230.0, "top": 60.0, "x1": 270.0, "bottom": 72.0, "text": "entry", "fontname": "ArialMT"},
    ]
    prof_wd = classify_page(wingdings)
    assert not prof_wd.suspect_ocr_layer
    assert prof_wd.kind == "native"


def test_watermark_exclusion_and_count():
    b = TestPdfBuilder()
    wm_only = b.new_page(width=595, height=842)
    for idx in range(8):
        wm_only.insert_text((80 + idx * 45, 100 + idx * 50), "CONFIDENTIAL", fontsize=3.0)

    native_with_wm = b.new_page(width=595, height=842)
    native_with_wm.insert_text((72, 72), "Real body paragraph with enough native text.\n" * 3, fontsize=12.0)
    for idx in range(4):
        native_with_wm.insert_text((100 + idx * 60, 400), "DRAFT", fontsize=3.5)

    with pdfplumber.open(io.BytesIO(b.to_bytes())) as doc:
        prof_wm = classify_page(doc.pages[0])
        assert prof_wm.watermark_word_count == 8
        assert prof_wm.word_count == 0
        assert prof_wm.text_area_ratio == 0.0
        assert prof_wm.kind == "native"

        prof_nat = classify_page(doc.pages[1])
        assert prof_nat.watermark_word_count == 4
        assert prof_nat.word_count >= 18
        assert prof_nat.kind == "native"


def test_partial_raster_regions_35pct_image():
    b = TestPdfBuilder()
    p_unocr = b.new_page(width=595, height=842)
    p_unocr.insert_text((72, 72), "Native header paragraph above an un-OCR raster diagram.\n" * 2, fontsize=12.0)
    p_unocr.insert_image((60, 220, 480, 640), stream=_png_bytes(420, 420))

    p_covered = b.new_page(width=595, height=842)
    p_covered.insert_image((60, 220, 480, 640), stream=_png_bytes(420, 420))
    p_covered.insert_text((80, 260), "Native text rendered cleanly inside the diagram box.\n" * 6, fontsize=12.0)

    with pdfplumber.open(io.BytesIO(b.to_bytes())) as doc:
        prof_unocr = classify_page(doc.pages[0])
        assert prof_unocr.kind == "hybrid"
        assert not prof_unocr.suspect_ocr_layer
        assert len(prof_unocr.partial_raster_regions) == 1
        assert prof_unocr.partial_raster_regions[0] == (60.0, 220.0, 480.0, 640.0)

        prof_cov = classify_page(doc.pages[1])
        assert prof_cov.partial_raster_regions == ()
        assert prof_cov.kind == "native"


def test_slide_with_real_text_over_background_image():
    b = TestPdfBuilder()
    slide = b.new_page(width=595, height=842)
    slide.insert_image((0, 0, 595, 842), stream=_png_bytes(595, 842))
    slide.insert_text((60, 80), "Quarterly Architecture Review Slide Title", fontsize=18.0, bold=True)
    slide.insert_text(
        (60, 140),
        "Bullet one describing real foreground native text on slide.\n"
        "Bullet two with substantial readable content area coverage.\n"
        "Bullet three ensuring text area ratio exceeds suspect threshold.",
        fontsize=14.0,
    )
    prof = _classify_builder(b)
    assert prof.kind == "hybrid"
    assert not prof.suspect_ocr_layer
    assert prof.suspect_reasons == ()
    assert prof.partial_raster_regions == ()


def test_zero_area_page_safe():
    for w, h in ((0.0, 0.0), (595.0, 0.0), (-10.0, 100.0)):
        prof = classify_page(_FakePage([], width=w, height=h), page_number=3)
        assert prof.page == 3
        assert prof.kind == "native"
        assert prof.word_count == 0
        assert not prof.suspect_ocr_layer
        assert prof.suspect_reasons == ()
        assert prof.partial_raster_regions == ()
        assert prof.watermark_word_count == 0


def _make_rotated_cropped_pdf(rotate: int, crop: tuple[int, int, int, int]) -> bytes:
    stream = (
        b"q 1 w 0 0 0 RG 0 0 0 rg 100 350 60 60 re B Q\n"
        b"BT /F1 10 Tf 1 1 1 rg 105 375 Tm (Hello) Tj ET"
    )
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 600] "
            f"/CropBox [{crop[0]} {crop[1]} {crop[2]} {crop[3]}] /Rotate {rotate} "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents 5 0 R >>"
        ).encode("ascii"),
        f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream",
    ]
    buf = bytearray(b"%PDF-1.4\n")
    offs = [0]
    for idx, body in enumerate(objs, start=1):
        offs.append(len(buf))
        buf.extend(f"{idx} 0 obj\n".encode("ascii") + body + b"\nendobj\n")
    xref = len(buf)
    buf.extend(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for off in offs[1:]:
        buf.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    buf.extend(f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii"))
    return bytes(buf)


def test_rotate_and_cropbox_coordinate_roundtrip():
    crop = (30, 70, 350, 510)
    for rot in (0, 90, 180, 270):
        data = _make_rotated_cropped_pdf(rot, crop)
        with pdfplumber.open(io.BytesIO(data)) as doc:
            page = doc.pages[0]
            words = extract_words_raw(page, with_style=True)
            assert words and all("upright" in w for w in words), (rot, words)
            rect = page.rects[0]
            rect_pl = (float(rect["x0"]), float(rect["top"]), float(rect["x1"]), float(rect["bottom"]))
            norm_bounds = normalize_page_bbox(page)
            norm_rect = normalize_point_or_box(rect_pl, page)

        fdoc = pdfium.PdfDocument(data)
        try:
            fpage = fdoc[0]
            fw, fh = fpage.get_size()
            assert max(abs(norm_bounds[2] - fw), abs(norm_bounds[3] - fh)) <= 0.01
            pbox_pl = page_bbox(page)
            pbox_fi = pdfium_page_box(fpage)
            assert max(abs(a - b) for a, b in zip(pbox_pl, pbox_fi)) <= 0.01, (rot, pbox_pl, pbox_fi)
            fi_rect = page_box_to_pdfium_box(rect_pl, pbox_pl)
            assert max(abs(a - b) for a, b in zip(fi_rect, norm_rect)) <= 0.01
            assert max(abs(a - b) for a, b in zip(pdfium_box_to_page_box(fi_rect, pbox_fi), rect_pl)) <= 0.01
            bgr, img_size = render_bgr(fpage, dpi=150)
        finally:
            fdoc.close()

        px_box = pt_to_px(norm_rect, norm_bounds, img_size)
        snapped_px = tuple(float(round(v)) for v in px_box)
        gray = np.min(bgr, axis=2)
        ys, xs = np.where(gray < 128)
        dark_px = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
        assert all(abs(a - b) <= 2.5 for a, b in zip(snapped_px, dark_px)), (rot, snapped_px, dark_px)

        roundtrip_norm = px_to_pt(snapped_px, norm_bounds, img_size)
        with pdfplumber.open(io.BytesIO(data)) as doc:
            roundtrip_pl = normalize_point_or_box(roundtrip_norm, doc.pages[0], inverse=True)
            pt_err = max(abs(a - b) for a, b in zip(roundtrip_pl, rect_pl))
            assert pt_err <= 0.5, (rot, pt_err, roundtrip_pl, rect_pl)

            pt_sample = (rect_pl[0], rect_pl[1])
            norm_pt = normalize_point_or_box(pt_sample, doc.pages[0])
            px_pt = tuple(float(round(v)) for v in pt_to_px(norm_pt, norm_bounds, img_size))
            rt_pt = normalize_point_or_box(px_to_pt(px_pt, norm_bounds, img_size), doc.pages[0], inverse=True)
            assert max(abs(a - b) for a, b in zip(rt_pt, pt_sample)) <= 0.5


def test_watermark_diagram_micro_labels_and_corner_logo():
    b = TestPdfBuilder()
    diag_page = b.new_page(width=595, height=842)
    diag_page.insert_text((72, 72), "Main body paragraph above a vector diagram.\n" * 2, fontsize=12.0)
    for idx in range(5):
        diag_page.insert_text((100 + idx * 50, 300), "a", fontsize=3.0)
        diag_page.insert_text((100 + idx * 50, 320), "uppermultiplicity", fontsize=3.5)

    logo_page = b.new_page(width=595, height=842)
    logo_page.insert_text((72, 160), "Native specification page with a small header banner.\n" * 4, fontsize=12.0)
    logo_page.insert_image((40, 20, 440, 95), stream=_png_bytes(400, 75))  # ~6% area in top header band

    with pdfplumber.open(io.BytesIO(b.to_bytes())) as doc:
        prof_diag = classify_page(doc.pages[0])
        assert prof_diag.watermark_word_count == 0
        assert prof_diag.word_count >= 18

        prof_logo = classify_page(doc.pages[1])
        assert prof_logo.partial_raster_regions == ()
        assert prof_logo.kind == "native"


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
