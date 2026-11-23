import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pypdfium2 as pdfium  # noqa: E402
from PIL import Image  # noqa: E402

from pdf import extract_ir  # noqa: E402
from pdf._test_pdf import TestPdfBuilder  # noqa: E402
from pdf.backends import LayoutCache, PageLayout, RunStats, file_sha256, register, registered  # noqa: E402
from pdf.backends.mineru import (  # noqa: E402
    MINERU_PIPELINE_BACKEND,
    MINERU_PIPELINE_REVISION,
    find_mineru_python,
    mineru_middle_json_to_cache_pages,
)
from pdf.figures import LAYOUT_BOX_CONFIDENCE, detect_figures  # noqa: E402
from pdf.models import SemanticRegion  # noqa: E402


def test_importing_figures_does_not_load_docling_or_torch():
    assert "docling" not in sys.modules
    assert "torch" not in sys.modules


def test_cache_and_live_confidence_match_and_table_boxes_emitted():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pdf_path = tmp_path / "sample.pdf"
        b = TestPdfBuilder()
        b.new_page(width=600, height=400)
        b.save(pdf_path)

        cache_root = tmp_path / "cache"
        items = [
            {"order": 0, "label": "picture", "bbox": [50.0, 50.0, 300.0, 250.0], "text": ""},
            {"order": 1, "label": "table", "bbox": [350.0, 50.0, 550.0, 250.0], "text": ""},
        ]
        LayoutCache(
            cache_root,
            backend="custom",
            revision="testrev",
            dpi=150,
            pdf_hash=file_sha256(pdf_path),
        ).save(1, items, (600.0, 400.0))

        class _CustomBackend:
            name = "custom"
            revision = "testrev"
            cache_dpi = 150
            last_stats: RunStats | None = None

            def analyze(self, _pdf_path, pages):
                self.last_stats = RunStats(infer_sec=0.01, pages=len(pages), in_process=True)
                return {p: PageLayout(items=items, frame=(600.0, 400.0)) for p in pages}

        register("layout", "custom", lambda **_kw: _CustomBackend())
        assert "custom" in registered("layout")

        from_cache = detect_figures(pdf_path, [1], layout="custom", cache_dir=cache_root)
        from_live = detect_figures(pdf_path, [1], layout="custom", cache_dir=tmp_path / "live_cache")

        assert from_cache == from_live
        assert len(from_cache[1]) == 2
        assert [b.kind for b in from_cache[1]] == ["figure", "table"]
        assert from_cache[1][1].source == "layout_table"
        assert all(b.confidence == LAYOUT_BOX_CONFIDENCE for b in from_cache[1])

        raw_only = detect_figures(
            pdf_path, [1], layout="custom", grouper="raw_picture", cache_dir=cache_root, include_tables=False
        )
        assert len(raw_only[1]) == 1
        assert raw_only[1][0].source == "layout_picture"
        assert "raw_picture" in registered("grouper") and "v1_anchor" in registered("grouper")


def test_matches_structure_branch_compute_boxes_on_dev_pages():
    layered = ROOT / "datasets" / "AUTOSAR" / "AUTOSAR_CP_EXP_LayeredSoftwareArchitecture.pdf"
    docling_cache = ROOT / "outputs" / "08_pdf_figure_crops" / layered.stem / "_docling_cache_dpi150"
    scratch_scripts = (
        Path.home()
        / ".gemini"
        / "antigravity"
        / "brain"
        / "369e81e0-055e-40f7-a048-69fa3d991a41"
        / "scratch"
        / "structure_scripts"
        / "scripts"
    )
    if not layered.exists() or not docling_cache.exists() or not scratch_scripts.exists():
        print("SKIP AUTOSAR PDF, reference cache, or scratch structure scripts not present")
        return

    dev_pages = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45]
    # Run our Heron-based detector BEFORE importing structure scripts (which pull docling).
    with tempfile.TemporaryDirectory() as tmp:
        actual = detect_figures(layered, dev_pages, cache_dir=Path(tmp) / "fresh", include_tables=False)
    assert "docling" not in sys.modules

    sys.path.insert(0, str(scratch_scripts))
    try:
        from run_pdf_figure_crops import compute_boxes  # noqa: E402
    except ImportError:
        print("SKIP structure scripts require docling (not installed)")
        return

    doc = pdfium.PdfDocument(str(layered))
    zoom = 150.0 / 72.0
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for pno in dev_pages:
                page = doc[pno - 1]
                pw, ph = page.get_size()
                png = tmp_dir / f"p{pno:03d}.png"
                page.render(scale=zoom).to_pil().save(str(png))
                cached = json.loads((docling_cache / f"p{pno:03d}.json").read_text(encoding="utf-8"))
                with Image.open(png) as im:
                    img_size = im.size
                ref_boxes = compute_boxes(
                    page,
                    png,
                    cached["items"],
                    tuple(cached["page_size"]),
                    img_size,
                    group=True,
                    refine=False,
                    add_missing=False,
                    min_area=0.015,
                    footer_frac=0.88,
                    v1_rule="anchor",
                )
                sx = float(pw) / img_size[0]
                sy = float(ph) / img_size[1]
                expected_pts = [
                    (
                        round(b["bbox"][0] * sx, 3),
                        round(b["bbox"][1] * sy, 3),
                        round(b["bbox"][2] * sx, 3),
                        round(b["bbox"][3] * sy, 3),
                    )
                    for b in ref_boxes
                ]
                got_pts = [b.bbox for b in actual[pno]]
                assert len(got_pts) == len(expected_pts), (pno, got_pts, expected_pts)
                for g, e in zip(got_pts, expected_pts):
                    for gv, ev in zip(g, e):
                        assert abs(gv - ev) <= 0.1, (pno, g, e)
    finally:
        doc.close()


def test_mineru_fixture_mapping_clamping_and_cache_hit():
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "mineru_dumps" / "autosar_p13_stripped.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert all("content" not in b for pg in fixture["pages"] for b in pg["blocks"])

    pages_out, unmapped, _post = mineru_middle_json_to_cache_pages(fixture, {13: (800.0, 500.0)})
    assert sum(unmapped.values()) == 0
    items13, psize13 = pages_out[13]
    assert psize13 == (800.0, 500.0)
    assert [it["label"] for it in items13] == ["picture"]

    synthetic = {
        "pages": [
            {
                "page_idx": 0,
                "blocks": [
                    {"type": "image", "bbox": [-0.1, 0.1, 0.5, 0.5], "score": 1.0},
                    {"type": "chart", "bbox": [0.6, 0.1, 1.1, 0.5], "score": 1.0},
                    {"type": "table", "bbox": [0.1, 0.55, 0.8, 0.85], "score": 1.0},
                    {"type": "equation", "bbox": [0.1, 0.86, 0.4, 0.92], "score": 1.0},
                    {"type": "inline_equation", "bbox": [0.1, 0.05, 0.2, 0.08], "score": 1.0},
                    {"type": "image", "bbox": [0.0, 0.0, 0.98, 0.98], "score": 1.0},
                    {"type": "unknown_block_type", "bbox": [0.2, 0.2, 0.3, 0.3], "score": 1.0},
                ],
            }
        ]
    }
    syn_out, syn_unmapped, syn_post = mineru_middle_json_to_cache_pages(synthetic, {1: (600.0, 400.0)})
    assert dict(syn_unmapped) == {"unknown_block_type": 1}
    assert syn_post[1][0] >= 1
    labels = [it["label"] for it in syn_out[1][0]]
    assert "formula" in labels and "table" in labels and labels.count("picture") == 2
    for it in syn_out[1][0]:
        x0, y0, x1, y1 = it["bbox"]
        assert 0.0 <= x0 < x1 <= 600.0 and 0.0 <= y0 < y1 <= 400.0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pdf_path = tmp_path / "mineru_sample.pdf"
        b = TestPdfBuilder()
        b.new_page(width=600, height=400)
        b.save(pdf_path)

        cache_root = tmp_path / "cache"
        LayoutCache(
            cache_root,
            backend=MINERU_PIPELINE_BACKEND,
            revision=MINERU_PIPELINE_REVISION,
            dpi=0,
            pdf_hash=file_sha256(pdf_path),
        ).save(1, syn_out[1][0], (600.0, 400.0))

        import pdf.backends.mineru as mineru_backend_mod

        orig_run = mineru_backend_mod.run_mineru_pdf
        try:
            mineru_backend_mod.run_mineru_pdf = lambda *_a, **_kw: (_ for _ in ()).throw(
                AssertionError("should hit cache")
            )
            out = detect_figures(pdf_path, [1], layout=MINERU_PIPELINE_BACKEND, cache_dir=cache_root)
        finally:
            mineru_backend_mod.run_mineru_pdf = orig_run
        assert any(b.kind == "figure" for b in out[1])
        assert any(b.kind == "table" for b in out[1])

        had_torch = "torch" in sys.modules
        ir_plain = extract_ir(pdf_path, figures=False, ocr=False)
        assert ir_plain["doc_type"] == "pdf"
        if not had_torch:
            assert "torch" not in sys.modules
        assert "mineru" not in sys.modules


def test_mineru_subprocess_integration_optional():
    try:
        find_mineru_python()
    except FileNotFoundError:
        print("SKIP MinerU venv not present")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pdf_path = tmp_path / "two_pages.pdf"
        b = TestPdfBuilder()
        p1 = b.new_page(width=600, height=400)
        p1.insert_text((40, 40), "Page one text", fontsize=12)
        p2 = b.new_page(width=600, height=400)
        img = Image.new("RGB", (300, 200), (220, 230, 240))
        buf = tmp_path / "box.png"
        img.save(buf)
        p2.insert_image((100, 80, 400, 280), filename=str(buf))
        b.save(pdf_path)

        cache_root = tmp_path / "cache"
        had_torch = "torch" in sys.modules
        res = detect_figures(pdf_path, [2], layout=MINERU_PIPELINE_BACKEND, cache_dir=cache_root)
        assert 2 in res
        assert "mineru" not in sys.modules
        if not had_torch:
            assert "torch" not in sys.modules
        p1_cache = LayoutCache(
            cache_root,
            backend=MINERU_PIPELINE_BACKEND,
            revision=MINERU_PIPELINE_REVISION,
            dpi=0,
            pdf_hash=file_sha256(pdf_path),
        ).load(1)
        assert p1_cache is not None


def test_grouped_figure_real_scores_and_legacy_fallback():
    import numpy as np

    from pdf.grouping import RawPictureGrouper, V1AnchorGrouper
    from pdf.layout import LayoutItem, clusters_to_items, postprocess_layout
    from pdf.models import GroupedFigure

    clusters = postprocess_layout(
        [
            LayoutItem("picture", (40.0, 40.0, 240.0, 200.0), 0.81),
            LayoutItem("table", (300.0, 40.0, 560.0, 200.0), 0.77),
        ],
        (600.0, 400.0),
        keep_empty_clusters=False,
    )
    items = clusters_to_items(clusters)
    assert [it["score"] for it in items] == [0.81, 0.77]

    bgr = np.full((400, 600, 3), 255, dtype=np.uint8)
    layout = PageLayout(
        items=[
            {"order": 0, "label": "picture", "bbox": [40.0, 40.0, 240.0, 140.0], "text": "", "score": 0.80},
            {"order": 1, "label": "formula", "bbox": [40.0, 145.0, 140.0, 165.0], "text": "", "score": 0.91},
            {"order": 2, "label": "table", "bbox": [300.0, 40.0, 560.0, 200.0], "text": "", "score": 0.73},
        ],
        frame=(600.0, 400.0),
    )
    grouped_v1 = V1AnchorGrouper().group(bgr, layout)
    assert len(grouped_v1) == 1 and isinstance(grouped_v1[0], GroupedFigure)
    assert grouped_v1[0].source == "v1_group"
    expected_wmean = round((20000.0 * 0.80 + 2000.0 * 0.91) / 22000.0, 4)
    assert grouped_v1[0].score == expected_wmean

    grouped_raw = RawPictureGrouper().group(bgr, layout)
    assert len(grouped_raw) == 1 and grouped_raw[0].score == 0.80

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pdf_path = tmp_path / "scored.pdf"
        b = TestPdfBuilder()
        b.new_page(width=600, height=400)
        b.save(pdf_path)

        class _ScoredBackend:
            name = "scored_mock"
            revision = "r1"
            cache_dpi = 150
            last_stats: RunStats | None = None

            def analyze(self, _pdf_path, pages):
                self.last_stats = RunStats(infer_sec=0.01, pages=len(pages), in_process=True)
                return {p: layout for p in pages}

        register("layout", "scored_mock", lambda **_kw: _ScoredBackend())
        detected = detect_figures(pdf_path, [1], layout="scored_mock", cache_dir=tmp_path / "cache")
        assert [round(f.confidence, 4) for f in detected[1]] == [expected_wmean, 0.73]


def test_extract_ir_single_pdf_open_and_no_double_render():
    import io

    import pdfplumber
    from PIL import ImageDraw

    import pdf._pdf_io as pdf_io_mod
    import pdf.backends.layout_cache as cache_mod
    import pdf.backends.layout_image as layout_image_mod
    import pdf.figures as figures_mod
    from pdf.layout.types import LayoutItem
    from pdf.ocr.types import OcrLine

    class _Det:
        def detect(self, image):
            w, h = float(image.width), float(image.height)
            return [
                LayoutItem("picture", (0.1 * w, 0.1 * h, 0.5 * w, 0.5 * h), 0.84),
                LayoutItem("text", (0.1 * w, 0.6 * h, 0.6 * w, 0.8 * h), 0.90),
            ]

    class _Ocr:
        def recognize(self, image, regions=None):
            del regions
            w, h = float(image.width), float(image.height)
            return [OcrLine("Scanned line", (0.1 * w, 0.6 * h, 0.5 * w, 0.75 * h), 0.91)]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pdf_path = tmp_path / "two_kinds.pdf"
        b = TestPdfBuilder()
        p1 = b.new_page(width=600, height=400)
        p1.insert_text((50, 40), "Candidate native page", fontsize=12)
        for i in range(8):
            p1.draw_rect((60 + i * 30, 80, 80 + i * 30, 100))

        img = Image.new("RGB", (600, 400), color=(255, 255, 255))
        ImageDraw.Draw(img).text((40, 80), "RASTER", fill=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        p2 = b.new_page(width=600, height=400)
        p2.insert_image((0, 0, 600, 400), stream=buf.getvalue())
        b.save(pdf_path)

        from pdf.backends import ComposedReader, ImageLayoutBackend

        layout_backend = ImageLayoutBackend(_Det(), name="mock_img_layout", revision="r1", dpi=150)
        ocr_reader = ComposedReader(_Det(), _Ocr())

        counts = {"plumber": 0, "pdfium": 0, "sha256": 0, "render_bgr": 0}
        orig_plumber = pdfplumber.open
        orig_pdfium = pdfium.PdfDocument
        orig_sha = cache_mod.file_sha256
        orig_render_fig = figures_mod.render_bgr
        orig_render_img = layout_image_mod.render_bgr
        orig_root = figures_mod._REPO_ROOT

        def _wrap_plumber(*a, **kw):
            counts["plumber"] += 1
            return orig_plumber(*a, **kw)

        def _wrap_pdfium(*a, **kw):
            counts["pdfium"] += 1
            return orig_pdfium(*a, **kw)

        def _wrap_sha(*a, **kw):
            counts["sha256"] += 1
            return orig_sha(*a, **kw)

        def _wrap_render(*a, **kw):
            counts["render_bgr"] += 1
            return pdf_io_mod.render_bgr(*a, **kw)

        pdfplumber.open = _wrap_plumber
        pdfium.PdfDocument = _wrap_pdfium
        cache_mod.file_sha256 = _wrap_sha
        figures_mod.render_bgr = _wrap_render
        layout_image_mod.render_bgr = _wrap_render
        figures_mod._REPO_ROOT = tmp_path
        try:
            ir = extract_ir(
                pdf_path,
                figures=True,
                ocr=True,
                tables=False,
                layout=layout_backend,
                reader=ocr_reader,
                structure=True,
            )
            assert counts == {"plumber": 1, "pdfium": 1, "sha256": 1, "render_bgr": 1}, counts
            assert layout_backend.last_rendered == {}
            p1_figs = [e for e in ir["elements"] if e["page"] == 1 and e["type"] == "figure"]
            assert len(p1_figs) == 1 and p1_figs[0]["confidence"] == 0.84

            class _WarnLayout:
                name = "warn_layout_str"
                revision = "w1"
                cache_dpi = 150
                last_stats = None

                def analyze(self, _pdf_path, pages):
                    self.last_stats = RunStats(
                        pages=len(pages),
                        in_process=True,
                        warnings=({"page": 1, "code": "layout_note", "severity": "info", "reason": "test_warn", "fallback": None},),
                    )
                    return {p: PageLayout(items=[], frame=(600.0, 400.0)) for p in pages}

            register("layout", "warn_layout_str", lambda **_kw: _WarnLayout())
            ir_warn = extract_ir(pdf_path, figures=True, ocr=False, layout="warn_layout_str")
            assert any(w.get("code") == "layout_note" for w in ir_warn["metadata"]["warnings"])

            # Verify detect_figures bounds in-memory rendered BGR buffers across batches and aggregates RunStats.
            multi_pdf = tmp_path / "multi_20.pdf"
            mb = TestPdfBuilder()
            for idx in range(20):
                mp = mb.new_page(width=400, height=300)
                mp.insert_text((30, 30), f"Page {idx + 1}", fontsize=10)
            mb.save(multi_pdf)

            max_live_rendered = 0
            orig_analyze = layout_backend.analyze

            def _tracking_analyze(p, pages, **kw):
                nonlocal max_live_rendered
                res = orig_analyze(p, pages, **kw)
                max_live_rendered = max(max_live_rendered, len(layout_backend.last_rendered))
                return res

            layout_backend.analyze = _tracking_analyze
            try:
                figs_20 = figures_mod.detect_figures(
                    multi_pdf, list(range(1, 21)), layout=layout_backend, cache_dir=tmp_path / "multi_cache"
                )
            finally:
                layout_backend.analyze = orig_analyze

            assert len(figs_20) == 20
            assert 1 <= max_live_rendered <= figures_mod.DEFAULT_RENDER_BATCH_SIZE
            assert layout_backend.last_rendered == {}
            assert layout_backend.last_stats is not None and layout_backend.last_stats.pages == 20

            first_stats = layout_backend.last_stats
            figs_20_cached = figures_mod.detect_figures(
                multi_pdf, list(range(1, 21)), layout=layout_backend, cache_dir=tmp_path / "multi_cache"
            )
            assert len(figs_20_cached) == 20
            assert layout_backend.last_stats is not first_stats and layout_backend.last_stats.pages == 0

            bad_cache = cache_mod.LayoutCache(
                tmp_path / "bad_item_cache",
                backend="x",
                revision="1",
                dpi=150,
                pdf_hash="abc",
            )
            bad_p = bad_cache.path_for(1)
            bad_p.parent.mkdir(parents=True, exist_ok=True)
            bad_p.write_text('{"items": [123], "page_size": [400, 300]}', encoding="utf-8")
            assert bad_cache.load(1) is None

            with pdfplumber.open(str(pdf_path)) as doc:
                crop, crop_box = pdf_io_mod.crop_page_rgb_with_bbox(
                    doc.pages[0], (10.0, 10.0, 110.0, 110.0), 150, pad_pt=5.0, max_pixels=10_000
                )
                assert crop.width * crop.height <= 12_000
                assert crop_box == (5.0, 5.0, 115.0, 115.0)
        finally:
            pdfplumber.open = orig_plumber
            pdfium.PdfDocument = orig_pdfium
            cache_mod.file_sha256 = orig_sha
            figures_mod.render_bgr = orig_render_fig
            layout_image_mod.render_bgr = orig_render_img
            figures_mod._REPO_ROOT = orig_root


def test_v2_vector_vision_dpi_invariance():
    from pdf._pdf_io import render_bgr
    from pdf.backends import get as get_backend
    from pdf.grouping import V2VectorVisionGrouper

    assert "v2_vector_vision" in registered("grouper")
    grouper = get_backend("grouper", "v2_vector_vision")
    assert isinstance(grouper, V2VectorVisionGrouper)

    b = TestPdfBuilder()
    page = b.new_page(width=600, height=400)
    page.draw_rect((45, 45, 340, 220))
    page.draw_rect((60, 60, 160, 140))
    page.draw_rect((210, 60, 320, 140))
    page.insert_text((220, 95), "Block", fontsize=11)

    layout = PageLayout(
        items=[
            {"order": 0, "label": "picture", "bbox": [55.0, 55.0, 170.0, 150.0], "text": "", "score": 0.80},
            {"order": 1, "label": "picture", "bbox": [205.0, 55.0, 325.0, 150.0], "text": "", "score": 0.90},
        ],
        frame=(600.0, 400.0),
    )

    doc = pdfium.PdfDocument(b.to_bytes())
    try:
        pt_boxes_by_dpi: dict[int, list[tuple[float, float, float, float]]] = {}
        scores_by_dpi: dict[int, list[float]] = {}
        for dpi in (100, 150, 200):
            bgr, (iw, ih) = render_bgr(doc[0], dpi)
            grouped = grouper.group(bgr, layout)
            assert len(grouped) == 1
            assert grouped[0].source == "v2_vector_vision"
            sx, sy = 600.0 / iw, 400.0 / ih
            gx0, gy0, gx1, gy1 = grouped[0].bbox
            pt_boxes_by_dpi[dpi] = [(gx0 * sx, gy0 * sy, gx1 * sx, gy1 * sy)]
            scores_by_dpi[dpi] = [grouped[0].score]

        assert scores_by_dpi[100] == scores_by_dpi[150] == scores_by_dpi[200]
        base = pt_boxes_by_dpi[150][0]
        for dpi in (100, 200):
            cur = pt_boxes_by_dpi[dpi][0]
            ix = max(0.0, min(base[2], cur[2]) - max(base[0], cur[0]))
            iy = max(0.0, min(base[3], cur[3]) - max(base[1], cur[1]))
            inter = ix * iy
            union = (base[2] - base[0]) * (base[3] - base[1]) + (cur[2] - cur[0]) * (cur[3] - cur[1]) - inter
            assert inter / union >= 0.95, (dpi, base, cur)
    finally:
        doc.close()


def test_v2_vector_vision_table_barrier_and_caption_separator():
    import cv2
    import numpy as np

    from pdf.grouping import V2VectorVisionGrouper

    grouper = V2VectorVisionGrouper()

    # Case 1: Table separated by >= 4 pt must never merge with picture.
    bgr_blank = np.full((800, 1200, 3), 255, dtype=np.uint8)
    layout_gap = PageLayout(
        items=[
            {"order": 0, "label": "picture", "bbox": [40.0, 50.0, 240.0, 200.0], "text": "", "score": 0.85},
            {"order": 1, "label": "table", "bbox": [246.0, 50.0, 460.0, 200.0], "text": "", "score": 0.88},
        ],
        frame=(600.0, 400.0),
    )
    res_gap = grouper.group(bgr_blank, layout_gap)
    assert len(res_gap) == 1
    assert res_gap[0].bbox[2] <= 510.0  # stays left of table (246 * 2 = 492 px)

    # Case 2: Bordered table grid even < 4 pt away must never merge with picture.
    bgr_grid = np.full((800, 1200, 3), 255, dtype=np.uint8)
    for x in (504, 640, 780, 920):
        cv2.line(bgr_grid, (x, 100), (x, 400), (0, 0, 0), 2)
    for y in (100, 200, 300, 400):
        cv2.line(bgr_grid, (504, y), (920, y), (0, 0, 0), 2)
    layout_grid = PageLayout(
        items=[
            {"order": 0, "label": "picture", "bbox": [40.0, 50.0, 250.0, 200.0], "text": "", "score": 0.85},
            {"order": 1, "label": "table", "bbox": [252.0, 50.0, 460.0, 200.0], "text": "", "score": 0.90},
        ],
        frame=(600.0, 400.0),
    )
    res_grid = grouper.group(bgr_grid, layout_grid)
    assert len(res_grid) == 1
    assert res_grid[0].bbox[2] <= 520.0

    # Case 3: Caption Separator (vertical and horizontal) prevents merging adjacent figures.
    layout_cap_v = PageLayout(
        items=[
            {"order": 0, "label": "picture", "bbox": [50.0, 40.0, 260.0, 140.0], "text": "", "score": 0.80},
            {"order": 1, "label": "caption", "bbox": [50.0, 144.0, 220.0, 162.0], "text": "Fig 1", "score": 0.90},
            {"order": 2, "label": "picture", "bbox": [50.0, 166.0, 260.0, 270.0], "text": "", "score": 0.86},
        ],
        frame=(600.0, 400.0),
    )
    res_cap_v = grouper.group(bgr_blank, layout_cap_v)
    assert len(res_cap_v) == 2

    layout_cap_h = PageLayout(
        items=[
            {"order": 0, "label": "picture", "bbox": [40.0, 50.0, 200.0, 200.0], "text": "", "score": 0.80},
            {"order": 1, "label": "caption", "bbox": [204.0, 90.0, 234.0, 140.0], "text": "Fig 2", "score": 0.90},
            {"order": 2, "label": "picture", "bbox": [238.0, 50.0, 400.0, 200.0], "text": "", "score": 0.86},
        ],
        frame=(600.0, 400.0),
    )
    res_cap_h = grouper.group(bgr_blank, layout_cap_h)
    assert len(res_cap_h) == 2

    # Case 4: Merged figures emit area-weighted cluster score.
    layout_merge = PageLayout(
        items=[
            {"order": 0, "label": "picture", "bbox": [50.0, 50.0, 150.0, 150.0], "text": "", "score": 0.80},
            {"order": 1, "label": "picture", "bbox": [160.0, 50.0, 360.0, 150.0], "text": "", "score": 0.92},
        ],
        frame=(600.0, 400.0),
    )
    res_merge = grouper.group(bgr_blank, layout_merge)
    assert len(res_merge) == 1
    expected = round((10000.0 * 0.80 + 20000.0 * 0.92) / 30000.0, 4)
    assert res_merge[0].score == expected


def test_detect_figures_exports_semantic_regions_out():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pdf_path = tmp_path / "semantic.pdf"
        b = TestPdfBuilder()
        b.new_page(width=600, height=400)
        b.save(pdf_path)

        items = [
            {"order": 0, "label": "title", "bbox": [30.0, 15.0, 300.0, 40.0], "text": "", "score": 0.92},
            {"order": 1, "label": "section_header", "bbox": [30.0, 50.0, 240.0, 70.0], "text": "", "score": 0.88},
            {"order": 2, "label": "picture", "bbox": [40.0, 80.0, 260.0, 220.0], "text": "", "score": 0.95},
            {"order": 3, "label": "caption", "bbox": [40.0, 225.0, 220.0, 242.0], "text": "", "score": 0.84},
            {"order": 4, "label": "table", "bbox": [300.0, 80.0, 560.0, 220.0], "text": "", "score": 0.91},
            {"order": 5, "label": "list_item", "bbox": [30.0, 255.0, 280.0, 275.0], "text": "", "score": 0.79},
            {"order": 6, "label": "formula", "bbox": [300.0, 250.0, 480.0, 285.0], "text": "", "score": 0.87},
            {"order": 7, "label": "code", "bbox": [30.0, 290.0, 320.0, 340.0], "text": "", "score": 0.83},
            {"order": 8, "label": "footnote", "bbox": [30.0, 365.0, 300.0, 385.0], "text": "", "score": 0.76},
            {"order": 9, "label": "text", "bbox": [300.0, 290.0, 560.0, 340.0], "text": "", "score": 0.90},
        ]

        class _SemBackend:
            name = "sem_mock"
            revision = "s1"
            cache_dpi = 150
            last_stats: RunStats | None = None

            def analyze(self, _pdf_path, pages):
                self.last_stats = RunStats(infer_sec=0.01, pages=len(pages), in_process=True)
                return {p: PageLayout(items=items, frame=(600.0, 400.0)) for p in pages}

        register("layout", "sem_mock", lambda **_kw: _SemBackend())
        sem_out: dict[int, list[SemanticRegion]] = {}
        figs = detect_figures(
            pdf_path,
            [1],
            layout="sem_mock",
            grouper="v2_vector_vision",
            cache_dir=tmp_path / "cache",
            semantic_regions_out=sem_out,
        )
        assert [f.kind for f in figs[1]] == ["figure", "table"]
        assert 1 in sem_out
        assert all(isinstance(r, SemanticRegion) for r in sem_out[1])
        labels = [r.label for r in sem_out[1]]
        assert set(labels) == {"title", "section_header", "caption", "footnote", "list_item", "formula", "code"}
        assert len(labels) == 7
        title_reg = [r for r in sem_out[1] if r.label == "title"][0]
        assert title_reg.bbox == (30.0, 15.0, 300.0, 40.0)
        assert title_reg.confidence == 0.92


def test_detect_figures_cropbox_and_rotate_coordinates():
    import io

    import pdfplumber
    from pdf._pdf_io import page_bbox, page_box_to_pdfium_box

    crop = (40, 60, 540, 760)
    for rot in (0, 90, 180, 270):
        # Build a 600x800 MediaBox PDF with CropBox [40, 60, 540, 760] and a large 120x120 rect inside CropBox.
        stream = b"q 1 w 0 0 0 RG 0 0 0 rg 140 300 120 120 re B Q"
        objs = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 800] "
                f"/CropBox [{crop[0]} {crop[1]} {crop[2]} {crop[3]}] /Rotate {rot} "
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
        pdf_bytes = bytes(buf)

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as doc:
            pl_page = doc.pages[0]
            pbox = page_bbox(pl_page)
            r = pl_page.rects[0]
            rect_pl = (float(r["x0"]), float(r["top"]), float(r["x1"]), float(r["bottom"]))
        rect_fi = page_box_to_pdfium_box(rect_pl, pbox)
        pw = pbox[2] - pbox[0]
        ph = pbox[3] - pbox[1]

        class _CropBackend:
            name = f"crop_mock_{rot}"
            revision = "c1"
            cache_dpi = 150
            last_stats: RunStats | None = None

            def analyze(self, _pdf_path, pages):
                self.last_stats = RunStats(infer_sec=0.01, pages=len(pages), in_process=True)
                items = [
                    {"order": 0, "label": "picture", "bbox": list(rect_fi), "text": "", "score": 0.91},
                    {
                        "order": 1,
                        "label": "caption",
                        "bbox": [rect_fi[0], rect_fi[3] + 5.0, rect_fi[2], rect_fi[3] + 20.0],
                        "text": "",
                        "score": 0.88,
                    },
                ]
                return {p: PageLayout(items=items, frame=(pw, ph)) for p in pages}

        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / f"rot_{rot}.pdf"
            pdf_path.write_bytes(pdf_bytes)
            backend = _CropBackend()
            sem_auto: dict[int, list[SemanticRegion]] = {}
            figs_auto = detect_figures(
                pdf_path, [1], layout=backend, grouper="raw_picture", cache_dir=Path(tmp) / "c1", semantic_regions_out=sem_auto
            )
            sem_pass: dict[int, list[SemanticRegion]] = {}
            figs_pass = detect_figures(
                pdf_path,
                [1],
                layout=backend,
                grouper="raw_picture",
                cache_dir=Path(tmp) / "c2",
                semantic_regions_out=sem_pass,
                page_boxes={1: pbox},
            )
            for figs, sem in ((figs_auto, sem_auto), (figs_pass, sem_pass)):
                fig_err = max(abs(a - b) for a, b in zip(figs[1][0].bbox, rect_pl))
                assert fig_err <= 0.5, (rot, fig_err, figs[1][0].bbox, rect_pl)
                cap_expected = (rect_pl[0], rect_pl[3] + 5.0, rect_pl[2], rect_pl[3] + 20.0)
                cap_err = max(abs(a - b) for a, b in zip(sem[1][0].bbox, cap_expected))
                assert cap_err <= 0.5, (rot, cap_err, sem[1][0].bbox, cap_expected)


def test_image_layout_backend_keep_empty_clusters_semantic_regions_and_cache_version():
    from pdf.backends import ImageLayoutBackend
    from pdf.layout.types import LayoutItem

    class _Det:
        def __init__(self) -> None:
            self.calls = 0

        def detect(self, image):
            self.calls += 1
            w, h = float(image.width), float(image.height)
            return [
                LayoutItem("title", (0.05 * w, 0.04 * h, 0.50 * w, 0.10 * h), 0.92),
                LayoutItem("section_header", (0.05 * w, 0.12 * h, 0.45 * w, 0.18 * h), 0.88),
                LayoutItem("picture", (0.05 * w, 0.22 * h, 0.50 * w, 0.52 * h), 0.95),
                LayoutItem("caption", (0.05 * w, 0.54 * h, 0.42 * w, 0.60 * h), 0.85),
                LayoutItem("list_item", (0.05 * w, 0.64 * h, 0.50 * w, 0.70 * h), 0.81),
                LayoutItem("formula", (0.55 * w, 0.22 * h, 0.90 * w, 0.30 * h), 0.87),
                LayoutItem("code", (0.55 * w, 0.35 * h, 0.90 * w, 0.48 * h), 0.83),
                LayoutItem("footnote", (0.05 * w, 0.90 * h, 0.55 * w, 0.95 * h), 0.78),
                LayoutItem("text", (0.55 * w, 0.55 * h, 0.90 * w, 0.70 * h), 0.90),
            ]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pdf_path = tmp_path / "real_sem.pdf"
        b = TestPdfBuilder()
        b.new_page(width=600, height=400)
        b.save(pdf_path)

        det = _Det()
        backend = ImageLayoutBackend(det, name="real_sem_img", revision="r1", dpi=150)
        assert backend.keep_empty_clusters is True
        assert backend.postprocess_version == "v3"

        sem_miss: dict[int, list[SemanticRegion]] = {}
        figs_miss = detect_figures(
            pdf_path,
            [1],
            layout=backend,
            grouper="v2_vector_vision",
            cache_dir=tmp_path / "cache",
            semantic_regions_out=sem_miss,
        )
        assert det.calls == 1
        assert len(figs_miss[1]) == 1
        labels_miss = {r.label for r in sem_miss[1]}
        assert {"section_header", "caption", "list_item", "formula", "code", "footnote"} <= labels_miss

        sem_hit: dict[int, list[SemanticRegion]] = {}
        figs_hit = detect_figures(
            pdf_path,
            [1],
            layout=backend,
            grouper="v2_vector_vision",
            cache_dir=tmp_path / "cache",
            semantic_regions_out=sem_hit,
        )
        assert det.calls == 1
        assert figs_hit == figs_miss
        assert sem_hit == sem_miss


def test_groupers_ignore_caption_and_wide_prose_text_over_visual_containers():
    import cv2
    import numpy as np

    from pdf.grouping import V1AnchorGrouper, V2VectorVisionGrouper

    bgr = np.full((800, 600, 3), 255, dtype=np.uint8)
    # Dense lines spanning prose + picture + caption so Canny morphology merges them into one contour.
    for y in range(180, 260, 8):
        cv2.line(bgr, (50, y), (550, y), (0, 0, 0), 2)
    cv2.rectangle(bgr, (180, 275), (420, 460), (0, 0, 0), 2)
    for y in range(475, 505, 8):
        cv2.line(bgr, (50, y), (550, y), (0, 0, 0), 2)

    layout = PageLayout(
        items=[
            {"order": 0, "label": "text", "bbox": [50.0, 175.0, 550.0, 260.0], "text": "", "score": 0.90},
            {"order": 1, "label": "picture", "bbox": [180.0, 275.0, 420.0, 460.0], "text": "", "score": 0.94},
            {"order": 2, "label": "caption", "bbox": [50.0, 472.0, 550.0, 505.0], "text": "", "score": 0.88},
        ],
        frame=(600.0, 800.0),
    )

    for grouper in (V1AnchorGrouper(), V2VectorVisionGrouper()):
        grouped = grouper.group(bgr, layout, min_area=0.015)
        assert len(grouped) == 1
        x0, y0, x1, y1 = grouped[0].bbox
        assert x0 >= 165.0 and x1 <= 435.0, (grouper.name, grouped[0].bbox)
        assert y0 >= 260.0 and y1 <= 470.0, (grouper.name, grouped[0].bbox)


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
