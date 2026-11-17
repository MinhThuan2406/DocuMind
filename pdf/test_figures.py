import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pymupdf  # noqa: E402
from PIL import Image  # noqa: E402

from pdf import extract_ir  # noqa: E402
from pdf.backends import LayoutCache, PageLayout, RunStats, file_sha256, register, registered  # noqa: E402
from pdf.figures import LAYOUT_BOX_CONFIDENCE, detect_figures  # noqa: E402
from pdf.layout import (  # noqa: E402
    MINERU_PIPELINE_BACKEND,
    MINERU_PIPELINE_REVISION,
    mineru_middle_json_to_cache_pages,
)
from pdf.layout.mineru_pipeline import find_mineru_python  # noqa: E402


def test_importing_figures_does_not_load_docling_or_torch():
    assert "docling" not in sys.modules
    assert "torch" not in sys.modules


def test_cache_and_live_confidence_match_and_table_boxes_emitted():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pdf_path = tmp_path / "sample.pdf"
        doc = pymupdf.open()
        doc.new_page(width=600, height=400)
        doc.save(str(pdf_path))
        doc.close()

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

    doc = pymupdf.open(str(layered))
    zoom = 150.0 / 72.0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for pno in dev_pages:
            page = doc[pno - 1]
            png = tmp_dir / f"p{pno:03d}.png"
            page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False).save(str(png))
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
            sx = page.rect.width / img_size[0]
            sy = page.rect.height / img_size[1]
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
        doc = pymupdf.open()
        doc.new_page(width=600, height=400)
        doc.save(str(pdf_path))
        doc.close()

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
        doc = pymupdf.open()
        p1 = doc.new_page(width=600, height=400)
        p1.insert_text((40, 40), "Page one text", fontsize=12)
        p2 = doc.new_page(width=600, height=400)
        img = Image.new("RGB", (300, 200), (220, 230, 240))
        buf = tmp_path / "box.png"
        img.save(buf)
        p2.insert_image(pymupdf.Rect(100, 80, 400, 280), filename=str(buf))
        doc.save(str(pdf_path))
        doc.close()

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
