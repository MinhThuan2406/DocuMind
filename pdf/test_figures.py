import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pymupdf  # noqa: E402
from PIL import Image  # noqa: E402

import pdf.figures as figures_mod  # noqa: E402
from pdf.figures import LAYOUT_BOX_CONFIDENCE, detect_figures  # noqa: E402
from pdf.layout import LayoutCache, file_sha256  # noqa: E402


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

        orig_runner = figures_mod._run_layout_items
        orig_get = figures_mod.get_layout_detector
        try:
            figures_mod._run_layout_items = lambda _det, _bgr: (items, (600.0, 400.0))
            figures_mod.get_layout_detector = lambda _backend, _device: (object(), "testrev")

            from_cache = detect_figures(pdf_path, [1], layout="custom", cache_dir=cache_root)
            from_live = detect_figures(pdf_path, [1], layout="custom", cache_dir=tmp_path / "live_cache")
        finally:
            figures_mod._run_layout_items = orig_runner
            figures_mod.get_layout_detector = orig_get

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
    from run_pdf_figure_crops import compute_boxes  # noqa: E402

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
