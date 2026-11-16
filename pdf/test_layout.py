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
from pdf.layout import (  # noqa: E402
    LayoutCache,
    LayoutItem,
    clamp_and_rescale,
    clusters_to_items,
    postprocess_layout,
)


def test_postprocess_thresholds_remapping_and_empty_containers():
    raw = [
        LayoutItem("title", (20.0, 10.0, 300.0, 40.0), 0.46),
        LayoutItem("text", (20.0, 50.0, 300.0, 100.0), 0.55),
        LayoutItem("text", (20.0, 110.0, 300.0, 140.0), 0.40),
        LayoutItem("key_value_region", (20.0, 150.0, 300.0, 220.0), 0.80),
        LayoutItem("picture", (20.0, 230.0, 300.0, 400.0), 0.70),
        LayoutItem("table", (22.0, 232.0, 298.0, 398.0), 0.65),
        LayoutItem("picture", (0.0, 0.0, 595.0, 840.0), 0.90),
    ]
    clusters_off = postprocess_layout(raw, (600.0, 850.0), keep_empty_clusters=False)
    items_off = clusters_to_items(clusters_off)
    # Full-page picture dropped, empty regular/container clusters dropped, table beats coincident picture.
    assert [it["label"] for it in items_off] == ["table"]

    clusters_on = postprocess_layout(raw, (600.0, 850.0), keep_empty_clusters=True)
    labels_on = [c.label for c in clusters_on]
    assert "section_header" in labels_on
    assert labels_on.count("text") == 1
    assert "table" in labels_on


def test_clamp_and_rescale_bounds_and_drops_unknown():
    items = clamp_and_rescale(
        [
            ("picture", (-10.0, 5.0, 650.0, 420.0), 0.9),
            ("unknown_label", (10.0, 10.0, 50.0, 50.0), 0.9),
        ],
        (600.0, 400.0),
        (300.0, 200.0),
    )
    assert len(items) == 1
    assert items[0].label == "picture"
    assert items[0].bbox == (0.0, 2.5, 300.0, 200.0)


def test_layout_cache_isolates_by_backend_revision_dpi_and_hash():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        c1 = LayoutCache(root, backend="heron", revision="rev1", dpi=150, pdf_hash="aaaa" * 16)
        c2 = LayoutCache(root, backend="heron", revision="rev2", dpi=150, pdf_hash="aaaa" * 16)
        c3 = LayoutCache(root, backend="heron", revision="rev1", dpi=200, pdf_hash="aaaa" * 16)
        c4 = LayoutCache(root, backend="heron", revision="rev1", dpi=150, pdf_hash="bbbb" * 16)
        assert len({c1.dir, c2.dir, c3.dir, c4.dir}) == 4
        c1.save(1, [{"order": 0, "label": "picture", "bbox": [1, 2, 3, 4], "text": ""}], (100.0, 100.0))
        assert c1.load(1) is not None
        assert c2.load(1) is None


def test_pipeline_works_with_docling_blocked():
    prev = sys.modules.get("docling", ...)
    sys.modules["docling"] = None  # type: ignore[assignment]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            native_pdf = tmp_path / "native.pdf"
            doc = pymupdf.open()
            p = doc.new_page(width=600, height=800)
            p.insert_text((50, 60), "Native text without docling installed", fontsize=12)
            doc.save(str(native_pdf))
            doc.close()

            ir_native = extract_ir(native_pdf, figures=True, ocr=True)
            assert ir_native["elements"] and "docling" not in sys.modules or sys.modules["docling"] is None

            img = Image.new("RGB", (600, 200), (255, 255, 255))
            ImageDraw.Draw(img).text((30, 60), "RAPIDOCR TEST LINE", fill=(0, 0, 0))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            raster_pdf = tmp_path / "raster.pdf"
            rdoc = pymupdf.open()
            rp = rdoc.new_page(width=600, height=200)
            rp.insert_image(rp.rect, stream=buf.getvalue())
            rdoc.save(str(raster_pdf))
            rdoc.close()

            ir_raster = extract_ir(raster_pdf, figures=False, ocr=True)
            assert any("RAPIDOCR" in e["text"].upper() for e in ir_raster["elements"] if e["type"] == "text")
    finally:
        if prev is ...:
            sys.modules.pop("docling", None)
        else:
            sys.modules["docling"] = prev


def test_postprocess_parity_with_recorded_dumps():
    dumps = sorted((Path(__file__).resolve().parent / "fixtures" / "layout_dumps").rglob("p*.json"))
    assert len(dumps) >= 12, len(dumps)
    for f in dumps:
        d = json.loads(f.read_text(encoding="utf-8"))
        raw = [LayoutItem(c["label"], tuple(c["bbox"]), c["confidence"]) for c in d["raw_clusters"]]
        got = postprocess_layout(raw, tuple(d["page_size"]), keep_empty_clusters=False)
        exp = d["postprocessed_clusters"]
        assert len(got) == len(exp), (f.name, len(got), len(exp))
        for g, e in zip(got, exp):
            assert g.label == e["label"]
            assert max(abs(a - b) for a, b in zip(g.bbox, e["bbox"])) <= 1e-3


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
