import io
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw  # noqa: E402

from pdf import extract_ir  # noqa: E402
from pdf._test_pdf import TestPdfBuilder  # noqa: E402
from pdf.backends import CachedLayoutBackend, LayoutCache, PageLayout, RunStats  # noqa: E402
from pdf.backends.layout_cache import DEFAULT_POSTPROCESS_VERSION  # noqa: E402
from pdf.backends.rss import peak_rss_mb  # noqa: E402
from pdf.backends.runtime import run_isolated_json  # noqa: E402
from pdf.layout import (  # noqa: E402
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
        c5 = LayoutCache(
            root,
            backend="heron",
            revision="rev1",
            dpi=150,
            pdf_hash="aaaa" * 16,
            postprocess_version="v3",
        )
        assert DEFAULT_POSTPROCESS_VERSION == "v2"
        assert DEFAULT_POSTPROCESS_VERSION in c1.dir.name
        assert len({c1.dir, c2.dir, c3.dir, c4.dir, c5.dir}) == 5
        assert c1.save(1, [{"order": 0, "label": "picture", "bbox": [1, 2, 3, 4], "text": ""}], (100.0, 100.0))
        assert c1.load(1) is not None
        assert c2.load(1) is None
        assert c5.load(1) is None


def test_layout_cache_load_recovers_from_corrupt_and_incomplete_files():
    with tempfile.TemporaryDirectory() as tmp:
        cache = LayoutCache(Path(tmp), backend="heron", revision="rev1", dpi=150, pdf_hash="aaaa" * 16)
        cache.dir.mkdir(parents=True, exist_ok=True)

        bad_payloads = [
            b'{"items": [{"label": "text"',
            b"\xff\xfe\xfd",
            b'["not", "a", "dict"]',
            b'{"items": "not_a_list", "page_size": [100, 200]}',
            b'{"items": [], "page_size": [100]}',
            b'{"items": [], "page_size": null}',
            b'{"items": [], "page_size": ["bad", 200]}',
        ]
        for idx, raw in enumerate(bad_payloads, start=1):
            cache.path_for(idx).write_bytes(raw)
            assert cache.load(idx) is None, idx


def test_layout_cache_atomic_save_and_error_cleanup():
    with tempfile.TemporaryDirectory() as tmp:
        cache = LayoutCache(Path(tmp), backend="heron", revision="rev1", dpi=150, pdf_hash="aaaa" * 16)
        orig_items = [{"order": 0, "label": "text", "bbox": [1.0, 2.0, 3.0, 4.0], "text": "ok"}]
        assert cache.save(1, orig_items, (600.0, 800.0)) is True
        assert cache.last_save_error is None
        assert [p.name for p in cache.dir.iterdir()] == ["p001.json"]

        with mock.patch("pdf.backends.layout_cache.os.replace", side_effect=OSError("disk error")):
            ok = cache.save(1, [{"order": 0, "label": "table", "bbox": [0, 0, 1, 1], "text": ""}], (10.0, 10.0))
        assert ok is False
        assert isinstance(cache.last_save_error, OSError)
        assert [p.name for p in cache.dir.iterdir()] == ["p001.json"]
        assert cache.load(1) == (orig_items, (600.0, 800.0))


def test_cached_layout_backend_uses_precomputed_hash_and_postprocess_version():
    class _DummyBackend:
        name = "dummy"
        revision = "r1"
        cache_dpi = 150
        last_stats = RunStats(infer_sec=0.01, pages=1, in_process=True)

        def __init__(self) -> None:
            self.calls = 0

        def analyze(self, _pdf_path, pages):
            self.calls += 1
            return {
                p: PageLayout(items=[{"label": "picture", "bbox": [0, 0, 10, 10]}], frame=(100.0, 100.0))
                for p in pages
            }

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pdf_path = root / "doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")
        backend = _DummyBackend()
        cached = CachedLayoutBackend(backend, root, postprocess_version="v9")
        with mock.patch("pdf.backends.layout_cache.file_sha256", side_effect=AssertionError("should not hash")):
            out1 = cached.analyze(pdf_path, [1], pdf_hash="1234" * 16)
            out2 = cached.analyze(pdf_path, [1], pdf_hash="1234" * 16)
        assert backend.calls == 1
        assert out1[1].items == out2[1].items
        assert (root / f"dummy_r1_dpi150_v9_{'1234' * 4}").is_dir()


def test_run_isolated_json_timeout():
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "slow_child.py"
        script.write_text("import time\ntime.sleep(5)\n", encoding="utf-8")
        try:
            run_isolated_json(
                Path(sys.executable),
                script,
                lambda out: [str(out)],
                label="SlowBackend",
                timeout_sec=0.2,
            )
            raise AssertionError("expected RuntimeError on timeout")
        except RuntimeError as exc:
            msg = str(exc)
            assert "SlowBackend" in msg
            assert "slow_child.py" in msg
            assert "0.2" in msg


def test_peak_rss_mb_darwin_vs_linux_units():
    fake_usage = types.SimpleNamespace(ru_maxrss=10 * 1024 * 1024)
    fake_resource = types.SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _who: fake_usage)
    with mock.patch.dict(sys.modules, {"resource": fake_resource}):
        with mock.patch.object(sys, "platform", "darwin"):
            assert peak_rss_mb() == 10.0
        with mock.patch.object(sys, "platform", "linux"):
            assert peak_rss_mb() == 10240.0


def test_pipeline_works_with_docling_blocked():
    prev = sys.modules.get("docling", ...)
    sys.modules["docling"] = None  # type: ignore[assignment]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            native_pdf = tmp_path / "native.pdf"
            b = TestPdfBuilder()
            p = b.new_page(width=600, height=800)
            p.insert_text((50, 60), "Native text without docling installed", fontsize=12)
            b.save(native_pdf)

            ir_native = extract_ir(native_pdf, figures=True, ocr=True)
            assert ir_native["elements"] and "docling" not in sys.modules or sys.modules["docling"] is None

            img = Image.new("RGB", (600, 200), (255, 255, 255))
            ImageDraw.Draw(img).text((30, 60), "RAPIDOCR TEST LINE", fill=(0, 0, 0))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            raster_pdf = tmp_path / "raster.pdf"
            rb = TestPdfBuilder()
            rp = rb.new_page(width=600, height=200)
            rp.insert_image((0, 0, 600, 200), stream=buf.getvalue())
            rb.save(raster_pdf)

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


def test_postprocess_table_vs_document_index_priority():
    # Real table overlapping document_index must survive unless document_index outscores it by >= 0.20.
    kept_tie = postprocess_layout(
        [
            LayoutItem("table", (40.0, 80.0, 520.0, 420.0), 0.62),
            LayoutItem("document_index", (38.0, 78.0, 524.0, 422.0), 0.70),
        ],
        (600.0, 850.0),
    )
    assert [c.label for c in kept_tie] == ["table"]

    kept_toc = postprocess_layout(
        [
            LayoutItem("table", (40.0, 80.0, 520.0, 420.0), 0.52),
            LayoutItem("document_index", (38.0, 78.0, 524.0, 422.0), 0.90),
        ],
        (600.0, 850.0),
    )
    assert [c.label for c in kept_toc] == ["document_index"]


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
