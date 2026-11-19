import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pdfplumber  # noqa: E402
from PIL import Image  # noqa: E402

import pdf.ir as ir_mod  # noqa: E402
from pdf._test_pdf import TestPdfBuilder  # noqa: E402
from pdf.backends import registered  # noqa: E402
from pdf.eval.diagrams import (  # noqa: E402
    bench_synthetic_diagrams,
    build_diagram_fixtures,
    check_diagram_integrity,
    diagram_graph_metrics,
    load_diagram_gt,
    main as diagrams_main,
)
from pdf.models import BBox, FigureBox, RasterPage  # noqa: E402


def _extract_ir_with_fig_boxes(pdf_path: Path, fig_boxes: list[FigureBox], **kwargs):
    import pdf.figures as figures_mod

    orig_detect = figures_mod.detect_figures
    figures_mod.detect_figures = lambda _p, pages, **_kw: {p: list(fig_boxes) for p in pages}
    try:
        return ir_mod.extract_ir(pdf_path, figures=True, ocr=False, tables=False, **kwargs)
    finally:
        figures_mod.detect_figures = orig_detect


def test_import_pdf_isolation_for_diagrams():
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import sys, pdf; "
                "bad = [m for m in ('pdf.diagrams', 'pdf.eval', 'torch', 'transformers', 'onnxruntime', 'docling') "
                "if m in sys.modules]; "
                "assert not bad, bad"
            ),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.returncode == 0


def test_diagram_backend_registered_and_extract_ir_off_returns_empty():
    assert "vector_first" in registered("diagram")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fixtures = build_diagram_fixtures(tmp_path)
        flow = next(f for f in fixtures if f["name"] == "flowchart")
        fig_box = FigureBox(tuple(flow["figure_bbox"]), "figure", "layout_picture", 0.95)
        ir = _extract_ir_with_fig_boxes(flow["pdf_path"], [fig_box], diagrams=False)
        assert ir["diagrams"] == []
        fig_elems = [e for e in ir["elements"] if e["type"] == "figure"]
        assert len(fig_elems) == 1
        assert "diagram_id" not in fig_elems[0]["metadata"]


def test_diagram_graph_metrics_and_integrity_validators():
    gt = {
        "nodes": [
            {"id": "g1", "shape_type": "rect", "label": "Start Node", "bbox": [10.0, 10.0, 110.0, 60.0]},
            {"id": "g2", "shape_type": "text_node", "label": "End Note", "bbox": [200.0, 10.0, 260.0, 30.0]},
        ],
        "edges": [
            {"id": "ge1", "source_node_id": "g1", "target_node_id": "g2", "direction": "bidirectional", "style": "dashed"},
        ],
        "groups": [
            {"id": "gg1", "label": "Outer", "bbox": [0.0, 0.0, 300.0, 100.0], "parent_group_id": None, "member_node_ids": ["g1"]},
        ],
    }
    pred = {
        "nodes": [
            {"id": "p1", "shape_type": "rect", "label": " start   node ", "bbox": [12.0, 11.0, 108.0, 59.0], "parent_group_id": "pg1"},
            {"id": "p2", "shape_type": "text_node", "label": "end note", "bbox": [205.0, 10.0, 240.0, 28.0], "parent_group_id": None},
        ],
        "edges": [
            {"id": "pe1", "source_node_id": "p2", "target_node_id": "p1", "direction": "bidirectional", "style": "dashed"},
        ],
        "groups": [
            {"id": "pg1", "label": "Outer", "bbox": [0.0, 0.0, 300.0, 100.0], "parent_group_id": None, "member_node_ids": ["p1"]},
        ],
    }
    assert check_diagram_integrity(pred) == []
    m = diagram_graph_metrics(pred, gt)
    assert m["node_f1"] == 1.0
    assert m["edge_f1"] == 1.0
    assert m["text_to_shape_acc"] == 1.0
    assert m["graph_f1"] == 1.0

    bad = {
        "nodes": [
            {"id": "n1", "shape_type": "rect", "label": "A", "bbox": [0, 0, 10, 10], "parent_group_id": "missing_grp"},
        ],
        "edges": [
            {"id": "e1", "source_node_id": "n1", "target_node_id": "missing_node", "direction": "directed", "style": "solid"},
        ],
        "groups": [
            {"id": "ga", "label": "GA", "bbox": [0, 0, 50, 50], "parent_group_id": "gb", "member_node_ids": ["missing_member"]},
            {"id": "gb", "label": "GB", "bbox": [0, 0, 50, 50], "parent_group_id": "ga", "member_node_ids": []},
        ],
    }
    violations = check_diagram_integrity(bad)
    assert len(violations) >= 4


def test_vector_flowchart_and_text_node_e2e():
    with tempfile.TemporaryDirectory() as tmp:
        fixtures = build_diagram_fixtures(Path(tmp))
        flow = next(f for f in fixtures if f["name"] == "flowchart")
        fig_box = FigureBox(tuple(flow["figure_bbox"]), "figure", "layout_picture", 0.95)
        ir = _extract_ir_with_fig_boxes(flow["pdf_path"], [fig_box], diagrams=True)

    assert json.loads(json.dumps(ir)) == ir
    assert len(ir["diagrams"]) == 1
    diag = ir["diagrams"][0]
    assert check_diagram_integrity(diag) == []

    fig_elems = [e for e in ir["elements"] if e["type"] == "figure" and e["level"] == 1]
    assert len(fig_elems) == 1
    assert fig_elems[0]["metadata"]["diagram_id"] == diag["id"]
    assert diag["element_id"] == fig_elems[0]["id"]

    child_text_ids = {e["id"] for e in ir["elements"] if e["parent_id"] == fig_elems[0]["id"]}
    linked_ids = {tid for n in diag["nodes"] for tid in n["text_element_ids"]}
    assert linked_ids and linked_ids.issubset(child_text_ids)

    labels = {n["label"].strip() for n in diag["nodes"]}
    assert {"Start", "Validate", "Process", "Done"}.issubset(labels)
    text_nodes = [n for n in diag["nodes"] if n["shape_type"] == "text_node"]
    assert any(n["label"].strip() == "Done" for n in text_nodes)
    assert any(e["label"].strip() == "ok" and e["direction"] == "directed" for e in diag["edges"])


def test_dashed_bidirectional_and_self_loop():
    from pdf.diagrams import diagram_to_ir_dict, extract_diagrams_on_page
    from pdf.text import extract_words

    with tempfile.TemporaryDirectory() as tmp:
        fixtures = build_diagram_fixtures(Path(tmp))
        by_name = {f["name"]: f for f in fixtures}

        for name in ("dashed_and_bidir", "self_loop"):
            fix = by_name[name]
            with pdfplumber.open(str(fix["pdf_path"])) as doc:
                page = doc.pages[0]
                words = extract_words(page)
                parsed = extract_diagrams_on_page(page, [tuple(fix["figure_bbox"])], words=words)
            assert len(parsed) == 1
            d = diagram_to_ir_dict("diag_0001", "elem_0001", 1, fix["pdf_path"].name, parsed[0])
            assert check_diagram_integrity(d) == []

            if name == "dashed_and_bidir":
                assert any(e["direction"] == "bidirectional" and e["style"] == "solid" for e in d["edges"])
                assert sum(1 for e in d["edges"] if e["style"] == "dashed") >= 2
            else:
                assert any(
                    e["source_node_id"] is not None and e["source_node_id"] == e["target_node_id"]
                    for e in d["edges"]
                )
                assert any(
                    e["source_node_id"] != e["target_node_id"]
                    for e in d["edges"]
                )


def test_nested_groups_dag_and_layered_no_arrows():
    from pdf.diagrams import diagram_to_ir_dict, extract_diagrams_on_page
    from pdf.text import extract_words

    with tempfile.TemporaryDirectory() as tmp:
        fixtures = build_diagram_fixtures(Path(tmp))
        by_name = {f["name"]: f for f in fixtures}

        swim = by_name["swimlane_dag"]
        with pdfplumber.open(str(swim["pdf_path"])) as doc:
            page = doc.pages[0]
            parsed = extract_diagrams_on_page(page, [tuple(swim["figure_bbox"])], words=extract_words(page))
        assert len(parsed) == 1
        d_swim = diagram_to_ir_dict("diag_0001", "elem_0001", 1, swim["pdf_path"].name, parsed[0])
        assert check_diagram_integrity(d_swim) == []
        assert len(d_swim["groups"]) >= 2
        assert any(g["parent_group_id"] is not None for g in d_swim["groups"])
        assert any(g["member_node_ids"] for g in d_swim["groups"])

        lay = by_name["layered_autosar"]
        with pdfplumber.open(str(lay["pdf_path"])) as doc:
            page = doc.pages[0]
            parsed_l = extract_diagrams_on_page(page, [tuple(lay["figure_bbox"])], words=extract_words(page))
        assert len(parsed_l) == 1
        d_lay = diagram_to_ir_dict("diag_0002", "elem_0002", 1, lay["pdf_path"].name, parsed_l[0])
        assert check_diagram_integrity(d_lay) == []
        assert d_lay["diagram_type"] == "layered"
        assert d_lay["edges"] == []
        assert len(d_lay["nodes"]) == 3


def test_closed_loop_from_four_line_segments():
    from pdf.diagrams import diagram_to_ir_dict, extract_diagrams_on_page
    from pdf.text import extract_words

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "four_lines.pdf"
        b = TestPdfBuilder()
        page = b.new_page(width=420.0, height=260.0)
        # Four separate line segments with ~1.0 pt corner gaps (within eps <= 1.5 pt).
        page.draw_line((60.0, 80.0), (179.0, 80.0), width=1.0)
        page.draw_line((180.0, 81.0), (180.0, 140.0), width=1.0)
        page.draw_line((179.0, 141.0), (61.0, 141.0), width=1.0)
        page.draw_line((60.0, 140.0), (60.0, 81.0), width=1.0)
        page.insert_text((95.0, 114.0), "LoopBox", fontsize=10.0)

        page.draw_rect((260.0, 80.0, 360.0, 140.0), width=1.0)
        page.insert_text((290.0, 114.0), "Target", fontsize=10.0)

        page.draw_line((180.0, 110.0), (260.0, 110.0), width=1.0)
        page.draw_polygon([(260.0, 110.0), (252.0, 106.0), (252.0, 114.0)], fill=True, stroke=False)
        b.save(pdf_path)

        fig_box: BBox = (45.0, 65.0, 375.0, 155.0)
        with pdfplumber.open(str(pdf_path)) as doc:
            p0 = doc.pages[0]
            parsed = extract_diagrams_on_page(p0, [fig_box], words=extract_words(p0))
        assert len(parsed) == 1
        d = diagram_to_ir_dict("diag_0001", "elem_0001", 1, pdf_path.name, parsed[0])
        assert check_diagram_integrity(d) == []
        assert any("loopbox" in n["label"].lower() for n in d["nodes"])
        assert len(d["edges"]) >= 1


def test_empty_or_raster_figure_emits_diagram_parse_empty_warning():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        empty_pdf = tmp_path / "empty_fig.pdf"
        b = TestPdfBuilder()
        p = b.new_page(width=400.0, height=300.0)
        p.insert_text((40.0, 40.0), "Surrounding native paragraph text.", fontsize=11.0)
        # Tiny decorative marks satisfy router.figure_candidate while yielding an empty diagram graph.
        for x in (100.0, 140.0, 180.0):
            p.draw_rect((x, 120.0, x + 6.0, 126.0), width=1.0)
        b.save(empty_pdf)

        ir_empty = _extract_ir_with_fig_boxes(
            empty_pdf,
            [FigureBox((50.0, 80.0, 350.0, 240.0), "figure", "layout_picture", 0.9)],
            diagrams=True,
        )
        assert ir_empty["diagrams"] == []
        assert any(w.get("code") == "diagram_parse_empty" for w in ir_empty["metadata"]["warnings"])

        raster_pdf = tmp_path / "raster_fig.pdf"
        img_path = tmp_path / "blank.png"
        Image.new("RGB", (400, 300), "white").save(img_path)
        rb = TestPdfBuilder()
        rp = rb.new_page(width=400.0, height=300.0)
        rp.insert_image((0.0, 0.0, 400.0, 300.0), filename=img_path)
        rb.save(raster_pdf)

        class _FakeReader:
            name = "fake_reader"
            last_stats = None

            def read(self, _path, pages):
                return {
                    pno: RasterPage(
                        blocks=(),
                        figures=(FigureBox((40.0, 40.0, 360.0, 260.0), "figure", "layout_picture", 0.9),),
                        reading_order=(0,),
                    )
                    for pno in pages
                }

        ir_raster = ir_mod.extract_ir(
            raster_pdf,
            figures=True,
            ocr=True,
            tables=False,
            diagrams=True,
            reader=_FakeReader(),
        )
        assert ir_raster["diagrams"] == []
        assert any(w.get("code") == "diagram_parse_empty" for w in ir_raster["metadata"]["warnings"])


def test_synthetic_diagram_benchmark_passes_kpis_and_split_guard():
    res = bench_synthetic_diagrams()
    assert res["diagrams"] == 6
    assert res["integrity_violations"] == 0, res["fixtures"]
    assert res["node_f1"] >= 0.85, res
    assert res["edge_f1"] >= 0.80, res
    assert res["text_to_shape_acc"] >= 0.90, res

    assert diagrams_main(["--task", "synthetic", "--split", "test"]) == 2
    assert diagrams_main(["--task", "gt"]) != 0

    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "eval_runs.jsonl"
        assert diagrams_main(["--task", "synthetic", "--split", "dev", "--log-file", str(log_path)]) == 0
        lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 1
        assert lines[0]["task"] == "diagrams:synthetic"


def test_golden_gt_loader_and_split_filtering():
    import pdf.figures as figures_mod

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fixtures = build_diagram_fixtures(tmp_path)
        flow = next(f for f in fixtures if f["name"] == "flowchart")
        x_cross = next(f for f in fixtures if f["name"] == "x_crossing")
        gt_json = tmp_path / "gt.json"
        gt_json.write_text(
            json.dumps({
                "diagrams": [
                    {
                        "name": "dev_flow",
                        "split": "dev",
                        "pdf_path": flow["pdf_path"].name,
                        "page": 1,
                        "diagram": flow["gt_diagram"],
                    },
                    {
                        "name": "test_cross",
                        "split": "test",
                        "pdf_path": str(x_cross["pdf_path"]),
                        "page": 1,
                        "diagram": x_cross["gt_diagram"],
                    },
                ]
            }),
            encoding="utf-8",
        )
        dev_entries = load_diagram_gt(gt_json, split="dev")
        test_entries = load_diagram_gt(gt_json, split="test")
        assert [e["name"] for e in dev_entries] == ["dev_flow"]
        assert [e["name"] for e in test_entries] == ["test_cross"]

        fig_box = FigureBox(tuple(flow["figure_bbox"]), "figure", "layout_picture", 0.95)
        orig_detect = figures_mod.detect_figures
        figures_mod.detect_figures = lambda _p, pages, **_kw: {p: [fig_box] for p in pages}
        try:
            log_path = tmp_path / "gt_eval.jsonl"
            rc = diagrams_main(["--task", "gt", "--gt", str(gt_json), "--split", "dev", "--log-file", str(log_path)])
            assert rc == 0
            records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            assert len(records) == 1
            assert records[0]["results"][0]["node_f1"] == 1.0
            assert records[0]["results"][0]["edge_f1"] == 1.0
        finally:
            figures_mod.detect_figures = orig_detect


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
