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
    bench_synthetic_visual_diagrams,
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


def test_fork_shared_border_stub_arrow_and_label_regressions():
    from pdf.diagrams import _is_dashed, _matches_arrow_poly, _ArrowPoly, extract_diagrams_on_page
    from pdf.text import extract_words

    # 1. _is_dashed handles zero-dash, dict, and bool.
    assert _is_dashed({"dash": ([0, 0], 0)}) is False
    assert _is_dashed({"dash": ([4, 3], 0)}) is True
    assert _is_dashed({"dash": {"dash": [4, 2], "phase": 0}}) is True
    assert _is_dashed({"dash": True}) is True

    # 2. _matches_arrow_poly handles skewed 3-vertex triangles.
    skew = _ArrowPoly(
        centroid=(280.0, 199.33),
        verts=[(285.0, 200.0), (277.0, 195.0), (278.0, 203.0)],
        bbox=(277.0, 195.0, 285.0, 203.0),
    )
    assert _matches_arrow_poly((285.0, 200.0), (0.985, 0.174), skew) is True

    with tempfile.TemporaryDirectory() as tmp:
        # 3. Orthogonal degree-3 fork + stacked shared-border boxes + stub pin.
        pdf_fork = Path(tmp) / "fork_shared.pdf"
        b = TestPdfBuilder()
        p = b.new_page(width=460.0, height=300.0)
        # Stacked shared-border line boxes ("Top" and "Bot").
        p.draw_line((40.0, 100.0), (120.0, 100.0))
        p.draw_line((40.0, 145.0), (120.0, 145.0))
        p.draw_line((40.0, 190.0), (120.0, 190.0))
        p.draw_line((40.0, 100.0), (40.0, 145.0))
        p.draw_line((120.0, 100.0), (120.0, 145.0))
        p.draw_line((40.0, 145.0), (40.0, 190.0))
        p.draw_line((120.0, 145.0), (120.0, 190.0))
        p.insert_text((65.0, 125.0), "Top", fontsize=10.0)
        p.insert_text((65.0, 170.0), "Bot", fontsize=10.0)
        # 5pt stub pin off Top (must NOT emit self-loop).
        p.draw_line((120.0, 110.0), (125.0, 110.0))
        # Fork from Top -> B and C.
        p.draw_line((120.0, 122.0), (190.0, 122.0))
        p.draw_line((190.0, 122.0), (190.0, 65.0))
        p.draw_line((190.0, 65.0), (260.0, 65.0))
        p.draw_polygon([(260.0, 65.0), (252.0, 61.0), (252.0, 69.0)], fill=True)
        p.draw_rect((260.0, 45.0, 340.0, 85.0))
        p.insert_text((285.0, 68.0), "B", fontsize=10.0)
        p.draw_line((190.0, 122.0), (190.0, 190.0))
        p.draw_line((190.0, 190.0), (260.0, 190.0))
        p.draw_polygon([(260.0, 190.0), (252.0, 186.0), (252.0, 194.0)], fill=True)
        p.draw_rect((260.0, 170.0, 340.0, 210.0))
        p.insert_text((285.0, 193.0), "C", fontsize=10.0)
        b.save(pdf_fork)

        with pdfplumber.open(str(pdf_fork)) as doc:
            d = extract_diagrams_on_page(doc.pages[0], [(20.0, 30.0, 360.0, 230.0)], words=extract_words(doc.pages[0]))[0]
        labels = {n.label: n.id for n in d.nodes}
        assert {"Top", "Bot", "B", "C"}.issubset(labels.keys()), labels
        assert not any(e.source_node_id == e.target_node_id for e in d.edges)
        pairs = {(e.source_node_id, e.target_node_id) for e in d.edges}
        assert (labels["Top"], labels["B"]) in pairs and (labels["Top"], labels["C"]) in pairs, pairs

        # 4. Bidir text_node at source + group header not stealing top-routed edge label + near-edge text_element_ids.
        pdf_lbl = Path(tmp) / "bidir_lbl.pdf"
        b2 = TestPdfBuilder()
        p2 = b2.new_page(width=480.0, height=320.0)
        p2.draw_rect((40.0, 40.0, 440.0, 270.0))
        p2.insert_text((55.0, 60.0), "MyGroup", fontsize=10.0, bold=True)
        p2.draw_rect((80.0, 140.0, 170.0, 190.0))
        p2.insert_text((100.0, 145.0), "NearEdge", fontsize=9.0)
        p2.draw_rect((300.0, 140.0, 390.0, 190.0))
        p2.insert_text((325.0, 168.0), "Dst", fontsize=10.0)
        p2.draw_line((125.0, 140.0), (125.0, 110.0))
        p2.draw_line((125.0, 110.0), (345.0, 110.0))
        p2.draw_line((345.0, 110.0), (345.0, 140.0))
        p2.draw_polygon([(345.0, 140.0), (341.0, 132.0), (349.0, 132.0)], fill=True)
        p2.insert_text((215.0, 102.0), "trigger", fontsize=9.0)
        p2.insert_text((80.0, 232.0), "SrcLabel", fontsize=10.0)
        p2.draw_line((135.0, 228.0), (295.0, 228.0))
        p2.draw_polygon([(135.0, 228.0), (143.0, 224.0), (143.0, 232.0)], fill=True)
        p2.draw_polygon([(295.0, 228.0), (287.0, 224.0), (287.0, 232.0)], fill=True)
        p2.draw_rect((295.0, 208.0, 390.0, 248.0))
        p2.insert_text((325.0, 232.0), "EndBox", fontsize=10.0)
        b2.save(pdf_lbl)

        ir2 = _extract_ir_with_fig_boxes(pdf_lbl, [FigureBox((30.0, 30.0, 450.0, 280.0), "figure", "layout_picture", 0.95)], diagrams=True)
        d2 = ir2["diagrams"][0]
        assert check_diagram_integrity(d2) == []
        assert any(g["label"] == "MyGroup" for g in d2["groups"]), d2["groups"]
        assert any(e["label"] == "trigger" for e in d2["edges"]), d2["edges"]
        assert any(n["shape_type"] == "text_node" and "SrcLabel" in n["label"] for n in d2["nodes"]), d2["nodes"]
        near = next(n for n in d2["nodes"] if "NearEdge" in n["label"])
        assert near["text_element_ids"], near


def test_bench_gt_diagrams_ignores_unrelated_zero_iou_diagram_and_checks_integrity():
    from pdf.eval.diagrams import bench_gt_diagrams, check_diagram_integrity
    from pdf.models import FigureBox
    import pdf.figures as figures_mod

    bad = {
        "nodes": [{"id": "n1", "parent_group_id": "g1", "text_element_ids": ["", "elem_0001", "elem_0001"]}],
        "edges": [
            {"id": "e1", "source_node_id": "n1", "target_node_id": "n1"},
            {"id": "e1", "source_node_id": "n1", "target_node_id": "n1"},
        ],
        "groups": [{"id": "g1", "member_node_ids": []}],
    }
    viols = check_diagram_integrity(bad)
    assert any("duplicate edge id" in v for v in viols)
    assert any("invalid text_element_id" in v for v in viols)
    assert any("duplicate text_element_id" in v for v in viols)
    assert any("missing from member_node_ids" in v for v in viols)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pdf_path = tmp_path / "two_regions.pdf"
        b = TestPdfBuilder()
        p = b.new_page(width=500.0, height=600.0)
        p.draw_rect((60.0, 60.0, 160.0, 110.0))
        p.insert_text((90.0, 90.0), "Alpha", fontsize=10.0)
        b.save(pdf_path)

        gt_json = tmp_path / "gt.json"
        gt_json.write_text(
            json.dumps([{
                "name": "disjoint",
                "split": "dev",
                "pdf_path": str(pdf_path),
                "page": 1,
                "diagram": {
                    "bbox": [60.0, 400.0, 260.0, 520.0],
                    "nodes": [{"id": "g1", "label": "Missing", "bbox": [70.0, 410.0, 150.0, 450.0], "shape_type": "rect"}],
                    "edges": [],
                    "groups": [],
                },
            }]),
            encoding="utf-8",
        )
        orig_detect = figures_mod.detect_figures
        figures_mod.detect_figures = lambda _p, pages, **_kw: {pno: [FigureBox((50.0, 50.0, 250.0, 150.0), "figure", "layout_picture", 0.95)] for pno in pages}
        try:
            res = bench_gt_diagrams(gt_path=gt_json, split="dev")
        finally:
            figures_mod.detect_figures = orig_detect
        assert res["fixtures"][0]["node_fp"] == 0 and res["fixtures"][0]["node_fn"] == 1


def _render_pdf_page_to_raster_pdf(src_pdf: Path, dst_pdf: Path, tmp_dir: Path, *, dpi: int = 150) -> tuple[float, float]:
    from pdf._pdf_io import page_bbox

    with pdfplumber.open(str(src_pdf)) as doc:
        p0 = doc.pages[0]
        pb = page_bbox(p0)
        img = p0.to_image(resolution=dpi).original.convert("RGB")
    img_path = tmp_dir / f"{dst_pdf.stem}.png"
    img.save(img_path)
    rb = TestPdfBuilder()
    rp = rb.new_page(width=pb[2], height=pb[3])
    rp.insert_image((0.0, 0.0, pb[2], pb[3]), filename=img_path)
    rb.save(dst_pdf)
    return pb[2], pb[3]


def test_visual_fallback_lazy_ocr_and_bitmap_on_native_page():
    import pdf.diagrams as diag_mod
    from pdf._pdf_io import page_bbox

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fixtures = build_diagram_fixtures(tmp_path)
        flow = next(f for f in fixtures if f["name"] == "flowchart")

        # 1. Vector path must never invoke OCR backend.
        orig_get_backend = diag_mod.get_backend
        ocr_calls = 0

        def _spy_get_backend(kind: str, name: str, **opts):
            nonlocal ocr_calls
            if kind == "ocr":
                ocr_calls += 1
            return orig_get_backend(kind, name, **opts)

        diag_mod.get_backend = _spy_get_backend
        try:
            with pdfplumber.open(str(flow["pdf_path"])) as doc:
                vec_out = diag_mod.extract_diagrams_on_page(doc.pages[0], [tuple(flow["figure_bbox"])], is_raster=False)
            assert len(vec_out) == 1 and vec_out[0].nodes
            assert getattr(vec_out[0], "metadata", None) is None
            assert ocr_calls == 0
        finally:
            diag_mod.get_backend = orig_get_backend

        # 2. Bitmap figure embedded on a native page triggers visual fallback when vectors are absent.
        with pdfplumber.open(str(flow["pdf_path"])) as doc:
            p0 = doc.pages[0]
            pb = page_bbox(p0)
            cropped = p0.crop(tuple(flow["figure_bbox"])).to_image(resolution=150).original.convert("RGB")
        bmp_path = tmp_path / "flow_crop.png"
        cropped.save(bmp_path)

        hybrid_pdf = tmp_path / "hybrid_bitmap_fig.pdf"
        hb = TestPdfBuilder()
        hp = hb.new_page(width=pb[2], height=pb[3])
        hp.insert_text((45.0, 40.0), "Native surrounding paragraph text.", fontsize=11.0)
        hp.insert_image(tuple(flow["figure_bbox"]), filename=bmp_path)
        hb.save(hybrid_pdf)

        fig_box = FigureBox(tuple(flow["figure_bbox"]), "figure", "layout_picture", 0.92)
        ir = _extract_ir_with_fig_boxes(hybrid_pdf, [fig_box], diagrams=True)
        assert len(ir["diagrams"]) == 1
        d = ir["diagrams"][0]
        assert check_diagram_integrity(d) == []
        assert d["source"] == "vector_graph"
        assert d["metadata"]["extraction_mode"] == "visual_fallback"
        assert 0.55 <= d["confidence"] < 0.90 < 1.0
        m = diagram_graph_metrics(d, flow["gt_diagram"])
        assert m["node_f1"] >= 0.85 and m["edge_f1"] >= 0.80 and m["text_to_shape_acc"] >= 0.90, m


def test_visual_raster_diagram_e2e_extract_ir_with_ocr():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fixtures = build_diagram_fixtures(tmp_path)
        swim = next(f for f in fixtures if f["name"] == "swimlane_dag")
        raster_pdf = tmp_path / "swim_raster.pdf"
        _render_pdf_page_to_raster_pdf(swim["pdf_path"], raster_pdf, tmp_path)

        fig_box = FigureBox(tuple(swim["figure_bbox"]), "figure", "layout_picture", 0.91)

        class _FakeReader:
            name = "fake_reader"
            last_stats = None

            def read(self, _path, pages):
                return {
                    pno: RasterPage(blocks=(), figures=(fig_box,), reading_order=(0,))
                    for pno in pages
                }

        ir = ir_mod.extract_ir(
            raster_pdf,
            figures=True,
            ocr=True,
            tables=False,
            diagrams=True,
            reader=_FakeReader(),
        )
        assert json.loads(json.dumps(ir)) == ir
        assert len(ir["diagrams"]) == 1
        d = ir["diagrams"][0]
        assert check_diagram_integrity(d) == []
        assert d["source"] == "vector_graph"
        assert d["metadata"]["extraction_mode"] == "visual_fallback"
        assert 0.55 <= d["confidence"] < 0.90
        assert len(d["groups"]) >= 2
        assert any(g["parent_group_id"] is not None for g in d["groups"])
        m = diagram_graph_metrics(d, swim["gt_diagram"])
        assert m["node_f1"] == 1.0 and m["edge_f1"] == 1.0 and m["text_to_shape_acc"] == 1.0, m


def test_visual_diamond_ellipse_and_chevron_shapes():
    import cv2
    import numpy as np
    from pdf.diagrams import diagram_to_ir_dict, extract_diagrams_on_page

    img = np.full((360, 760, 3), 255, dtype=np.uint8)
    # Ellipse ("Start") -> Diamond ("Check") -> Rect ("Finish") with open V-chevron arrowheads.
    cv2.ellipse(img, (110, 180), (70, 42), 0, 0, 360, (0, 0, 0), 2)
    cv2.putText(img, "Start", (78, 187), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

    diamond_pts = np.array([[380, 120], [455, 180], [380, 240], [305, 180]], dtype=np.int32)
    cv2.polylines(img, [diamond_pts], True, (0, 0, 0), 2)
    cv2.putText(img, "Check", (345, 187), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.rectangle(img, (560, 140), (700, 220), (0, 0, 0), 2)
    cv2.putText(img, "Finish", (594, 187), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.line(img, (180, 180), (305, 180), (0, 0, 0), 2)
    cv2.line(img, (292, 172), (305, 180), (0, 0, 0), 2)
    cv2.line(img, (292, 188), (305, 180), (0, 0, 0), 2)

    cv2.line(img, (455, 180), (560, 180), (0, 0, 0), 2)
    cv2.line(img, (547, 172), (560, 180), (0, 0, 0), 2)
    cv2.line(img, (547, 188), (560, 180), (0, 0, 0), 2)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        img_path = tmp_path / "shapes.png"
        Image.fromarray(img).save(img_path)
        pdf_path = tmp_path / "shapes_raster.pdf"
        rb = TestPdfBuilder()
        rp = rb.new_page(width=400.0, height=200.0)
        rp.insert_image((0.0, 0.0, 400.0, 200.0), filename=img_path)
        rb.save(pdf_path)

        with pdfplumber.open(str(pdf_path)) as doc:
            parsed = extract_diagrams_on_page(doc.pages[0], [(5.0, 5.0, 395.0, 195.0)], is_raster=True)
        assert len(parsed) == 1
        d = diagram_to_ir_dict("diag_0001", "elem_0001", 1, pdf_path.name, parsed[0])
        assert check_diagram_integrity(d) == []
        assert d["metadata"]["extraction_mode"] == "visual_fallback"
        assert 0.55 <= d["confidence"] < 0.90
        by_type = {n["shape_type"]: n["label"].lower() for n in d["nodes"]}
        assert "ellipse" in by_type and "start" in by_type["ellipse"], d["nodes"]
        assert "diamond" in by_type and "check" in by_type["diamond"], d["nodes"]
        assert "rect" in by_type and "finish" in by_type["rect"], d["nodes"]
        assert sum(1 for e in d["edges"] if e["direction"] == "directed") >= 2, d["edges"]


def test_visual_synthetic_diagram_benchmark_passes_kpis():
    res = bench_synthetic_visual_diagrams()
    assert res["diagrams"] == 6
    assert res["integrity_violations"] == 0, res["fixtures"]
    assert res["node_f1"] >= 0.85, res
    assert res["edge_f1"] >= 0.80, res
    assert res["text_to_shape_acc"] >= 0.90, res
    for row in res["fixtures"]:
        assert row.get("extraction_mode") == "visual_fallback", row
        assert 0.55 <= row["confidence"] < 0.90, row

    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "eval_runs.jsonl"
        assert diagrams_main(["--task", "visual", "--split", "dev", "--log-file", str(log_path)]) == 0
        lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 1
        assert lines[0]["task"] == "diagrams:visual"


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
