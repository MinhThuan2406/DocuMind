import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdf.eval import (  # noqa: E402
    OMNIDOC_SAMPLE_PAGES,
    char_counts,
    kendall_tau_matched,
    kendall_tau_penalized,
    load_autosar_gt,
    load_omnidoc_reading_order,
    load_omnidoc_tables,
    match_count,
    merged_cell_span_f1,
    prf,
    strip_markup,
)
from pdf.eval.calibration import eval_figures_calibration, main as calibration_main  # noqa: E402
from pdf.eval.compare import main as compare_main  # noqa: E402
from pdf.eval.compare_hierarchy import main as hierarchy_main  # noqa: E402
from pdf.eval.gt import (  # noqa: E402
    check_split_guard,
    load_autosar_diagram_gt,
    load_autosar_table_gt,
    load_router_gt,
    validate_autosar_diagram_gt,
    validate_autosar_table_gt,
    validate_router_gt,
)
from pdf.eval.metrics import (  # noqa: E402
    calibration_bins,
    confidence_separation_summary,
    expected_calibration_error,
    matched_pred_indices,
)
from pdf.eval.reading_order import bench_synthetic_reading_order, main as reading_order_main  # noqa: E402
from pdf.eval.tables import main as tables_main  # noqa: E402
from pdf.models import FigureBox, TableCell  # noqa: E402


def test_pdf_init_does_not_import_eval():
    # Subprocess check confirms importing pdf keeps pdf.eval lazy.
    proc = subprocess.run(
        [sys.executable, "-B", "-c", "import sys, pdf; assert 'pdf.eval' not in sys.modules"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.returncode == 0


def test_pdf_eval_init_does_not_import_heavy_ml_libs():
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import sys, pdf.eval; "
                "bad = [m for m in ('torch', 'transformers', 'onnxruntime', 'docling') if m in sys.modules]; "
                "assert not bad, bad"
            ),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.returncode == 0


def test_strip_markup_preserves_math_inequalities():
    sample = r"Let $a < x + 1$ and $x > 0$. <table><tr><td>42</td></tr></table> \frac{a}{b}"
    cleaned = strip_markup(sample)
    assert "a < x + 1" in cleaned
    assert "x > 0" in cleaned
    assert "<table>" not in cleaned and "<td>" not in cleaned
    assert "42" in cleaned


def test_match_count_strict_and_lenient():
    gt = [[0.0, 0.0, 100.0, 100.0]]
    loose_pred = [[0.0, 0.0, 140.0, 140.0]]
    wide_pred = [[0.0, 0.0, 142.0, 142.0]]
    assert match_count(loose_pred, gt, "strict") == 1
    assert match_count(loose_pred, gt, "lenient") == 1
    assert match_count(wide_pred, gt, "strict") == 0
    assert match_count(wide_pred, gt, "lenient") == 0
    assert prf(27, 3, 3) == (0.9, 0.9, 0.9)


def test_load_autosar_gt_skips_small_and_splits():
    with tempfile.TemporaryDirectory() as tmp:
        gt_path = Path(tmp) / "gt.json"
        gt_path.write_text(
            json.dumps({
                "documents": {
                    "DocA": {
                        "dev_pages": [1, 5],
                        "dev_figures": {
                            "1": [
                                {"bbox": [10, 10, 100, 100], "flags": []},
                                {"bbox": [10, 10, 20, 20], "flags": ["small"]},
                            ]
                        },
                        "test_pages": [2],
                        "figures": {"2": [{"bbox": [20, 20, 200, 200], "flags": []}]},
                    },
                    "DocB": {
                        "test_pages": [3],
                        "figures": {},
                    },
                }
            }),
            encoding="utf-8",
        )
        dev = load_autosar_gt(split="dev", gt_path=gt_path)
        assert len(dev) == 1 and dev[0].stem == "DocA"
        assert dev[0].boxes_by_page[1] == [[10.0, 10.0, 100.0, 100.0]]
        assert dev[0].boxes_by_page[5] == []
        test = load_autosar_gt(split="test", gt_path=gt_path)
        assert [d.stem for d in test] == ["DocA", "DocB"]


def test_split_guard_refuses_test_without_final():
    assert compare_main(["--task", "figures", "--split", "test", "--backends", "heron"]) == 2
    assert tables_main(["--task", "omnidoc", "--split", "test"]) == 2
    assert hierarchy_main(["--split", "test"]) == 2
    assert hierarchy_main(["--split", "test", "--final"]) == 2
    assert reading_order_main(["--task", "synthetic", "--split", "test"]) == 2


def test_char_counts_whitespace_normalized():
    assert char_counts("A B\nC") == char_counts("ABC")


def test_merged_cell_span_f1():
    gt_html = (
        "<table>"
        '<tr><td rowspan="2">A</td><td colspan="2">B</td></tr>'
        "<tr><td>C</td><td>D</td></tr>"
        "</table>"
    )
    exact_cells = [
        TableCell(row=0, col=0, rowspan=2, colspan=1, text="A", bbox=None),
        TableCell(row=0, col=1, rowspan=1, colspan=2, text="B", bbox=None),
        TableCell(row=1, col=1, rowspan=1, colspan=1, text="C", bbox=None),
        TableCell(row=1, col=2, rowspan=1, colspan=1, text="D", bbox=None),
    ]
    assert merged_cell_span_f1(exact_cells, gt_html) == 1.0

    # Grid occupancy check: row=1, col=1 has colspan=2 because col=0 is occupied by (0,0) rowspan=2.
    row1_span_html = (
        "<table>"
        '<tr><td rowspan="2">A</td><td>B</td><td>C</td></tr>'
        '<tr><td colspan="2">D</td></tr>'
        "</table>"
    )
    ir_cells = [
        {"row": 0, "col": 0, "rowspan": 2, "colspan": 1},
        {"row": 1, "col": 1, "rowspan": 1, "colspan": 2},
    ]
    assert merged_cell_span_f1(ir_cells, row1_span_html) == 1.0

    partial_cells = [{"row": 0, "col": 0, "rowspan": 2, "colspan": 1}]
    assert merged_cell_span_f1(partial_cells, gt_html) == round(2.0 / 3.0, 4)
    assert merged_cell_span_f1([], gt_html) == 0.0
    assert merged_cell_span_f1([{"row": 0, "col": 0, "rowspan": 1, "colspan": 1}], "<table><tr><td>X</td></tr></table>") == 1.0


def test_kendall_tau_matched_and_penalized():
    assert kendall_tau_matched(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert kendall_tau_penalized(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert kendall_tau_matched(["c", "b", "a"], ["a", "b", "c"]) == -1.0
    assert kendall_tau_penalized(["c", "b", "a"], ["a", "b", "c"]) == -1.0
    assert kendall_tau_matched(["a", "c", "b"], ["a", "b", "c"]) == round(1.0 / 3.0, 4)

    # Missing or extra elements leave matched tau at 1.0 but strictly lower penalized tau.
    assert kendall_tau_matched(["a", "b"], ["a", "b", "c"]) == 1.0
    assert kendall_tau_penalized(["a", "b"], ["a", "b", "c"]) < 1.0
    assert kendall_tau_matched(["a", "b", "c", "extra"], ["a", "b", "c"]) == 1.0
    assert kendall_tau_penalized(["a", "b", "c", "extra"], ["a", "b", "c"]) < 1.0
    assert kendall_tau_penalized(["c", "b", "a", "extra"], ["a", "b", "c"]) == -1.0
    assert kendall_tau_matched([], []) == 1.0
    assert kendall_tau_penalized([], []) == 1.0
    assert kendall_tau_matched([], ["a"]) == 0.0
    assert kendall_tau_penalized([], ["a"]) == -1.0
    assert kendall_tau_matched(["a"], ["a"]) == 1.0
    assert kendall_tau_matched(["a"], ["a", "b", "c"]) == 0.0


def test_match_elements_to_gt_skips_level2_children():
    from pdf.eval.reading_order import _match_elements_to_gt

    gt_blocks = [
        {"id": 10, "bbox": [0.0, 0.0, 200.0, 200.0]},
        {"id": 20, "bbox": [0.0, 250.0, 200.0, 450.0]},
    ]
    pred_elems = [
        {"level": 1, "bbox": [0.0, 0.0, 96.0, 96.0]},
        {"level": 2, "bbox": [0.0, 120.0, 96.0, 216.0]},
        {"level": 1, "bbox": [0.0, 120.0, 96.0, 216.0]},
    ]
    matched = _match_elements_to_gt(pred_elems, gt_blocks, min_iou=0.5)
    assert matched == [10, 20]


def test_load_omnidoc_reading_order_excludes_sample_pages_and_splits():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        gt_file = tmp_path / "omni.json"
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()

        def _det(order: int, anno_id: int, y: float) -> dict:
            return {
                "category_type": "text_block",
                "poly": [10, y, 100, y, 100, y + 20, 10, y + 20],
                "ignore": False,
                "order": order,
                "anno_id": anno_id,
                "text": f"block {anno_id}",
            }

        sample_name = OMNIDOC_SAMPLE_PAGES[0]
        candidate_names = [f"doc_{i:02d}.png" for i in range(12)]
        docs = [
            {
                "page_info": {"image_path": sample_name, "page_attribute": {"language": "english"}},
                "layout_dets": [_det(0, 1, 10), _det(1, 2, 40)],
            },
            {
                "page_info": {"image_path": "zh_doc.png", "page_attribute": {"language": "simplified_chinese"}},
                "layout_dets": [_det(0, 1, 10), _det(1, 2, 40)],
            },
        ]
        (img_dir / sample_name).write_bytes(b"x")
        (img_dir / "zh_doc.png").write_bytes(b"x")
        for idx, name in enumerate(candidate_names):
            (img_dir / name).write_bytes(b"x")
            docs.append({
                "page_info": {"image_path": name, "page_attribute": {"language": "english"}},
                "layout_dets": [_det(1, 20 + idx, 50), _det(0, 10 + idx, 10)],
            })
        gt_file.write_text(json.dumps(docs), encoding="utf-8")

        dev = load_omnidoc_reading_order(split="dev", gt_file=gt_file, img_dir=img_dir)
        test = load_omnidoc_reading_order(split="test", gt_file=gt_file, img_dir=img_dir)
        dev_names = {d["image_name"] for d in dev}
        test_names = {d["image_name"] for d in test}
        assert sample_name not in dev_names and sample_name not in test_names
        assert "zh_doc.png" not in dev_names and "zh_doc.png" not in test_names
        assert dev_names and test_names
        assert dev_names.isdisjoint(test_names)
        assert dev_names | test_names == set(candidate_names)
        assert [b["order"] for b in dev[0]["blocks"]] == [0, 1]


def test_load_omnidoc_tables_deterministic_disjoint_splits():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        gt_file = tmp_path / "omni.json"
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        names = [f"tbl_{i:02d}.png" for i in range(10)]
        docs = []
        for name in names:
            (img_dir / name).write_bytes(b"x")
            docs.append({
                "page_info": {"image_path": name, "page_attribute": {"language": "english"}},
                "layout_dets": [
                    {
                        "category_type": "table",
                        "poly": [0, 0, 100, 0, 100, 100, 0, 100],
                        "html": "<table><tr><td>A</td></tr></table>",
                    }
                ],
            })
        gt_file.write_text(json.dumps(docs), encoding="utf-8")

        dev = load_omnidoc_tables(split="dev", gt_file=gt_file, img_dir=img_dir)
        test = load_omnidoc_tables(split="test", gt_file=gt_file, img_dir=img_dir)
        dev_imgs = {item["image_name"] for item in dev}
        test_imgs = {item["image_name"] for item in test}
        assert dev_imgs and test_imgs
        assert dev_imgs.isdisjoint(test_imgs)
        assert dev_imgs | test_imgs == set(names)

        # Multi-table page keeps all tables on that page when whole_pages=True.
        multi_gt = tmp_path / "multi_omni.json"
        (img_dir / "multi.png").write_bytes(b"x")
        multi_gt.write_text(
            json.dumps([
                {
                    "page_info": {"image_path": "multi.png", "page_attribute": {"language": "english"}},
                    "layout_dets": [
                        {"category_type": "table", "poly": [0, 0, 50, 0, 50, 50, 0, 50], "html": "<table><tr><td>1</td></tr></table>"},
                        {"category_type": "table", "poly": [0, 60, 50, 60, 50, 100, 0, 100], "html": "<table><tr><td>2</td></tr></table>"},
                    ],
                }
            ]),
            encoding="utf-8",
        )
        assert len(load_omnidoc_tables(split=None, limit=1, whole_pages=False, gt_file=multi_gt, img_dir=img_dir)) == 1
        assert len(load_omnidoc_tables(split=None, limit=1, whole_pages=True, gt_file=multi_gt, img_dir=img_dir)) == 2


def test_compare_raster_test_split_refused():
    from pdf.eval import compare

    assert compare.main(["--task", "raster", "--split", "test"]) == 2


def test_eval_raster_worker_accepts_ocr_high_dpi():
    from collections import Counter
    from unittest.mock import patch
    from pdf.eval import compare

    fake_selected = [("p1.png", Path("p1.png"), Counter("hello"))]
    fake_ir = {
        "elements": [
            {"type": "text", "page": 1, "source": "ocr", "confidence": 0.9, "text": "hel"},
            {"type": "text", "page": 1, "source": "ocr_high_dpi", "confidence": 0.95, "text": "lo"},
        ]
    }
    with (
        patch.object(compare, "load_omnidoc_gt", return_value=fake_selected),
        patch.object(compare, "get_backend", return_value=object()),
        patch.object(compare, "build_raster_pdf"),
        patch("pdf.extract_ir", return_value=fake_ir),
    ):
        res = compare._eval_raster_worker("composed", "heron-onnx", "rapidocr-onnx", "cpu", 1)
    assert res["pooled_recall"] == 1.0
    assert res["pooled_precision"] == 1.0


def test_synthetic_reading_order_fixtures():
    res = bench_synthetic_reading_order()
    assert res["pages"] == 3
    assert res["mean_tau_matched"] == 1.0
    assert res["mean_tau_penalized"] == 1.0


def test_calibration_bins_and_expected_calibration_error():
    empty_bins = calibration_bins([], [], n_bins=5)
    assert len(empty_bins) == 5
    assert all(b["count"] == 0 and b["accuracy"] is None and b["avg_confidence"] is None for b in empty_bins)
    assert expected_calibration_error([], [], n_bins=10) == 0.0

    confs_perfect = [0.8] * 10
    labels_perfect = [1] * 8 + [0] * 2
    assert expected_calibration_error(confs_perfect, labels_perfect, n_bins=10) == 0.0

    ece = expected_calibration_error([0.95, 0.95, 0.25], [1, 0, 1], n_bins=10)
    assert abs(ece - 0.55) < 1e-6

    bins = calibration_bins([0.0, 1.0], [0, 1], n_bins=10)
    assert bins[0]["count"] == 1 and bins[9]["count"] == 1
    assert expected_calibration_error([0.0, 1.0], [0, 1], n_bins=10) == 0.0

    for bad_call in (
        lambda: expected_calibration_error([0.5], [1, 0]),
        lambda: expected_calibration_error([0.5], [1], n_bins=0),
        lambda: expected_calibration_error([float("nan")], [1]),
        lambda: expected_calibration_error([True], [1]),
        lambda: expected_calibration_error([0.5], [2]),
        lambda: confidence_separation_summary([True], [0.5]),
    ):
        try:
            bad_call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_confidence_separation_summary():
    summary = confidence_separation_summary([0.96, 0.88, 0.65], [0.95, 0.72, 0.60])
    assert summary["tp_count"] == 3 and summary["fp_count"] == 3
    assert summary["mean_tp"] == round((0.96 + 0.88 + 0.65) / 3.0, 6)
    assert summary["mean_fp"] == round((0.95 + 0.72 + 0.60) / 3.0, 6)
    assert summary["mean_tp_gt_mean_fp"] is True
    assert summary["fp_high_conf_count"] == 1
    assert summary["fp_high_conf_rate"] == round(1.0 / 3.0, 6)
    assert summary["low_conf_count"] == 2
    assert summary["tp_low_conf_count"] == 1
    assert summary["fp_low_conf_count"] == 1

    inverted = confidence_separation_summary([0.60], [0.90])
    assert inverted["mean_tp_gt_mean_fp"] is False

    no_fp = confidence_separation_summary([0.90], [])
    assert no_fp["mean_fp"] is None
    assert no_fp["mean_tp_gt_mean_fp"] is None
    assert no_fp["fp_high_conf_count"] == 0 and no_fp["fp_high_conf_rate"] == 0.0


def test_custom_gt_loaders_fallback_and_schema_validation():
    assert load_router_gt("dev") == []
    assert load_autosar_diagram_gt("dev") == []
    assert load_autosar_table_gt("dev") == []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        missing = tmp_path / "missing.json"
        assert load_router_gt("dev", gt_path=missing, missing_ok=True) == []
        try:
            load_router_gt("dev", gt_path=missing)
            raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError:
            pass

        router_file = tmp_path / "router.json"
        router_file.write_text(
            json.dumps([
                {
                    "doc": "DocA.pdf",
                    "page": 1,
                    "kind": "native",
                    "suspect_ocr_layer": False,
                    "has_partial_raster": False,
                    "split": "dev",
                },
                {
                    "doc": "DocB",
                    "page": 2,
                    "kind": "hybrid",
                    "suspect_ocr_layer": True,
                    "has_partial_raster": True,
                    "split": "test",
                },
            ]),
            encoding="utf-8",
        )
        r_dev = load_router_gt("dev", gt_path=router_file, pdf_dir=tmp_path)
        assert len(r_dev) == 1 and r_dev[0]["pdf_path"] == tmp_path / "DocA.pdf"
        assert len(load_router_gt("test", gt_path=router_file, pdf_dir=tmp_path)) == 1
        assert validate_router_gt([{"doc": "A", "page": 0, "kind": "bad", "suspect_ocr_layer": "no", "has_partial_raster": False}])

        diag_file = tmp_path / "diagrams.json"
        diag_file.write_text(
            json.dumps([
                {
                    "doc": "AUTOSAR_CP_EXP_LayeredSoftwareArchitecture.pdf",
                    "page": 12,
                    "bbox_pt": [40, 80, 400, 320],
                    "nodes": [
                        {"id": "n1", "bbox_pt": [50, 90, 150, 140], "label": "App"},
                        {"id": "n2", "bbox_pt": [200, 90, 300, 140], "label": "RTE"},
                    ],
                    "edges": [
                        {"source_id": "n1", "target_id": "n2", "direction": "directed", "label": "call"}
                    ],
                    "groups": [
                        {"id": "g1", "bbox_pt": [45, 85, 310, 150], "member_ids": ["n1", "n2"]}
                    ],
                    "split": "dev",
                }
            ]),
            encoding="utf-8",
        )
        d_dev = load_autosar_diagram_gt("dev", gt_path=diag_file)
        assert len(d_dev) == 1 and d_dev[0]["bbox_pt"] == [40.0, 80.0, 400.0, 320.0]
        bad_diag = [
            {
                "doc": "Doc",
                "page": 1,
                "bbox_pt": [0, 0, 100, 100],
                "nodes": [{"id": "n1", "bbox_pt": [10, 10, 20, 20], "label": "A"}],
                "edges": [{"source_id": "n1", "target_id": "missing", "direction": "directed", "label": ""}],
                "groups": [],
            }
        ]
        assert validate_autosar_diagram_gt(bad_diag)

        tbl_file = tmp_path / "tables.json"
        tbl_file.write_text(
            json.dumps([
                {
                    "doc": "AUTOSAR_CP_EXP_VFB.pdf",
                    "page": 17,
                    "bbox_pt": [50, 100, 500, 300],
                    "html": "<table><tr><td>A</td></tr></table>",
                    "has_merged": False,
                    "split": "dev",
                }
            ]),
            encoding="utf-8",
        )
        t_dev = load_autosar_table_gt("dev", gt_path=tbl_file)
        assert len(t_dev) == 1 and t_dev[0]["has_merged"] is False
        assert validate_autosar_table_gt([{"doc": "D", "page": 1, "bbox_pt": [10, 10, 5, 20], "html": "plain", "has_merged": 1}])

        bad_file = tmp_path / "bad_tbl.json"
        bad_file.write_text(json.dumps([{"doc": ""}]), encoding="utf-8")
        try:
            load_autosar_table_gt("dev", gt_path=bad_file)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_calibration_split_guard_and_cli_logging():
    assert check_split_guard("dev", False) is True
    assert check_split_guard("test", True) is True
    assert calibration_main(["--split", "test"]) == 2
    assert calibration_main(["--task", "figures", "--split", "test"]) == 2

    from unittest.mock import patch
    from pdf.eval import calibration

    with tempfile.TemporaryDirectory() as tmp:
        log_file = Path(tmp) / "eval_runs.jsonl"
        with (
            patch.object(calibration, "eval_figures_calibration", return_value={"task": "figures_calibration", "ece_10": 0.05}),
            patch.object(calibration, "eval_tables_calibration", return_value={"task": "tables_calibration", "ece_10": 0.04}),
            patch.object(calibration, "eval_monotonicity_invariants", return_value={"task": "monotonicity_invariants", "invariant_ok": True}),
        ):
            rc = calibration_main(["--task", "all", "--split", "dev", "--log-file", str(log_file)])
        assert rc == 0
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["task"] == "calibration:all"
        assert rec["split"] == "dev"
        assert len(rec["results"]) == 3


def test_eval_figures_calibration_unit():
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmp:
        gt_path = Path(tmp) / "gt.json"
        gt_path.write_text(
            json.dumps({
                "documents": {
                    "DocA": {
                        "dev_pages": [1],
                        "dev_figures": {"1": [{"bbox": [10, 10, 110, 110], "flags": []}]},
                        "test_pages": [],
                        "figures": {},
                    }
                }
            }),
            encoding="utf-8",
        )
        fake_preds = {
            1: [
                FigureBox((10.0, 10.0, 110.0, 110.0), "figure", "v1_group", 0.90),
                FigureBox((200.0, 200.0, 300.0, 300.0), "figure", "v1_group", 0.60),
            ]
        }
        assert matched_pred_indices([list(b.bbox) for b in fake_preds[1]], [[10.0, 10.0, 110.0, 110.0]]) == {0}
        with (
            patch("pdf.backends.get", return_value=object()),
            patch("pdf.figures.detect_figures", return_value=fake_preds),
        ):
            res = eval_figures_calibration(split="dev", gt_path=gt_path)
        assert res["tp"] == 1 and res["fp"] == 1 and res["fn"] == 0
        assert res["mean_conf_tp"] == 0.9 and res["mean_conf_fp"] == 0.6
        assert res["mean_tp_gt_mean_fp"] is True
        assert res["fp_ge_095"] == 0
        assert res["low_conf_lt_070"] == 1
        assert abs(res["ece_10"] - 0.35) < 1e-6


def test_eval_calibration_init_no_heavy_ml_imports():
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import sys, pdf.eval.calibration; "
                "bad = [m for m in ('torch', 'transformers', 'onnxruntime', 'docling') if m in sys.modules]; "
                "assert not bad, bad"
            ),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.returncode == 0


def test_table_source_monotonicity_rejects_auto_refined_below_auto_tatr():
    from types import SimpleNamespace
    from unittest.mock import patch
    from pdf.eval import calibration

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")
        gt_path = tmp_path / "tables_gt.json"
        gt_path.write_text(
            json.dumps([{
                "doc": "doc.pdf",
                "page": 1,
                "bbox_pt": [10, 10, 200, 200],
                "html": "<table><tr><td>A</td></tr></table>",
                "has_merged": False,
                "split": "dev",
            }]),
            encoding="utf-8",
        )

        class _DummyDoc:
            pages = [object()]

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        pl_tbl = SimpleNamespace(num_rows=1, num_cols=1, confidence=1.0, source="pdfplumber")
        tt_tbl = SimpleNamespace(num_rows=1, num_cols=1, confidence=0.80, source="tatr_onnx")
        au_tbl = SimpleNamespace(num_rows=1, num_cols=1, confidence=0.85, source="auto_refined")

        def _fake_extract(page, boxes, *, words=(), extractor=None, device="auto"):
            return {"pl": [pl_tbl], "tt": [tt_tbl], "au": [au_tbl]}[extractor]

        with (
            patch.object(calibration, "load_autosar_table_gt", side_effect=lambda **kw: load_autosar_table_gt(pdf_dir=tmp_path, **kw)),
            patch("pdfplumber.open", return_value=_DummyDoc()),
            patch("pdf.text.extract_words", return_value=[]),
            patch("pdf.backends.get", side_effect=lambda kind, name, **kw: {"pdfplumber": "pl", "tatr-onnx": "tt", "auto": "au"}[name]),
            patch("pdf.tables.extract_tables_on_page", side_effect=_fake_extract),
        ):
            res = calibration._check_table_source_monotonicity(
                split="dev",
                device="cpu",
                layout="heron-onnx",
                table_gt_path=gt_path,
                e2e_auto_tatr_confs=[0.90],
            )
        assert res["paired_monotonicity_ok"] is True
        assert res["source_mean_monotonicity_ok"] is False


def test_eval_monotonicity_invariants_unit():
    from unittest.mock import patch
    from pdf.eval import calibration

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")
        router_gt_path = tmp_path / "router.json"
        router_gt_path.write_text(
            json.dumps([
                {"doc": "doc.pdf", "page": 1, "kind": "native", "suspect_ocr_layer": False, "has_partial_raster": False, "split": "dev"},
                {"doc": "doc.pdf", "page": 2, "kind": "hybrid", "suspect_ocr_layer": True, "has_partial_raster": False, "split": "dev"},
            ]),
            encoding="utf-8",
        )

        good_ir = {
            "metadata": {"warnings": [{"page": 2, "code": "suspect_ocr_layer"}]},
            "elements": [
                {"type": "text", "page": 1, "confidence": 1.0},
                {"type": "text", "page": 2, "confidence": 0.65},
            ],
        }
        bad_ir = {
            "metadata": {"warnings": [{"page": 2, "code": "suspect_ocr_layer"}]},
            "elements": [
                {"type": "text", "page": 1, "confidence": 1.0},
                {"type": "text", "page": 2, "confidence": 0.98},
            ],
        }

        with (
            patch.object(calibration, "load_router_gt", side_effect=lambda **kw: load_router_gt(pdf_dir=tmp_path, **kw)),
            patch("pdf.extract_ir", return_value=good_ir),
        ):
            ok_res = calibration.eval_monotonicity_invariants(split="dev", router_gt_path=router_gt_path)
        assert ok_res["invariant_ok"] is True
        assert ok_res["low_conf_pages_with_warning"] == 1

        with (
            patch.object(calibration, "load_router_gt", side_effect=lambda **kw: load_router_gt(pdf_dir=tmp_path, **kw)),
            patch("pdf.extract_ir", return_value=bad_ir),
        ):
            bad_res = calibration.eval_monotonicity_invariants(split="dev", router_gt_path=router_gt_path)
        assert bad_res["suspect_ocr_all_le_085"] is False
        assert bad_res["invariant_ok"] is False


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
