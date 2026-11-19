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
from pdf.eval.compare import main as compare_main  # noqa: E402
from pdf.eval.compare_hierarchy import main as hierarchy_main  # noqa: E402
from pdf.eval.reading_order import bench_synthetic_reading_order, main as reading_order_main  # noqa: E402
from pdf.eval.tables import main as tables_main  # noqa: E402
from pdf.models import TableCell  # noqa: E402


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


def test_synthetic_reading_order_fixtures():
    res = bench_synthetic_reading_order()
    assert res["pages"] == 3
    assert res["mean_tau_matched"] == 1.0
    assert res["mean_tau_penalized"] == 1.0


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
