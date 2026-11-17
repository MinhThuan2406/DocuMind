import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdf.eval import char_counts, load_autosar_gt, match_count, prf, strip_markup  # noqa: E402
from pdf.eval.compare import main as compare_main  # noqa: E402


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


def test_strip_markup_preserves_math_inequalities():
    sample = r"Let $a < x + 1$ and $x > 0$. <table><tr><td>42</td></tr></table> \frac{a}{b}"
    cleaned = strip_markup(sample)
    assert "a < x + 1" in cleaned
    assert "x > 0" in cleaned
    assert "<table>" not in cleaned and "<td>" not in cleaned
    assert "42" in cleaned


def test_match_count_strict_and_lenient():
    gt = [[0.0, 0.0, 100.0, 100.0]]
    loose_pred = [[0.0, 0.0, 140.0, 140.0]]  # area = 1.96x gt, iou = 10000/19600 = 0.510
    wide_pred = [[0.0, 0.0, 142.0, 142.0]]   # area = 2.016x gt, iou = 10000/20164 = 0.496
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
    rc = compare_main(["--task", "figures", "--split", "test", "--backends", "heron"])
    assert rc == 2


def test_char_counts_whitespace_normalized():
    assert char_counts("A B\nC") == char_counts("ABC")


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
