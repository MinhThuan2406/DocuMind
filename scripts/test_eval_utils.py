"""
Small sanity tests for eval_utils and run_e2e_reading_order_eval.

Run:  .venv\\Scripts\\python scripts\\test_eval_utils.py
(plain asserts, no pytest needed; functions are also pytest-compatible)
"""

import random
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from eval_utils import (  # noqa: E402
    boxes_covering_gts,
    classify_gt_completeness,
    compute_kendall_tau,
    detection_prf,
    docling_bbox_to_pixel,
    fragmented_gt_count,
    gt_elements_from_page,
    iou_matrix,
    match_one_to_one,
    over_merge_count,
    piece_containment,
    split_count,
    whole_capture,
)
from ir_postprocessor import compute_iou  # noqa: E402
from diagram_grouping_v2 import GroupParamsV2, group_whole_diagrams_image_v2  # noqa: E402
from run_e2e_reading_order_eval import (  # noqa: E402
    build_comparison,
    cache_dir_for,
    comparison_markdown,
    describe_docling,
    evaluate_page_ok,
    iter_doc_items,
    score_method,
    stratified_sample,
)


def _approx(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) < tol


# ----------------------------------------------------------------------
# Kendall's Tau
# ----------------------------------------------------------------------
def test_kendall_undefined_for_fewer_than_two_pairs():
    assert compute_kendall_tau([], []) is None
    assert compute_kendall_tau([0], [0]) is None
    # None entries are dropped, leaving only one valid pair
    assert compute_kendall_tau([0, None], [0, 1]) is None


def test_kendall_identical_and_reversed():
    assert compute_kendall_tau([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0
    assert compute_kendall_tau([0, 1, 2, 3], [3, 2, 1, 0]) == -1.0


def test_kendall_ties_count_zero():
    # pairs: (0,1) tie in a -> 0; (0,2) concordant; (1,2) concordant -> (2-0)/3
    assert _approx(compute_kendall_tau([0, 0, 1], [0, 1, 2]), 2 / 3)


# ----------------------------------------------------------------------
# Geometry and matching
# ----------------------------------------------------------------------
def test_iou_matrix_matches_scalar_compute_iou():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 100, size=(20, 4))
    a[:, 2:] = a[:, :2] + rng.uniform(1, 50, size=(20, 2))
    b = rng.uniform(0, 100, size=(15, 4))
    b[:, 2:] = b[:, :2] + rng.uniform(1, 50, size=(15, 2))
    m = iou_matrix(a, b)
    for i in range(20):
        for j in range(15):
            assert abs(m[i, j] - compute_iou(list(a[i]), list(b[j]))) < 1e-9


def test_docling_bbox_scaling():
    # page units (points) -> image pixels; 96-dpi image gives scale 4/3 horizontally
    out = docling_bbox_to_pixel([10, 20, 30, 40], 4 / 3, 2.0)
    assert out == [13.33, 40.0, 40.0, 80.0], out


def test_match_one_to_one_is_one_to_one_and_thresholded():
    gt = [[0, 0, 100, 100], [200, 0, 300, 100]]
    # two predictions overlap GT 0; only the better one may match it
    pred = [[0, 0, 100, 100], [5, 5, 105, 105], [400, 400, 500, 500]]
    g2p, p2g = match_one_to_one(pred, gt, iou_thr=0.3)
    assert g2p == {0: 0}, g2p
    assert p2g == {0: 0}, p2g
    # IoU below threshold is never matched
    g2p, _ = match_one_to_one([[50, 0, 150, 100]], gt, iou_thr=0.5)  # IoU ~ 0.33 with GT 0
    assert g2p == {}, g2p


# ----------------------------------------------------------------------
# Penalized vs matched Tau
# ----------------------------------------------------------------------
def test_penalized_tau_penalises_missed_box():
    # GT orders 0..3 in a vertical stack; Docling misses GT order 1.
    gt_items = [{"id": f"g{k}", "order": k, "bbox": [0, 100 * k, 50, 100 * k + 50]} for k in range(4)]
    gt_to_pred = {0: 0, 2: 1, 3: 2}      # predicted indices 0,1,2 match GT 0,2,3 in correct order
    pos_pred = {0: 0, 1: 1, 2: 2}        # predicted reading order = correct order
    res = score_method(gt_items, gt_to_pred, pos_pred)
    assert res["n_matched"] == 3
    assert _approx(res["coverage"], 0.75)
    assert _approx(res["tau_matched"], 1.0)
    # Hand-computed: C=4, D=2 over 6 pairs -> (4-2)/6
    assert _approx(res["tau_penalized"], 1 / 3), res["tau_penalized"]


def test_score_method_undefined_for_single_gt():
    gt_items = [{"id": "g0", "order": 0, "bbox": [0, 0, 1, 1]}]
    res = score_method(gt_items, {}, {})
    assert res["tau_penalized"] is None and res["tau_matched"] is None
    assert res["coverage"] == 0.0


def test_gt_filter_matches_tier1_rules():
    page = {"layout_dets": [
        {"category_type": "text_block", "poly": [0, 0, 10, 0, 10, 10, 0, 10], "order": 0},
        {"category_type": "text_block", "poly": [0, 0, 10, 0, 10, 10, 0, 10], "order": 1, "ignore": True},
        {"category_type": "text_block", "poly": [0, 0, 10, 0, 10, 10, 0, 10], "order": None},
        {"category_type": "abandon", "poly": [0, 0, 10, 0, 10, 10, 0, 10], "order": 2},
        {"category_type": "table", "bbox": [0, 0, 20, 20], "order": 3},
        {"category_type": "text_block", "poly": [], "order": 4},
    ]}
    gt = gt_elements_from_page(page)
    assert [g["order"] for g in gt] == [0, 3], gt
    assert gt[0]["bbox"] == [0.0, 0.0, 10.0, 10.0]


# ----------------------------------------------------------------------
# Page-level wiring (synthetic cache, no Docling)
# ----------------------------------------------------------------------
def test_evaluate_page_orders_methods_correctly():
    # Three stacked text boxes. Docling iterate order is reversed (bottom first).
    item = {"page_info": {"image_path": "x.png", "width": 500, "height": 500,
                          "page_attribute": {"layout": "single_column", "data_source": "t", "special_issue": ["None"]}},
            "layout_dets": []}
    gt_items = [
        {"id": f"gt_{k}", "type": "text_block", "order": k, "bbox": [50, 50 + 100 * k, 300, 100 + 100 * k]}
        for k in range(3)
    ]
    cache = {"status": "ok", "docling_time_sec": 1.0, "aspect_mismatch": False, "items": [
        {"order": 0, "label": "text", "bbox": [50, 250, 300, 300], "text": ""},
        {"order": 1, "label": "text", "bbox": [50, 150, 300, 200], "text": ""},
        {"order": 2, "label": "text", "bbox": [50, 50, 300, 100], "text": ""},
    ]}
    rec = evaluate_page_ok(0, item, cache, gt_items, iou_thr=0.3)
    assert rec["methods"]["A"]["tau_penalized"] == -1.0, rec["methods"]["A"]
    assert rec["methods"]["B"]["tau_penalized"] == 1.0, rec["methods"]["B"]
    assert rec["methods"]["C"]["tau_penalized"] == 1.0, rec["methods"]["C"]
    assert rec["methods"]["D"]["tau_penalized"] == 1.0, rec["methods"]["D"]
    assert rec["methods"]["D"]["coverage"] == 1.0
    assert rec["special_issue"] == []  # 'None' token is not a real issue


# ----------------------------------------------------------------------
# Stratified sampling
# ----------------------------------------------------------------------
def test_stratified_sample_counts_and_determinism():
    rng = random.Random(1)
    indexed = []
    for i in range(400):
        layout = rng.choice(["single_column", "double_column", "three_column"])
        src = rng.choice(["book", "note", "newspaper", "rare"] if i % 97 else ["rare"])
        indexed.append((i, {"page_info": {"page_attribute": {"layout": layout, "data_source": src}}}))
    n_strata = len({(it["page_info"]["page_attribute"]["layout"], it["page_info"]["page_attribute"]["data_source"])
                    for _, it in indexed})
    s1 = stratified_sample(indexed, 60, seed=42)
    s2 = stratified_sample(indexed, 60, seed=42)
    assert len(s1) == 60, len(s1)
    assert [i for i, _ in s1] == [i for i, _ in s2]           # deterministic
    assert len({i for i, _ in s1}) == 60                       # no duplicates
    covered = {(it["page_info"]["page_attribute"]["layout"], it["page_info"]["page_attribute"]["data_source"])
               for _, it in s1}
    assert len(covered) == n_strata                            # every stratum represented
    s3 = stratified_sample(indexed, 60, seed=7)
    assert [i for i, _ in s3] != [i for i, _ in s1]


def test_detection_perfect_match():
    gt = [[0, 0, 100, 100], [200, 200, 300, 300]]
    r = detection_prf(gt, gt, 0.5)
    assert r["tp"] == 2 and _approx(r["precision"], 1.0) and _approx(r["recall"], 1.0) and _approx(r["f1"], 1.0)
    assert _approx(r["mean_iou_matched"], 1.0)


def test_detection_missed_and_extra_box():
    gt = [[0, 0, 100, 100], [200, 200, 300, 300]]
    pred = [[0, 0, 100, 100], [500, 500, 600, 600]]  # one correct, one spurious
    r = detection_prf(pred, gt, 0.5)
    assert r["tp"] == 1
    assert _approx(r["precision"], 0.5) and _approx(r["recall"], 0.5) and _approx(r["f1"], 0.5)


def test_detection_no_predictions_is_zero_not_error():
    gt = [[0, 0, 100, 100]]
    r = detection_prf([], gt, 0.5)
    assert r["precision"] == 0.0 and r["recall"] == 0.0 and r["f1"] == 0.0 and r["mean_iou_matched"] is None


def test_fragmentation_two_partial_pieces():
    gt_fig = [[0, 0, 100, 100]]
    # each piece overlaps the GT with IoU 0.4 (in [0.1, 0.5)) -> one fragmented figure
    assert fragmented_gt_count([[0, 0, 40, 100], [60, 0, 100, 100]], gt_fig) == 1
    # a single near-perfect box is not fragmented
    assert fragmented_gt_count([[0, 0, 100, 95]], gt_fig) == 0


def test_over_merge_figures_and_text():
    figs = [[10, 10, 90, 90], [110, 10, 190, 90]]
    texts = [[10, 100, 190, 120]]
    # one box that swallows both figures -> over-merge
    assert over_merge_count([[0, 0, 200, 130]], figs, [])["over_merge_boxes"] == 1
    # one box that swallows two text blocks -> over-merge
    texts2 = [[10, 100, 90, 120], [110, 100, 190, 120]]
    assert over_merge_count([[0, 90, 200, 130]], [], texts2)["over_merge_boxes"] == 1
    # a box covering a single figure and a text block only partially (<50% of its area) -> not over-merged
    res = over_merge_count([[0, 0, 100, 105]], figs[:1], [[10, 100, 90, 130]])
    assert res["over_merge_boxes"] == 0 and res["text_blocks_absorbed"] == 0


def test_piece_containment_is_share_of_piece_inside():
    assert _approx(piece_containment([10, 10, 20, 20], [0, 0, 100, 100]), 1.0)
    # half of the piece lies outside the container
    assert _approx(piece_containment([0, 0, 100, 100], [50, 0, 200, 100]), 0.5)
    assert _approx(piece_containment([500, 500, 600, 600], [0, 0, 100, 100]), 0.0)


def test_classify_intact_broken_missed_other():
    gt = [0, 0, 100, 100]
    assert classify_gt_completeness(gt, [[0, 0, 100, 100]]) == ("intact", 1)
    # two partial pieces, each IoU 0.4 (no whole match), both fully inside the GT -> broken
    assert classify_gt_completeness(gt, [[0, 0, 40, 100], [60, 0, 100, 100]]) == ("broken", 2)
    # nothing near the GT -> missed
    assert classify_gt_completeness(gt, [[500, 500, 600, 600]]) == ("missed", 0)
    # one partial piece only -> other
    assert classify_gt_completeness(gt, [[0, 0, 40, 100]]) == ("other", 1)


def test_whole_capture_requires_exactly_one_match():
    gt = [0, 0, 100, 100]
    assert whole_capture(gt, [[0, 0, 100, 100]], 0.5) is True
    # second box also reaches IoU 0.81 >= 0.5 -> two matches, not whole
    assert whole_capture(gt, [[0, 0, 100, 100], [5, 5, 95, 95]], 0.5) is False
    # at a stricter threshold only the exact box matches
    assert whole_capture(gt, [[0, 0, 100, 100], [5, 5, 95, 95]], 0.9) is True
    assert whole_capture(gt, [], 0.5) is False


def test_split_counts_pieces_inside_gt():
    gt = [0, 0, 100, 100]
    preds = [[0, 0, 40, 100], [60, 0, 100, 100], [500, 500, 600, 600]]
    assert split_count(gt, preds, 0.5) == 2
    assert split_count(gt, [[0, 0, 100, 100]], 0.5) == 1


def test_boxes_covering_gts_uses_gt_area_coverage():
    gts = [[10, 10, 90, 90], [150, 150, 190, 190], [300, 300, 400, 400]]
    covering = boxes_covering_gts([[0, 0, 200, 200]], gts, 0.5)
    assert covering == [[0, 1]], covering


# ----------------------------------------------------------------------
# FURNITURE option (content layers), cache naming, before/after comparison
# ----------------------------------------------------------------------
def test_cache_dir_naming_keeps_legacy_and_separates_furniture():
    root = Path("out")
    assert cache_dir_for(root, "auto", "cpu") == root / "cache"
    assert cache_dir_for(root, "auto", "cpu", False) == root / "cache"
    torch_body = cache_dir_for(root, "torch", "cuda")
    torch_full = cache_dir_for(root, "torch", "cuda", True)
    assert torch_body == root / "cache_ocr-torch_dev-cuda", torch_body
    assert torch_full == root / "cache_ocr-torch_dev-cuda_furniture", torch_full
    assert cache_dir_for(root, "auto", "cpu", True) == root / "cache_ocr-auto_dev-cpu_furniture"
    assert len({root / "cache", torch_body, torch_full}) == 3


def test_iter_doc_items_passes_content_layers_only_when_requested():
    from docling_core.types.doc import ContentLayer

    class FakeDoc:
        def __init__(self):
            self.kwargs = None

        def iterate_items(self, **kwargs):
            self.kwargs = kwargs
            return iter([("a", 0)])

    d = FakeDoc()
    assert list(iter_doc_items(d, False)) == [("a", 0)]
    assert d.kwargs == {}, d.kwargs  # legacy call: no argument at all, same as before
    list(iter_doc_items(d, True))
    assert d.kwargs == {"included_content_layers": set(ContentLayer)}, d.kwargs


def test_furniture_predictions_are_counted_and_matched_when_present():
    item = {"page_info": {"image_path": "x.png", "width": 500, "height": 500,
                          "page_attribute": {"layout": "single_column", "data_source": "t", "special_issue": ["None"]}}}
    gt_items = [
        {"id": "gt_0", "type": "header", "order": 0, "bbox": [50, 10, 300, 40]},
        {"id": "gt_1", "type": "text_block", "order": 1, "bbox": [50, 100, 300, 200]},
    ]
    with_furniture = {"status": "ok", "docling_time_sec": 1.0, "aspect_mismatch": False, "items": [
        {"order": 0, "label": "page_header", "bbox": [50, 10, 300, 40], "content_layer": "furniture", "text": ""},
        {"order": 1, "label": "text", "bbox": [50, 100, 300, 200], "content_layer": "body", "text": ""},
    ]}
    rec = evaluate_page_ok(0, item, with_furniture, gt_items, iou_thr=0.3)
    assert rec["n_pred"] == 2 and rec["n_furniture_pred"] == 1, rec
    assert rec["gt_by_type"]["header"] == [1, 1], rec["gt_by_type"]
    assert rec["methods"]["A"]["coverage"] == 1.0
    assert rec["methods"]["A"]["tau_penalized"] == 1.0

    body_only = {"status": "ok", "docling_time_sec": 1.0, "aspect_mismatch": False, "items": [
        {"order": 0, "label": "text", "bbox": [50, 100, 300, 200], "text": ""},
    ]}
    rec0 = evaluate_page_ok(0, item, body_only, gt_items, iou_thr=0.3)
    assert rec0["n_furniture_pred"] == 0 and rec0["n_pred"] == 1, rec0
    assert rec0["gt_by_type"]["header"] == [1, 0], rec0["gt_by_type"]


def _mini_run(a_tau, b_tau, cov, n_matched, pages):
    summary = {m: {"tau_penalized": {"mean": a_tau if m == "A" else b_tau},
                   "tau_matched": {"mean": None}, "coverage": {"mean": cov}} for m in "ABCD"}
    pw = {"n_compared": 2, "wins": 1, "ties": 0, "losses": 1, "mean_delta": 0.1, "median_delta": 0.1,
          "bootstrap_ci95_mean_delta": None}
    return {
        "config": {"ocr_backend": "torch", "device": "cuda", "device_resolved_observed": ["cuda:0"]},
        "summary": summary,
        "pairwise": {"B_vs_A": {"tau_penalized": pw, "tau_matched": pw}},
        "breakdown": {"layout": {"single_column": {"n_pages": 2, "methods": {
            "A": {"tau_penalized_mean": a_tau}, "B": {"tau_penalized_mean": b_tau}}}}},
        "coverage_by_gt_type": {"header": {"n_gt": 4, "n_matched": n_matched, "coverage": round(n_matched / 4, 4)}},
        "per_page": [{"image_path": p, "status": "ok", "device_resolved": "cuda:0"} for p in pages],
    }


def test_build_comparison_reports_before_after_deltas():
    base = _mini_run(a_tau=0.5, b_tau=0.6, cov=0.7, n_matched=1, pages=["p1.png", "p2.png"])
    new = _mini_run(a_tau=0.5, b_tau=0.9, cov=0.8, n_matched=3, pages=["p2.png", "p1.png"])
    cmp = build_comparison(new, base)
    assert cmp["same_page_set"] is True and cmp["baseline_config_match"] is True, cmp
    assert abs(cmp["layout_B_minus_A"]["single_column"]["delta_baseline"] - 0.1) < 1e-9, cmp
    assert abs(cmp["layout_B_minus_A"]["single_column"]["delta_new"] - 0.4) < 1e-9, cmp
    assert cmp["coverage_by_gt_type"]["header"]["matched_baseline"] == 1
    assert cmp["coverage_by_gt_type"]["header"]["matched_new"] == 3
    assert any(line.startswith("### 7.1") for line in comparison_markdown(cmp))
    other_page = _mini_run(a_tau=0.5, b_tau=0.6, cov=0.7, n_matched=1, pages=["p9.png"])
    assert build_comparison(new, other_page)["same_page_set"] is False


def test_describe_docling_reports_actual_settings_not_defaults():
    res = {"config": {"ocr_backend": "torch", "device": "cuda", "device_resolved_observed": ["cuda:0"]},
           "per_page": []}
    text = describe_docling(res)
    assert "torch" in text and "cuda:0" in text and "BODY" in text and "CPU" not in text, text
    res["config"]["include_furniture"] = True
    assert "FURNITURE" in describe_docling(res)
    legacy = {"config": {"ocr_backend": "auto", "device": "cpu"},
              "per_page": [{"device_resolved": "cpu"}]}
    assert "cpu" in describe_docling(legacy)


# ----------------------------------------------------------------------
# Diagram grouping v2 (diagram_grouping_v2.group_whole_diagrams_image_v2), explicit containers
# ----------------------------------------------------------------------
def _pic(b):
    return {"label": "picture", "bbox": list(b)}


def _grp(items, W=1000, H=1000, containers=(), params=None):
    return group_whole_diagrams_image_v2(items, (W, H), containers=list(containers),
                                         params=params or GroupParamsV2())


def test_v2_merges_aligned_broken_pieces_into_one_box():
    res = _grp([_pic([100, 100, 300, 250]), _pic([320, 100, 520, 250])])
    assert len(res["diagram_groups"]) == 1 and res["standalone"] == [], res
    box = res["diagram_groups"][0]["bbox"]
    assert box[0] <= 100 and box[1] <= 100 and box[2] >= 520 and box[3] >= 250, box
    assert compute_iou(box, [100, 100, 520, 250]) > 0.9, box


def test_v2_single_picture_is_left_unchanged():
    res = _grp([_pic([100, 100, 300, 250])])
    assert res["diagram_groups"] == [], res
    assert res["standalone"] == [[100.0, 100.0, 300.0, 250.0]], res["standalone"]


def test_v2_keeps_far_figures_separate():
    res = _grp([_pic([100, 100, 300, 250]), _pic([900, 100, 1000, 250])])
    assert res["diagram_groups"] == [], res
    assert len(res["standalone"]) == 2, res["standalone"]


def test_v2_formula_between_pictures_blocks_merge():
    items = [_pic([100, 100, 300, 250]), _pic([320, 100, 520, 250]),
             {"label": "formula", "bbox": [200, 120, 420, 240]}]
    res = _grp(items)
    assert res["diagram_groups"] == [], res
    assert len(res["standalone"]) == 2, res["standalone"]


def test_v2_container_evidence_needs_picture_only_container():
    a, b = [100, 100, 300, 250], [450, 100, 650, 250]   # gap 150 px: only container evidence can merge
    clean = [[90, 90, 660, 260]]                         # area 96900 <= 1.35 x union 82500
    assert len(_grp([_pic(a), _pic(b)], containers=clean)["diagram_groups"]) == 1
    with_formula = [{"label": "formula", "bbox": [300, 252, 340, 258]}]  # inside container, outside union
    res = _grp([_pic(a), _pic(b)] + with_formula, containers=clean)
    assert res["diagram_groups"] == [], res


def test_v2_big_pieces_must_nearly_touch():
    big_apart = [_pic([50, 100, 350, 400]), _pic([400, 100, 700, 400])]    # each 9 % of page, gap 50 px
    assert _grp(big_apart)["diagram_groups"] == [], "big apart pieces must stay separate"
    big_touch = [_pic([50, 100, 350, 400]), _pic([360, 100, 660, 400])]    # gap 10 px
    assert len(_grp(big_touch)["diagram_groups"]) == 1


def test_v2_group_union_limited_by_page_share():
    tiny = GroupParamsV2(max_group_frac=0.05)
    res = _grp([_pic([100, 100, 300, 250]), _pic([320, 100, 520, 250])], params=tiny)
    assert res["diagram_groups"] == [], res


# ----------------------------------------------------------------------
# Image reading for container detection: non-ASCII paths and bad files
# ----------------------------------------------------------------------
def _write_frame_png(path):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (1000, 800), (255, 255, 255))
    ImageDraw.Draw(img).rectangle([150, 150, 650, 500], outline=(0, 0, 0), width=6)
    img.save(str(path))


def test_container_detection_same_for_cjk_and_ascii_paths():
    import tempfile
    from test_diagram_grouping import detect_visual_containers_opencv
    with tempfile.TemporaryDirectory() as tmp:
        cjk = Path(tmp) / "测试图像"
        cjk.mkdir()
        plain = Path(tmp) / "plain"
        plain.mkdir()
        _write_frame_png(cjk / "图_1.png")
        _write_frame_png(plain / "frame_1.png")
        a = detect_visual_containers_opencv(str(cjk / "图_1.png"))
        b = detect_visual_containers_opencv(str(plain / "frame_1.png"))
        assert len(b) >= 1, b
        assert a == b, (a, b)


def test_unreadable_or_empty_image_raises_clear_error():
    import tempfile
    from test_diagram_grouping import detect_visual_containers_opencv
    cases = {"garbage.png": b"not an image at all", "empty.png": b""}
    with tempfile.TemporaryDirectory() as tmp:
        for name, payload in cases.items():
            f = Path(tmp) / name
            f.write_bytes(payload)
            try:
                detect_visual_containers_opencv(str(f))
            except ValueError as e:
                assert "image" in str(e), str(e)
            else:
                raise AssertionError(f"{name}: expected ValueError, got a result")



def main():
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # report and continue
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
