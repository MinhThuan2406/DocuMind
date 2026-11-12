"""
Diagram / figure cluster capture evaluation (v2): whole-capture, split and merge rates.

Scope: every OmniDocBench page with >=1 GT 'figure' (ignore=False), OCR OFF (cache_ocr-off_dev-cuda).

Per GT figure G (from the raw Docling picture boxes, R1):
  intact  : some raw box has IoU >= 0.5 with G
  broken  : no such box, and >= 2 raw boxes have containment >= 0.7 inside G
  missed  : no such box, and no raw box has containment >= 0.7 inside G
  other   : no such box, exactly one raw box is contained (single partial piece)
  variant : same, but the pieces also include formula/table boxes

Regions:
  R1 = raw Docling picture boxes (OCR off)
  R3 = group_whole_diagrams_image() output (OCR off), called with docling_page_size = image_size

Metrics per region and GT subset (broken / intact / all):
  whole-capture : share of GT with exactly one predicted box of IoU >= t (t = 0.3, 0.5, 0.7)
  split         : share of GT with >= 2 predicted boxes that have >= 0.5 of their own area inside G
  merge (subset): among predicted boxes that cover >= 0.5 of at least one GT in the subset,
                  share that cover >= 0.5 of >= 2 distinct GT figures
  merge (global): same over all predicted boxes, and over boxes covering >= 1 GT figure

Near distinct figures: pairs of non-overlapping GT figures with edge gap <= 80 px on the same page.

Usage:
  .venv\\Scripts\\python scripts\\run_diagram_crop_eval_v2.py
  .venv\\Scripts\\python scripts\\run_diagram_crop_eval_v2.py --limit 10
"""

import argparse
import json
import math
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from eval_utils import (  # noqa: E402
    area_coverage,
    boxes_covering_gts,
    box_area,
    classify_gt_completeness,
    poly_to_bbox,
    split_count,
    whole_capture,
)
from ir_postprocessor import load_font  # noqa: E402
from run_diagram_crop_eval import (  # noqa: E402
    IMAGES_DIR,
    OFF_CACHE_DIR,
    OffRunner,
    PRED_FIG_LABELS,
    ROOT,
    DATASET_JSON,
    gt_boxes_by_kind,
    item_boxes,
)
from run_e2e_reading_order_eval import Logger, load_cache, save_json_atomic  # noqa: E402
from test_diagram_grouping import (  # noqa: E402
    box_inside_box,
    detect_visual_containers_opencv,
    group_whole_diagrams_image,
)

OUT_DIR = ROOT / "outputs/05_diagram_crop_eval"
VIZ_DIR = OUT_DIR / "viz"
THRS = (0.3, 0.5, 0.7)
PIECE_THR = 0.7
SPLIT_CONT_THR = 0.5
COVER_THR = 0.5
NEAR_GAP_PX = 80.0
N_BOOT = 2000
SUBSETS = ("broken", "intact", "all")


# ----------------------------------------------------------------------
# Grouping call (same convention as run_diagram_crop_eval.grouped_boxes)
# ----------------------------------------------------------------------
def group_call(items: List[Dict[str, Any]], img_path: Path, img_size: Tuple[int, int]) -> Dict[str, Any]:
    docling_items = [
        {"order": it["order"], "label": it["label"], "bbox": it["bbox"], "text": it.get("text", "")}
        for it in items if it.get("bbox") and len(it["bbox"]) == 4
    ]
    res = group_whole_diagrams_image(str(img_path), docling_items, img_size, img_size)
    grouped = res.get("grouped_items", [])
    group_boxes = [g["bbox"] for g in grouped if g.get("label") == "diagram_group"]
    standalone = [g["bbox"] for g in grouped if g.get("label") in PRED_FIG_LABELS]
    res["_group_boxes"] = group_boxes
    res["_standalone"] = standalone
    res["_r3_boxes"] = group_boxes + standalone
    return res


def group_diagnostics(res: Dict[str, Any], img_size: Tuple[int, int], containers: List[List[float]]) -> List[Dict[str, Any]]:
    """
    Per diagram group: which labels were merged, whether the final box was expanded beyond the
    union of its members (the group function pads by 8 px and clamps to the image, so any extra
    expansion came from a visual-container snap), and whether the union lies inside an OpenCV container.
    """
    W, H = img_size
    originals = res.get("original_items", [])
    out = []
    for g in res.get("diagram_groups", []):
        idxs = [i for i in g.get("merged_element_indices", []) if i < len(originals)]
        if not idxs:
            continue
        boxes = [originals[i]["bbox"] for i in idxs]
        ub = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
        expected = [max(0.0, ub[0] - 8), max(0.0, ub[1] - 8), min(float(W), ub[2] + 8), min(float(H), ub[3] + 8)]
        fb = g["bbox"]
        snapped = any(abs(fb[k] - expected[k]) > 1.0 for k in range(4))
        in_container = any(box_inside_box(ub, c, tolerance=0.15) for c in containers)
        labels = dict(Counter(originals[i]["label"] for i in idxs))
        out.append({"bbox": fb, "labels": labels, "n_merged": len(idxs), "snapped": snapped,
                    "in_container": in_container})
    return out


def cause_for(gt: List[float], groups: List[Dict[str, Any]], standalone: List[List[float]]) -> Dict[str, Any]:
    covering = [g for g in groups if area_coverage(gt, g["bbox"]) >= 0.5]
    if covering:
        g = max(covering, key=lambda x: area_coverage(gt, x["bbox"]))
        labels = g["labels"]
        if g["snapped"]:
            cause = "container_snap"
        elif labels.get("formula"):
            cause = "formula_absorbed"
        elif labels.get("table"):
            cause = "table_absorbed"
        elif any(l not in ("picture", "figure") for l in labels):
            cause = "other_label_absorbed"
        else:
            cause = "picture_proximity_merge"
        return {"cause": cause, "group_labels": labels, "group_snapped": g["snapped"],
                "group_in_container": g["in_container"], "group_n_merged": g["n_merged"]}
    if any(area_coverage(gt, b) >= 0.5 for b in standalone):
        return {"cause": "standalone_box_iou_low", "group_labels": {}, "group_snapped": False,
                "group_in_container": False, "group_n_merged": 0}
    return {"cause": "no_covering_box", "group_labels": {}, "group_snapped": False,
            "group_in_container": False, "group_n_merged": 0}


def variant_class(cls: str, n_pieces_variant: int) -> str:
    if cls == "intact":
        return "intact"
    if n_pieces_variant >= 2:
        return "broken"
    if n_pieces_variant == 0:
        return "missed"
    return "other"


# ----------------------------------------------------------------------
# Calling-convention check: pixel boxes + image_size  vs  pt boxes + docling_page_size_pt
# ----------------------------------------------------------------------
def calling_convention_check(items: List[Dict[str, Any]], img_path: Path, on_cache: Dict[str, Any]) -> Dict[str, Any]:
    w, h = on_cache["image_size"]
    dpt = on_cache["docling_page_size_pt"]
    sx, sy = w / dpt[0], h / dpt[1]
    pt_items = []
    for it in items:
        if it.get("bbox") and len(it["bbox"]) == 4:
            b = it["bbox"]
            pt_items.append({**it, "bbox": [b[0] / sx, b[1] / sy, b[2] / sx, b[3] / sy]})
    res_pt = group_whole_diagrams_image(str(img_path), [
        {"order": it["order"], "label": it["label"], "bbox": it["bbox"], "text": it.get("text", "")}
        for it in pt_items], (w, h), (dpt[0], dpt[1]))
    res_px = group_call(items, img_path, (w, h))
    a = sorted(res_px["_r3_boxes"])
    b = sorted(g["bbox"] for g in res_pt.get("grouped_items", []) if g.get("label") == "diagram_group" or g.get("label") in PRED_FIG_LABELS)
    same_count = len(a) == len(b)
    max_diff = max((max(abs(p - q) for p, q in zip(x, y)) for x, y in zip(a, b)), default=0.0) if same_count else None
    return {"n_px_call": len(a), "n_pt_call": len(b), "same_count": same_count,
            "max_abs_diff_px": None if max_diff is None else round(max_diff, 3)}


# ----------------------------------------------------------------------
# Aggregation helpers
# ----------------------------------------------------------------------
def rate(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den else None


def subset_gt(gts: List[Dict[str, Any]], subset: str) -> List[Dict[str, Any]]:
    if subset == "all":
        return gts
    return [g for g in gts if g["cls"] == subset]


def merge_stats(boxes: List[Dict[str, Any]], region: str, subset_ids: Optional[set]) -> Dict[str, Any]:
    """Box-level merge: boxes covering >=2 distinct GT figures (each >= 0.5 of its area)."""
    cands = [b for b in boxes if b["region"] == region and b["covered"]]
    if subset_ids is not None:
        cands = [b for b in cands if any(c in subset_ids for c in b["covered"])]
    merged = sum(1 for b in cands if len(b["covered"]) >= 2)
    return {"boxes_considered": len(cands), "merged_boxes": merged, "merge_rate": rate(merged, len(cands))}


def summarize_region(gts: List[Dict[str, Any]], region: str, boxes: List[Dict[str, Any]], subset: str) -> Dict[str, Any]:
    n = len(gts)
    out: Dict[str, Any] = {"n_gt": n}
    for t in THRS:
        key = f"{t:.1f}"
        out[f"whole_capture@{key}"] = rate(sum(1 for g in gts if g[f"{region}_whole@{key}"]), n)
    out["split_rate"] = rate(sum(1 for g in gts if g[f"{region}_split"] >= 2), n)
    ids = {g["gid"] for g in gts}
    out["merge_subset"] = merge_stats(boxes, region, ids if subset != "all" else None)
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    log = Logger(OUT_DIR / "run_v2.log")
    wall0 = time.time()
    log("=" * 80)
    log(f"Cluster capture eval v2 start | argv={' '.join(sys.argv[1:])}")

    data = json.load(open(DATASET_JSON, encoding="utf-8"))
    pool = []
    for i, it in enumerate(data):
        fig, _ = gt_boxes_by_kind(it)
        if fig and (IMAGES_DIR / it["page_info"]["image_path"]).exists():
            pool.append((i, it))
    if args.limit:
        pool = pool[: args.limit]
    log(f"Pages with >=1 GT figure: {len(pool)}")

    off = OffRunner(log)
    gts: List[Dict[str, Any]] = []
    boxes: List[Dict[str, Any]] = []      # predicted boxes with covering GT ids (global ids)
    near: List[Dict[str, Any]] = []        # near-distinct figure pairs
    gid = 0
    errors: List[Dict[str, Any]] = []
    cv_check: List[Dict[str, Any]] = []
    fresh = 0
    page_recs: List[Dict[str, Any]] = []

    for pos, (idx, item) in enumerate(pool, 1):
        p = item["page_info"]
        attr = p.get("page_attribute", {})
        img_rel = p["image_path"]
        img_path = IMAGES_DIR / img_rel
        stem = Path(img_rel).stem
        try:
            cache, src = off.get(idx, img_rel)
            if src == "docling":
                fresh += 1
            items = cache["items"]
            w, h = cache["image_size"]
            page_area = float(w * h)
            fig_gt, _ = gt_boxes_by_kind(item)
            r1 = item_boxes(items, PRED_FIG_LABELS)
            extra = item_boxes(items, ("formula", "table"))

            res = group_call(items, img_path, (w, h))
            containers = detect_visual_containers_opencv(str(img_path))
            groups = group_diagnostics(res, (w, h), containers)
            r3 = res["_r3_boxes"]
            standalone = res["_standalone"]

            if len(cv_check) < 5:
                cv_check.append({"stem": stem, **calling_convention_check(items, img_path, cache)})

            # Per-GT classification (global ids)
            page_gids = []
            for k, g in enumerate(fig_gt):
                cls, n_p = classify_gt_completeness(g, r1, piece_thr=PIECE_THR, iou_thr=0.5)
                n_pv = sum(1 for q in r1 + extra if piece_containment_ok(q, g))
                rec = {
                    "gid": gid, "stem": stem, "dataset_index": idx, "gt_k": k,
                    "bbox": [round(v, 1) for v in g],
                    "data_source": attr.get("data_source", "unknown"),
                    "layout": attr.get("layout", "unknown"),
                    "cls": cls, "n_pieces": n_p,
                    "cls_variant": variant_class(cls, n_pv), "n_pieces_variant": n_pv,
                    "page_area": page_area,
                }
                for t in THRS:
                    key = f"{t:.1f}"
                    rec[f"R1_whole@{key}"] = whole_capture(g, r1, t)
                    rec[f"R3_whole@{key}"] = whole_capture(g, r3, t)
                rec["R1_split"] = split_count(g, r1, SPLIT_CONT_THR)
                rec["R3_split"] = split_count(g, r3, SPLIT_CONT_THR)
                if rec["R3_whole@0.5"] is False and cls in ("broken", "intact"):
                    rec["R3_cause"] = cause_for(g, groups, standalone)
                gts.append(rec)
                page_gids.append(gid)
                gid += 1

            # Per-box coverage (for merge rates)
            cov_r1 = boxes_covering_gts(r1, fig_gt, COVER_THR)
            cov_r3 = boxes_covering_gts(r3, fig_gt, COVER_THR)
            for b, cov in zip(r1, cov_r1):
                boxes.append({"region": "R1", "covered": [page_gids[c] for c in cov]})
            for b, cov in zip(r3, cov_r3):
                boxes.append({"region": "R3", "covered": [page_gids[c] for c in cov]})

            # Near distinct figure pairs
            for a_i in range(len(fig_gt)):
                for b_i in range(a_i + 1, len(fig_gt)):
                    A, B = fig_gt[a_i], fig_gt[b_i]
                    ix = min(A[2], B[2]) - max(A[0], B[0])
                    iy = min(A[3], B[3]) - max(A[1], B[1])
                    if ix > 0 and iy > 0:
                        continue  # overlapping: not distinct separate figures
                    dx = max(0.0, max(A[0], B[0]) - min(A[2], B[2]))
                    dy = max(0.0, max(A[1], B[1]) - min(A[3], B[3]))
                    gap = math.hypot(dx, dy)
                    if gap <= NEAR_GAP_PX:
                        m1 = any(a_i in cov and b_i in cov for cov in cov_r1)
                        m3 = any(a_i in cov and b_i in cov for cov in cov_r3)
                        near.append({"stem": stem, "gap_px": round(gap, 1),
                                     "R1_merged": m1, "R3_merged": m3,
                                     "page_gids": [page_gids[a_i], page_gids[b_i]]})

            page_recs.append({"stem": stem, "data_source": attr.get("data_source", "unknown"),
                              "layout": attr.get("layout", "unknown"), "n_fig_gt": len(fig_gt),
                              "n_r1": len(r1), "n_r3": len(r3), "n_groups": len(res.get("diagram_groups", [])),
                              "n_standalone": len(standalone), "off_source": src,
                              "containers": len(containers)})
            if pos % 25 == 0 or pos == len(pool):
                log(f"[{pos}/{len(pool)}] {stem[:24]} off={src} gt={len(fig_gt)} R1 boxes={len(r1)} "
                    f"R3 boxes={len(r3)} groups={len(res.get('diagram_groups', []))} fresh_docling={fresh}")
        except Exception as e:
            errors.append({"stem": stem, "error": f"{type(e).__name__}: {str(e)[:300]}"})
            log(f"[{pos}/{len(pool)}] {stem[:24]} ERROR {type(e).__name__}: {str(e)[:200]}")

    log(f"Metrics computed: GT={len(gts)} boxes={len(boxes)} near_pairs={len(near)} errors={len(errors)}")

    # Aggregation -------------------------------------------------------------
    by_cls = Counter(g["cls"] for g in gts)
    by_cls_v = Counter(g["cls_variant"] for g in gts)
    dist = Counter(min(g["n_pieces"], 4) for g in gts)
    dist_v = Counter(min(g["n_pieces_variant"], 4) for g in gts)

    sub_tables: Dict[str, Any] = {}
    for s in SUBSETS:
        sg = subset_gt(gts, s)
        sub_tables[s] = {"n_gt": len(sg),
                         "R1": summarize_region(sg, "R1", boxes, s),
                         "R3": summarize_region(sg, "R3", boxes, s)}

    # gains / losses vs R1 (whole capture at 0.5)
    def paired_counts(sg):
        gain = sum(1 for g in sg if g["R3_whole@0.5"] and not g["R1_whole@0.5"])
        loss = sum(1 for g in sg if g["R1_whole@0.5"] and not g["R3_whole@0.5"])
        both = sum(1 for g in sg if g["R1_whole@0.5"] and g["R3_whole@0.5"])
        neither = sum(1 for g in sg if not g["R1_whole@0.5"] and not g["R3_whole@0.5"])
        return {"R3_gains": gain, "R3_losses": loss, "both_whole": both, "neither_whole": neither}

    paired = {s: paired_counts(subset_gt(gts, s)) for s in SUBSETS}

    causes = {
        "harm_intact": dict(Counter(g["R3_cause"]["cause"] for g in gts
                                    if g["cls"] == "intact" and g.get("R3_cause") and g["R1_whole@0.5"])),
        "fail_broken": dict(Counter(g["R3_cause"]["cause"] for g in gts
                                    if g["cls"] == "broken" and g.get("R3_cause"))),
    }
    causes_snapped = {
        "harm_intact_container_snap": sum(1 for g in gts if g["cls"] == "intact" and g.get("R3_cause")
                                          and g["R3_cause"]["group_snapped"]),
    }

    def breakdown(key: str) -> Dict[str, Any]:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for g in gts:
            groups[g[key]].append(g)
        out = {}
        for k, gs in sorted(groups.items()):
            out[k] = {"n_gt": len(gs), **{c: sum(1 for g in gs if g["cls"] == c) for c in ("broken", "intact", "missed", "other")},
                      "R1_whole@0.5": rate(sum(1 for g in gs if g["R1_whole@0.5"]), len(gs)),
                      "R3_whole@0.5": rate(sum(1 for g in gs if g["R3_whole@0.5"]), len(gs))}
        return out

    near_tab = {
        "pairs": len(near),
        "pages_with_pair": len({n["stem"] for n in near}),
        "R1_pairs_merged": sum(1 for n in near if n["R1_merged"]),
        "R3_pairs_merged": sum(1 for n in near if n["R3_merged"]),
    }
    near_pages = {n["stem"] for n in near}
    near_box_stats = {}
    for region in ("R1", "R3"):
        cands = [b for b in boxes if b["region"] == region and b["covered"]
                 and any(gts[c]["stem"] in near_pages for c in b["covered"])]
        merged = sum(1 for b in cands if len(b["covered"]) >= 2)
        near_box_stats[region] = {"boxes_considered": len(cands), "merged_boxes": merged,
                                  "merge_rate": rate(merged, len(cands))}
    near_tab["box_level_on_near_pages"] = near_box_stats

    glob_merge = {}
    for region in ("R1", "R3"):
        all_b = [b for b in boxes if b["region"] == region]
        cov_b = [b for b in all_b if b["covered"]]
        glob_merge[region] = {
            "all_boxes": len(all_b),
            "merged_all_boxes": sum(1 for b in all_b if len(b["covered"]) >= 2),
            "merge_rate_all_boxes": rate(sum(1 for b in all_b if len(b["covered"]) >= 2), len(all_b)),
            "boxes_covering_gt": len(cov_b),
            "merge_rate_covering_gt": rate(sum(1 for b in cov_b if len(b["covered"]) >= 2), len(cov_b)),
        }

    dist_pages = Counter(p["n_groups"] > 0 for p in page_recs)
    results: Dict[str, Any] = {
        "config": {"scope": "all pages with >=1 GT figure", "pages": len(pool), "ocr": "off",
                   "device": "cuda", "cache": OFF_CACHE_DIR.name, "piece_containment_thr": PIECE_THR,
                   "split_containment_thr": SPLIT_CONT_THR, "cover_thr": COVER_THR,
                   "near_gap_px": NEAR_GAP_PX, "thresholds": list(THRS),
                   "gt": "OmniDocBench category_type figure (ignore=False)"},
        "counts": {"pages": len(pool), "page_errors": len(errors), "gt_total": len(gts),
                   "off_fresh_docling_runs": fresh, "pages_with_any_group": dist_pages.get(True, 0)},
        "errors": errors,
        "calling_convention_check": cv_check,
        "gt_class_counts": dict(by_cls),
        "gt_class_counts_variant": dict(by_cls_v),
        "n_pieces_distribution": {str(k): v for k, v in sorted(dist.items())},
        "n_pieces_distribution_variant": {str(k): v for k, v in sorted(dist_v.items())},
        "subset_tables": sub_tables,
        "paired_vs_R1": paired,
        "r3_causes": causes,
        "r3_causes_container_snap_intact": causes_snapped,
        "near_distinct_figures": near_tab,
        "merge_global": glob_merge,
        "breakdown_data_source": breakdown("data_source"),
        "breakdown_layout": breakdown("layout"),
        "per_gt": gts,
        "per_page": page_recs,
    }
    save_json_atomic(OUT_DIR / "results_v2.json", results)
    log(f"Saved {OUT_DIR / 'results_v2.json'}")

    # Known diagram samples and visualisations --------------------------------
    samples = known_samples(log, off)
    results["known_diagram_samples"] = samples
    save_json_atomic(OUT_DIR / "results_v2.json", results)

    viz_list = make_viz(gts, log)
    results["viz"] = viz_list
    save_json_atomic(OUT_DIR / "results_v2.json", results)

    write_report(results, OUT_DIR / "report_v2.md")
    log(f"DONE: pages={len(pool)} errors={len(errors)} fresh_docling={fresh} wall={round(time.time() - wall0, 1)}s")


def piece_containment_ok(q: Sequence[float], g: Sequence[float]) -> bool:
    from eval_utils import piece_containment
    return piece_containment(q, g) >= PIECE_THR


# ----------------------------------------------------------------------
# Known diagram samples (outputs/02_diagram_grouping)
# ----------------------------------------------------------------------
KNOWN = [
    ("01_ppt_cards", "PPT_PresentationSpecification_page_010.png"),
    ("02_network_topology", "page-b1b8cc79-a7b3-4ac5-a7c2-49416b195ccb.png"),
    ("03_healthcare_infographic", "yanbaor2_ey-how-covid-19-has-triggered-a-sprint-toward-smarter-health-care.pdf_5.jpg"),
]


def known_samples(log, off: OffRunner) -> List[Dict[str, Any]]:
    out = []
    data = {it["page_info"]["image_path"]: (i, it) for i, it in enumerate(json.load(open(DATASET_JSON, encoding="utf-8")))}
    for name, fn in KNOWN:
        idx, item = data[fn]
        cache, src = off.get(idx, fn)
        items = cache["items"]
        w, h = cache["image_size"]
        res = group_call(items, IMAGES_DIR / fn, (w, h))
        fig_gt, _ = gt_boxes_by_kind(item)
        r1 = item_boxes(items, PRED_FIG_LABELS)
        r3 = res["_r3_boxes"]
        out.append({
            "sample": name, "file": fn, "off_source": src, "n_gt_figure": len(fig_gt),
            "raw_picture_boxes_ocr_off": len(r1),
            "diagram_groups": len(res.get("diagram_groups", [])),
            "standalone_pictures_after_grouping": len(res["_standalone"]),
            "r3_boxes": len(r3),
            "R1_whole@0.5": None if not fig_gt else [whole_capture(g, r1, 0.5) for g in fig_gt],
            "R3_whole@0.5": None if not fig_gt else [whole_capture(g, r3, 0.5) for g in fig_gt],
            "R1_split": None if not fig_gt else [split_count(g, r1, SPLIT_CONT_THR) for g in fig_gt],
            "R3_split": None if not fig_gt else [split_count(g, r3, SPLIT_CONT_THR) for g in fig_gt],
        })
    # vector flowchart PDF (no GT): group_whole_diagrams_vector only
    try:
        from test_diagram_grouping import group_whole_diagrams_vector
        pdf = OUT_DIR.parent / "02_diagram_grouping" / "04_vector_flowchart.pdf"
        vres = group_whole_diagrams_vector(str(pdf))
        out.append({"sample": "04_vector_flowchart_pdf", "file": pdf.name, "off_source": "vector",
                    "n_gt_figure": None, "raw_drawings": vres.get("total_drawings"),
                    "diagram_groups": len(vres.get("diagram_groups", [])),
                    "note": "vector branch group_whole_diagrams_vector, no GT"})
    except Exception as e:
        out.append({"sample": "04_vector_flowchart_pdf", "error": f"{type(e).__name__}: {str(e)[:200]}"})
    return out


# ----------------------------------------------------------------------
# Visualisations: GT blue, prediction red
# ----------------------------------------------------------------------
def _draw_case(stem: str, title: str, gts_page: List[Dict[str, Any]], pred_boxes: List[List[float]],
               focus_gid: int, out_path: Path, extra_note: str) -> None:
    img = Image.open(IMAGES_DIR / next(p for p in _IMG_BY_STEM[stem])).convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size
    font = load_font(max(18, W // 60))
    for g in gts_page:
        b = g["bbox"]
        w_ = 6 if g["gid"] == focus_gid else 3
        draw.rectangle(b, outline=(30, 90, 230), width=w_)
    for pb in pred_boxes:
        draw.rectangle(pb, outline=(220, 30, 30), width=4)
    banner_h = int(font.size * 2.6) if hasattr(font, "size") else 60
    draw.rectangle([0, 0, W, banner_h], fill=(20, 20, 20))
    draw.text((12, 6), title, fill=(255, 255, 255), font=font)
    draw.text((12, banner_h // 2 + 2), extra_note, fill=(255, 215, 0), font=font)
    if W > 1600:
        scale = 1600 / W
        img = img.resize((1600, int(H * scale)))
    img.save(out_path, "PNG")


_IMG_BY_STEM: Dict[str, List[str]] = {}


def make_viz(gts: List[Dict[str, Any]], log) -> List[Dict[str, Any]]:
    rng = random.Random(42)
    # map stem -> image file name
    for it in json.load(open(DATASET_JSON, encoding="utf-8")):
        fn = it["page_info"]["image_path"]
        _IMG_BY_STEM.setdefault(Path(fn).stem, []).append(fn)
    groups = {
        "broken_fixed_by_R3": [g for g in gts if g["cls"] == "broken" and g["R3_whole@0.5"]],
        "broken_still_wrong_R3": [g for g in gts if g["cls"] == "broken" and not g["R3_whole@0.5"]],
        "intact_harmed_by_R3": [g for g in gts if g["cls"] == "intact" and g["R1_whole@0.5"] and not g["R3_whole@0.5"]],
    }
    out = []
    for key, pool_g in groups.items():
        pick = rng.sample(pool_g, min(6, len(pool_g)))
        for n, g in enumerate(sorted(pick, key=lambda x: x["gid"]), 1):
            page = [x for x in gts if x["stem"] == g["stem"]]
            # predicted R3 boxes for this page: recompute to draw
            cache = load_cache(OFF_CACHE_DIR / f"{g['stem']}.json")
            items = cache["items"]
            w, h = cache["image_size"]
            res = group_call(items, IMAGES_DIR / _IMG_BY_STEM[g["stem"]][0], (w, h))
            pred = res["_r3_boxes"]
            cause = g.get("R3_cause", {}).get("cause", "-") if isinstance(g.get("R3_cause"), dict) else "-"
            name = f"{key}_{n:02d}.png"
            title = f"{key} | {g['stem'][:34]} | GT blue, R3 red | cls={g['cls']} | cause={cause}"
            note = f"GT figs={len(page)} R3 boxes={len(pred)} | focus GT bbox={[round(v) for v in g['bbox']]}"
            _draw_case(g["stem"], title, page, pred, g["gid"], VIZ_DIR / name, note)
            out.append({"case": key, "file": f"viz/{name}", "stem": g["stem"], "gid": g["gid"],
                        "cls": g["cls"], "cause": cause})
    log(f"Saved {len(out)} visualisations to {VIZ_DIR}")
    return out


# ----------------------------------------------------------------------
# Report (Vietnamese)
# ----------------------------------------------------------------------
def f3(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:.3f}"


def pct(v: Optional[float]) -> str:
    return "—" if v is None else f"{100 * v:.1f}%"


def write_report(res: Dict[str, Any], path: Path) -> None:
    cfg, cnt = res["config"], res["counts"]
    st = res["subset_tables"]
    L: List[str] = []
    L.append("# Đánh giá lại trích xuất cụm hình (whole-capture / split / merge), OCR tắt\n")
    L.append(f"> **Phạm vi:** {cnt['pages']} trang OmniDocBench có >= 1 GT `figure` | **GT figure:** {cnt['gt_total']} | "
             f"**Lỗi trang:** {cnt['page_errors']} | **Docling OCR-tắt chạy mới:** {cnt['off_fresh_docling_runs']} trang (còn lại từ cache)\n")
    L.append("---\n")

    L.append("## 1. Số GT theo loại (vỡ / nguyên vẹn / bỏ sót / khác)\n")
    c = res["gt_class_counts"]
    cv = res["gt_class_counts_variant"]
    L.append("| Loại | Định nghĩa | Số GT (chính) | % | Số GT (biến thể formula/table) |")
    L.append("|---|---|---|---|---|")
    total = cnt["gt_total"] or 1
    defs = {
        "intact": "có box thô IoU >= 0.5",
        "broken": "không có box IoU >= 0.5, >= 2 box thô có containment >= 0.7 trong G",
        "missed": "không có box IoU >= 0.5 và không có box thô nào chứa >= 0.7 trong G",
        "other": "không có box IoU >= 0.5, đúng 1 box thô chứa >= 0.7 trong G",
    }
    for k in ("intact", "broken", "missed", "other"):
        L.append(f"| **{k}** | {defs[k]} | {c.get(k, 0)} | {pct(c.get(k, 0) / total)} | {cv.get(k, 0)} |")
    L.append("")
    L.append("Phân bố số mảnh thô có containment >= 0.7 trong G (|P(G)|): " +
             ", ".join(f"{k}: {v}" for k, v in res["n_pieces_distribution"].items()) + " (4 = 4 trở lên)\n")

    L.append("### Phân rã theo data_source\n")
    L.append("| data_source | GT | intact | broken | missed | other | R1 whole@0.5 | R3 whole@0.5 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k, row in res["breakdown_data_source"].items():
        L.append(f"| `{k}` | {row['n_gt']} | {row['intact']} | {row['broken']} | {row['missed']} | {row['other']} | "
                 f"{pct(row['R1_whole@0.5'])} | {pct(row['R3_whole@0.5'])} |")
    L.append("")
    L.append("### Phân rã theo layout\n")
    L.append("| layout | GT | intact | broken | missed | other | R1 whole@0.5 | R3 whole@0.5 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k, row in res["breakdown_layout"].items():
        L.append(f"| `{k}` | {row['n_gt']} | {row['intact']} | {row['broken']} | {row['missed']} | {row['other']} | "
                 f"{pct(row['R1_whole@0.5'])} | {pct(row['R3_whole@0.5'])} |")
    L.append("")

    L.append("## 2. Whole-capture / split / merge: R1 (thô) và R3 (gom), theo tập GT\n")
    L.append("| Tập GT | n | Vùng | Whole@0.3 | Whole@0.5 | Whole@0.7 | Split | Merge (tập) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s, label in (("broken", "(a) vỡ"), ("intact", "(b) nguyên vẹn"), ("all", "(c) toàn bộ")):
        for reg in ("R1", "R3"):
            t = st[s][reg]
            m = t["merge_subset"]
            L.append(f"| {label} | {st[s]['n_gt']} | {reg} | {pct(t['whole_capture@0.3'])} | {pct(t['whole_capture@0.5'])} | "
                     f"{pct(t['whole_capture@0.7'])} | {pct(t['split_rate'])} | {pct(m['merge_rate'])} ({m['merged_boxes']}/{m['boxes_considered']} box) |")
    L.append("")
    L.append("Merge (tập) = trong số box dự đoán phủ >= 0.5 của ít nhất một GT thuộc tập, tỉ lệ box phủ >= 0.5 của >= 2 GT figure khác nhau.\n")

    L.append("## 3. Gom có giúp trên tập vỡ, có hại trên tập nguyên vẹn?\n")
    L.append("| Tập | Gom sửa được (R3 whole, R1 không) | Gom làm hỏng (R1 whole, R3 không) | Cả hai đúng | Cả hai sai |")
    L.append("|---|---|---|---|---|")
    for s, label in (("broken", "(a) vỡ"), ("intact", "(b) nguyên vẹn"), ("all", "(c) toàn bộ")):
        p = res["paired_vs_R1"][s]
        L.append(f"| {label} | {p['R3_gains']} | {p['R3_losses']} | {p['both_whole']} | {p['neither_whole']} |")
    L.append("")

    L.append("## 4. Nguyên nhân lỗi của R3 (đọc từ `group_whole_diagrams_image`)\n")
    L.append("Nguyên nhân = nhóm gom có chứa bao nhiêu từng loại nhãn, và box cuối có bị nở ra so với hợp các thành viên (đệm 8 px) hay không (container snap).\n")
    c3 = res["r3_causes"]
    L.append(f"- Nguyên nhân khi GT nguyên vẹn bị R3 làm hỏng: {c3['harm_intact'] or '—'}")
    L.append(f"- Nguyên nhân khi GT vỡ vẫn sai ở R3: {c3['fail_broken'] or '—'}")
    L.append(f"- GT nguyên vẹn bị hỏng mà nhóm gom bị container snap: {res['r3_causes_container_snap_intact']['harm_intact_container_snap']}\n")
    L.append("Các nguyên nhân trong code: `container_snap` (box cuối nở tới một container OpenCV), `formula_absorbed` / `table_absorbed` (proximity dilation hút công thức/bảng vào hình), `picture_proximity_merge` (ghép hai hình ảnh gần nhau), `standalone_box_iou_low` (box picture độc lập nhưng không khớp IoU), `no_covering_box`.\n")

    L.append("## 5. Hình riêng biệt gần nhau (near distinct figures)\n")
    nd = res["near_distinct_figures"]
    L.append(f"- Cặp hình không chồng lấn, khoảng cách cạnh <= {cfg['near_gap_px']:.0f} px: **{nd['pairs']}** cặp trên **{nd['pages_with_pair']}** trang.")
    L.append(f"- Cặp bị gom vào một box: R1 **{nd['R1_pairs_merged']}** cặp, R3 **{nd['R3_pairs_merged']}** cặp.")
    bl = nd["box_level_on_near_pages"]
    L.append(f"- Merge rate box-level trên các trang này: R1 {pct(bl['R1']['merge_rate'])} ({bl['R1']['merged_boxes']}/{bl['R1']['boxes_considered']}), "
             f"R3 {pct(bl['R3']['merge_rate'])} ({bl['R3']['merged_boxes']}/{bl['R3']['boxes_considered']}).\n")

    L.append("## 6. Merge toàn cục\n")
    mg = res["merge_global"]
    L.append("| Vùng | Tổng box | Box merge | Merge rate (mọi box) | Box phủ >= 1 GT | Merge rate (box phủ GT) |")
    L.append("|---|---|---|---|---|---|")
    for reg in ("R1", "R3"):
        g = mg[reg]
        L.append(f"| {reg} | {g['all_boxes']} | {g['merged_all_boxes']} | {pct(g['merge_rate_all_boxes'])} | "
                 f"{g['boxes_covering_gt']} | {pct(g['merge_rate_covering_gt'])} |")
    L.append("")

    L.append("## 7. Kiểm tra cách gọi `group_whole_diagrams_image`\n")
    cc = res["calling_convention_check"]
    ok = all(x["same_count"] and (x["max_abs_diff_px"] or 0) <= 0.2 for x in cc) if cc else False
    L.append(f"Gọi bằng box pixel + `docling_page_size = image_size` so với box pt + `docling_page_size_pt`: "
             f"{'khớp' if ok else 'KHÔNG khớp'} trên {len(cc)} trang "
             f"({', '.join(f'{x['stem'][:16]}: {x['n_px_call']} box, lệch tối đa {x['max_abs_diff_px']} px' for x in cc)}).\n")

    L.append("## 8. Bốn mẫu sơ đồ đã biết\n")
    L.append("| Mẫu | Nguồn | GT figure | Box thô (picture, OCR tắt) | Diagram group | Standalone sau gom | Whole R1 | Whole R3 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in res["known_diagram_samples"]:
        if "error" in s:
            L.append(f"| {s['sample']} | lỗi | — | — | — | — | — | {s['error']} |")
            continue
        if s.get("n_gt_figure") is None:
            L.append(f"| {s['sample']} | {s['file']} (vector) | n/a | {s.get('raw_drawings')} drawings | {s['diagram_groups']} | — | — | — |")
            continue
        w1 = "—" if s["R1_whole@0.5"] is None else f"{sum(s['R1_whole@0.5'])}/{len(s['R1_whole@0.5'])}"
        w3 = "—" if s["R3_whole@0.5"] is None else f"{sum(s['R3_whole@0.5'])}/{len(s['R3_whole@0.5'])}"
        L.append(f"| {s['sample']} | {s['file'][:40]} | {s['n_gt_figure']} | {s['raw_picture_boxes_ocr_off']} | {s['diagram_groups']} | "
                 f"{s['standalone_pictures_after_grouping']} | {w1} | {w3} |")
    L.append("")

    L.append("## 9. Ảnh minh hoạ\n")
    L.append("Thư mục `viz/`: 6 ca `broken_fixed_by_R3`, 6 ca `broken_still_wrong_R3`, 6 ca `intact_harmed_by_R3` (chọn ngẫu nhiên, seed 42). GT xanh, dự đoán R3 đỏ, GT tiêu điểm nét dày. Nguyên nhân ghi trên tiêu đề.\n")
    for v in res.get("viz", []):
        L.append(f"- `{v['file']}`: {v['case']} | `{v['stem'][:40]}` | cls={v['cls']} | cause={v['cause']}")
    L.append("")

    L.append("## 10. Giới hạn\n")
    L.append("- GT `figure` của OmniDocBench gồm cả ảnh chụp, biểu đồ và sơ đồ, không chỉ sơ đồ khối. Đây là proxy, không phải đo trên sơ đồ khối tổng quát.")
    L.append(f"- Phạm vi: {cnt['pages']} trang đã có GT figure trong dataset (toàn bộ, không lấy mẫu). Không có tập giữ lại (held-out). Ngưỡng của `group_whole_diagrams_image` và các guard được đặt sẵn, không được tối ưu lại.")
    L.append("- Định nghĩa vỡ dựa vào box thô của Docling làm chuẩn. Nếu Docling đã chia nhỏ hình ngay từ đầu, GT vẫn được xếp vào nhóm vỡ, nhưng nó chỉ đo được sự vỡ thể hiện trong dataset này.")
    L.append("- Mẫu sơ đồ đã biết chỉ có 3 ảnh raster (có GT figure ở 2 trang) và 1 PDF vector không có GT. Không đủ để kết luận về sơ đồ khối nói chung.")
    L.append("- Merge rate tập con là tỉ lệ box dự đoán, không phải tỉ lệ GT; một box có thể đóng góp vào nhiều tập.")
    L.append("- Lỗi trang trong `results_v2.json` > `errors`; không tính vào thống kê.\n")
    path.write_text("\n".join(L).strip() + "\n", encoding="utf-8")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Cluster capture evaluation v2 (OCR off).")
    ap.add_argument("--limit", type=int, default=0, help="only the first N pages (smoke test)")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    run(parse_args(argv))


if __name__ == "__main__":
    main()
