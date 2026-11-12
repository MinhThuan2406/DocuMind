"""
Diagram / figure crop evaluation: does disabling OCR change crop quality?

Region sets per page (all predictions are image-pixel boxes):
  R1  Docling 'picture' boxes, raw (no grouping), OCR on      (cache_ocr-torch_dev-cuda)
  R2  group_whole_diagrams_image() output, OCR on             (same cache)
  R3  group_whole_diagrams_image() output, OCR off            (cache_ocr-off_dev-cuda)
  R4  Docling 'picture' boxes, raw, OCR off                   (cache_ocr-off_dev-cuda)

Ground truth: OmniDocBench 'figure' boxes (ignore=False). OmniDocBench figures include
photos and charts, not only flow diagrams, so this is a proxy for the diagram-crop goal.

Metrics (per page and aggregated):
  precision / recall / F1 at IoU 0.5 (primary; 0.3 and 0.7 also reported), Hungarian one-to-one.
  over-merge box: predicted box covering (>=50% of GT area) >=2 GT figures, or >=2 'other'
  GT blocks (text-like categories listed in TEXT_LIKE_GT).
  fragmented GT figure: >=2 predicted boxes with IoU in [0.1, 0.5) against that GT figure.
  largest predicted box area / page area.

Usage:
  .venv\\Scripts\\python scripts\\run_diagram_crop_eval.py                 # 100 pages, seed 42
  .venv\\Scripts\\python scripts\\run_diagram_crop_eval.py --limit 5      # smoke test
"""

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from eval_utils import (  # noqa: E402
    box_area,
    detection_prf,
    fragmented_gt_count,
    over_merge_count,
    poly_to_bbox,
)
from run_e2e_reading_order_eval import (  # noqa: E402
    Logger,
    build_converter,
    cache_dir_for,
    load_cache,
    run_docling_page,
    save_json_atomic,
    stratified_sample,
)
from test_diagram_grouping import group_whole_diagrams_image  # noqa: E402

ROOT = SCRIPT_DIR.parent
DATASET_JSON = ROOT / "datasets/OmniDocBench/OmniDocBench.json"
IMAGES_DIR = ROOT / "datasets/OmniDocBench/images"
E2E_DIR = ROOT / "outputs/04_e2e_eval"
OUT_DIR = ROOT / "outputs/05_diagram_crop_eval"

ON_CACHE_DIR = cache_dir_for(E2E_DIR, "torch", "cuda")   # OCR on (torch GPU), 300 pages already cached
OFF_CACHE_DIR = cache_dir_for(E2E_DIR, "off", "cuda")    # OCR off, same device

GT_FIG_CAT = "figure"
PRED_FIG_LABELS = ("picture", "figure")
TEXT_LIKE_GT = (
    "text_block", "title", "header", "footer", "page_number", "page_footnote",
    "figure_caption", "figure_footnote", "table_caption", "table_footnote",
    "equation_caption", "equation_isolated", "equation_explanation", "reference",
    "code_txt", "code_txt_caption",
)
REGIONS = {
    "R1": "docling_picture_raw_ocr_on",
    "R2": "grouped_ocr_on",
    "R3": "grouped_ocr_off",
    "R4": "docling_picture_raw_ocr_off",
}
IOU_THRS = (0.3, 0.5, 0.7)
PRIMARY_THR = 0.5
N_BOOT = 2000
MIN_GROUP_PAGES = 4


# ----------------------------------------------------------------------
# Ground truth and predictions
# ----------------------------------------------------------------------
def gt_boxes_by_kind(item: Dict[str, Any]) -> Tuple[List[List[float]], List[List[float]]]:
    fig: List[List[float]] = []
    other: List[List[float]] = []
    for d in item.get("layout_dets", []):
        if d.get("ignore", False):
            continue
        cat = d.get("category_type", "")
        box = d.get("poly") or d.get("polygon") or d.get("bbox") or []
        b = poly_to_bbox(box)
        if not b:
            continue
        if cat == GT_FIG_CAT:
            fig.append(b)
        elif cat in TEXT_LIKE_GT:
            other.append(b)
    return fig, other


def has_gt_figure(item: Dict[str, Any]) -> bool:
    fig, _ = gt_boxes_by_kind(item)
    return len(fig) > 0


def item_boxes(items: List[Dict[str, Any]], labels: Sequence[str]) -> List[List[float]]:
    return [it["bbox"] for it in items if it.get("bbox") and len(it["bbox"]) == 4 and it.get("label") in labels]


def grouped_boxes(items: List[Dict[str, Any]], image_path: Path, img_size: Tuple[int, int]) -> Tuple[List[List[float]], int]:
    """
    Calls the production group_whole_diagrams_image() on the cached Docling items.
    Cached bboxes are already in image pixels, so docling_page_size = image_size gives scale 1.0
    and the function sees exactly the coordinates the pipeline would see.
    Returns predicted boxes (diagram groups + standalone pictures) and the number of groups.
    """
    docling_items = [
        {"order": it["order"], "label": it["label"], "bbox": it["bbox"], "text": it.get("text", "")}
        for it in items if it.get("bbox") and len(it["bbox"]) == 4
    ]
    res = group_whole_diagrams_image(str(image_path), docling_items, img_size, img_size)
    grouped = res.get("grouped_items", [])
    boxes = [g["bbox"] for g in grouped if g.get("label") == "diagram_group" or g.get("label") in PRED_FIG_LABELS]
    return boxes, len(res.get("diagram_groups", []))


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------
def score_region(preds: List[List[float]], fig_gt: List[List[float]], other_gt: List[List[float]],
                 page_area: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n_pred": len(preds), "n_gt": len(fig_gt)}
    for thr in IOU_THRS:
        r = detection_prf(preds, fig_gt, thr)
        key = f"{thr:.1f}"
        out[f"tp@{key}"] = r["tp"]
        out[f"precision@{key}"] = round(r["precision"], 4)
        out[f"recall@{key}"] = round(r["recall"], 4)
        out[f"f1@{key}"] = round(r["f1"], 4)
        if thr == PRIMARY_THR:
            out["mean_iou_matched"] = None if r["mean_iou_matched"] is None else round(r["mean_iou_matched"], 4)
            out["iou_sum_matched"] = 0.0 if r["mean_iou_matched"] is None else round(r["mean_iou_matched"] * r["tp"], 4)
    out.update(over_merge_count(preds, fig_gt, other_gt))
    out["fragmented_gt"] = fragmented_gt_count(preds, fig_gt)
    areas = [box_area(p) for p in preds]
    out["max_pred_area_ratio"] = round(max(areas) / page_area, 4) if areas and page_area > 0 else 0.0
    return out


def label_counts(items: List[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(it.get("label", "?") for it in items if it.get("bbox")))


# ----------------------------------------------------------------------
# OFF-cache generation (Docling, do_ocr=False)
# ----------------------------------------------------------------------
class OffRunner:
    def __init__(self, log: Logger):
        self.log = log
        self.conv = None
        self.resolved = None
        OFF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def get(self, idx: int, img_rel: str) -> Tuple[Dict[str, Any], str]:
        stem = Path(img_rel).stem
        path = OFF_CACHE_DIR / f"{stem}.json"
        c = load_cache(path)
        if c is not None and c.get("status") == "ok":
            return c, "cache"
        if self.conv is None:
            from docling.utils.accelerator_utils import decide_device
            self.log(f"Initialising DocumentConverter: ocr_backend=off device=cuda cache_dir={OFF_CACHE_DIR.name}")
            self.conv = build_converter("off", "cuda")
            self.resolved = decide_device("cuda")
        cache = run_docling_page(self.conv, IMAGES_DIR / img_rel, track_gpu=self.resolved.startswith("cuda"))
        cache.update({"status": "ok", "image_path": img_rel, "dataset_index": idx,
                      "ocr_backend": "off", "device_requested": "cuda", "device_resolved": self.resolved})
        save_json_atomic(path, cache)
        return cache, "docling"


# ----------------------------------------------------------------------
# Aggregation helpers
# ----------------------------------------------------------------------
def micro_macro(recs: List[Dict[str, Any]], region: str) -> Dict[str, Any]:
    rows = [r["regions"][region] for r in recs]
    tp = sum(x["tp@0.5"] for x in rows)
    npred = sum(x["n_pred"] for x in rows)
    ngt = sum(x["n_gt"] for x in rows)
    micro_p = tp / npred if npred else 0.0
    micro_r = tp / ngt if ngt else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    iou_sum = sum(x["iou_sum_matched"] for x in rows)
    f1s = [x["f1@0.5"] for x in rows]
    return {
        "n_pages": len(rows),
        "tp": tp, "n_pred": npred, "n_gt": ngt,
        "micro_precision": round(micro_p, 4), "micro_recall": round(micro_r, 4), "micro_f1": round(micro_f1, 4),
        "macro_f1_mean": round(statistics.mean(f1s), 4) if f1s else None,
        "macro_f1_median": round(statistics.median(f1s), 4) if f1s else None,
        "macro_precision_mean": round(statistics.mean(x["precision@0.5"] for x in rows), 4) if rows else None,
        "macro_recall_mean": round(statistics.mean(x["recall@0.5"] for x in rows), 4) if rows else None,
        "micro_f1@0.3": round(_micro_f1_thr(rows, "0.3"), 4),
        "micro_f1@0.7": round(_micro_f1_thr(rows, "0.7"), 4),
        "mean_iou_matched": round(iou_sum / tp, 4) if tp else None,
        "over_merge_boxes": sum(x["over_merge_boxes"] for x in rows),
        "pages_with_over_merge": sum(1 for x in rows if x["over_merge_boxes"] > 0),
        "text_blocks_absorbed": sum(x["text_blocks_absorbed"] for x in rows),
        "fragmented_gt": sum(x["fragmented_gt"] for x in rows),
        "pages_with_fragmentation": sum(1 for x in rows if x["fragmented_gt"] > 0),
        "mean_max_pred_area_ratio": round(statistics.mean(x["max_pred_area_ratio"] for x in rows), 4) if rows else None,
        "median_max_pred_area_ratio": round(statistics.median(x["max_pred_area_ratio"] for x in rows), 4) if rows else None,
        "pages_max_area_over_45pct": sum(1 for x in rows if x["max_pred_area_ratio"] > 0.45),
        "mean_n_pred": round(statistics.mean(x["n_pred"] for x in rows), 3) if rows else None,
    }


def _micro_f1_thr(rows: List[Dict[str, Any]], key: str) -> float:
    tp = sum(x[f"tp@{key}"] for x in rows)
    npred = sum(x["n_pred"] for x in rows)
    ngt = sum(x["n_gt"] for x in rows)
    p = tp / npred if npred else 0.0
    r = tp / ngt if ngt else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def paired(recs: List[Dict[str, Any]], a: str, b: str, seed: int = 42) -> Dict[str, Any]:
    """Per-page F1@0.5 comparison: a minus b (wins/ties/losses for a)."""
    d = np.array([r["regions"][a]["f1@0.5"] - r["regions"][b]["f1@0.5"] for r in recs], dtype=float)
    out: Dict[str, Any] = {"n_pages": int(d.size), "a_wins": 0, "ties": 0, "a_losses": 0,
                           "mean_delta_f1": None, "median_delta_f1": None, "bootstrap_ci95_mean_delta": None}
    if d.size == 0:
        return out
    eps = 1e-9
    out.update({"a_wins": int(np.sum(d > eps)), "ties": int(np.sum(np.abs(d) <= eps)),
                "a_losses": int(np.sum(d < -eps)),
                "mean_delta_f1": round(float(d.mean()), 4), "median_delta_f1": round(float(np.median(d)), 4)})
    if d.size >= 2:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, d.size, size=(N_BOOT, d.size))
        boot = d[idx].mean(axis=1)
        out["bootstrap_ci95_mean_delta"] = [round(float(np.percentile(boot, 2.5)), 4),
                                            round(float(np.percentile(boot, 97.5)), 4)]
    return out


def group_by(recs: List[Dict[str, Any]], key: str, min_pages: int = 1) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in recs:
        groups[r[key]].append(r)
    out = {}
    for g, rs in sorted(groups.items()):
        if len(rs) < min_pages:
            continue
        out[g] = {"n_pages": len(rs), **{reg: {"micro_f1": micro_macro(rs, reg)["micro_f1"],
                                                "micro_precision": micro_macro(rs, reg)["micro_precision"],
                                                "micro_recall": micro_macro(rs, reg)["micro_recall"]}
                                         for reg in REGIONS}}
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = Logger(OUT_DIR / "run.log")
    wall0 = time.time()
    log("=" * 80)
    log(f"Diagram crop eval start | argv={' '.join(sys.argv[1:])}")

    data = json.load(open(DATASET_JSON, encoding="utf-8"))
    pool = []
    for i, it in enumerate(data):
        fn = it["page_info"]["image_path"]
        if not (IMAGES_DIR / fn).exists() or not has_gt_figure(it):
            continue
        if load_cache(ON_CACHE_DIR / f"{Path(fn).stem}.json") is None:
            continue
        pool.append((i, it))
    log(f"Pages with >=1 GT figure and an OCR-on cache: {len(pool)}")
    sel = stratified_sample(pool, args.n_pages, args.seed)
    if args.limit:
        sel = sel[: args.limit]
    sel = sorted(sel, key=lambda x: x[0])
    (OUT_DIR / "pages.txt").write_text("\n".join(it["page_info"]["image_path"] for _, it in sel) + "\n", encoding="utf-8")
    log(f"Selected {len(sel)} pages (stratified layout x data_source, seed {args.seed})")

    off = OffRunner(log)
    recs: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    fresh_off = 0
    group_time = 0.0

    for pos, (idx, item) in enumerate(sel, 1):
        p = item["page_info"]
        attr = p.get("page_attribute", {})
        img_rel = p["image_path"]
        stem = Path(img_rel).stem
        img_path = IMAGES_DIR / img_rel
        try:
            on = load_cache(ON_CACHE_DIR / f"{stem}.json")
            off_c, src = off.get(idx, img_rel)
            if src == "docling":
                fresh_off += 1
            w, h = (int(on["image_size"][0]), int(on["image_size"][1]))
            page_area = float(w * h)
            fig_gt, other_gt = gt_boxes_by_kind(item)

            t0 = time.time()
            r2_boxes, r2_groups = grouped_boxes(on["items"], img_path, (w, h))
            r3_boxes, r3_groups = grouped_boxes(off_c["items"], img_path, (w, h))
            group_time += time.time() - t0
            preds = {
                "R1": item_boxes(on["items"], PRED_FIG_LABELS),
                "R2": r2_boxes,
                "R3": r3_boxes,
                "R4": item_boxes(off_c["items"], PRED_FIG_LABELS),
            }
            regions = {k: score_region(v, fig_gt, other_gt, page_area) for k, v in preds.items()}
            rec = {
                "dataset_index": idx, "image_path": img_rel, "stem": stem,
                "layout": attr.get("layout", "unknown"), "data_source": attr.get("data_source", "unknown"),
                "special_issue": [s for s in attr.get("special_issue", []) if s != "None"] or ["standard_page"],
                "image_size": [w, h], "n_fig_gt": len(fig_gt), "n_other_gt": len(other_gt),
                "groups_on": r2_groups, "groups_off": r3_groups,
                "labels_on": label_counts(on["items"]), "labels_off": label_counts(off_c["items"]),
                "docling_time_on": on.get("docling_time_sec"), "docling_time_off": off_c.get("docling_time_sec"),
                "stage_on": on.get("stage_times_sec", {}), "stage_off": off_c.get("stage_times_sec", {}),
                "off_source": src,
                "regions": regions,
            }
            recs.append(rec)
            f1 = {k: regions[k]["f1@0.5"] for k in REGIONS}
            log(f"[{pos}/{len(sel)}] {stem[:24]:24s} off={src:7s} figGT={len(fig_gt):2d} "
                f"F1 R1={f1['R1']:.2f} R2={f1['R2']:.2f} R3={f1['R3']:.2f} R4={f1['R4']:.2f} "
                f"text ON={rec['labels_on'].get('text', 0)} OFF={rec['labels_off'].get('text', 0)} "
                f"groups ON={r2_groups} OFF={r3_groups}")
        except Exception as e:  # record and continue
            errors.append({"image_path": img_rel, "error": f"{type(e).__name__}: {str(e)[:300]}"})
            log(f"[{pos}/{len(sel)}] {stem[:24]} ERROR {type(e).__name__}: {str(e)[:200]}")

    off_times = [r["docling_time_off"] for r in recs if r["docling_time_off"] is not None and r["off_source"] == "docling"]
    on_times = [r["docling_time_on"] for r in recs if r["docling_time_on"] is not None]
    stage_keys = sorted({k for r in recs for k in r["stage_on"]} | {k for r in recs for k in r["stage_off"]})

    def stage_med(key_dict: str, skip_first: bool = True) -> Dict[str, float]:
        rs = recs[1:] if skip_first and len(recs) > 1 else recs
        return {k: round(statistics.median([r[key_dict].get(k, 0.0) for r in rs]), 3) for k in stage_keys}

    label_keys = sorted({k for r in recs for k in r["labels_on"]} | {k for r in recs for k in r["labels_off"]})
    label_totals = {k: {"on_total": sum(r["labels_on"].get(k, 0) for r in recs),
                        "off_total": sum(r["labels_off"].get(k, 0) for r in recs),
                        "on_pages_with": sum(1 for r in recs if r["labels_on"].get(k, 0) > 0),
                        "off_pages_with": sum(1 for r in recs if r["labels_off"].get(k, 0) > 0)} for k in label_keys}

    worst = sorted(recs, key=lambda r: r["regions"]["R3"]["f1@0.5"] - r["regions"]["R2"]["f1@0.5"])[:10]
    worst_rows = [{
        "stem": r["stem"][:40], "layout": r["layout"], "data_source": r["data_source"],
        "f1_R2": r["regions"]["R2"]["f1@0.5"], "f1_R3": r["regions"]["R3"]["f1@0.5"],
        "n_pred_R2": r["regions"]["R2"]["n_pred"], "n_pred_R3": r["regions"]["R3"]["n_pred"],
        "n_fig_gt": r["n_fig_gt"], "text_on": r["labels_on"].get("text", 0), "text_off": r["labels_off"].get("text", 0),
        "section_header_on": r["labels_on"].get("section_header", 0), "section_header_off": r["labels_off"].get("section_header", 0),
        "over_merge_R2": r["regions"]["R2"]["over_merge_boxes"], "over_merge_R3": r["regions"]["R3"]["over_merge_boxes"],
    } for r in worst]

    results: Dict[str, Any] = {
        "config": {
            "pages_file": str((OUT_DIR / "pages.txt").relative_to(ROOT)),
            "n_pages_requested": args.n_pages, "limit": args.limit, "seed": args.seed,
            "pool_pages": len(pool), "selected": len(sel),
            "regions": REGIONS, "primary_iou": PRIMARY_THR, "iou_thresholds": list(IOU_THRS),
            "gt": "OmniDocBench category_type 'figure' (ignore=False)",
            "text_like_gt": list(TEXT_LIKE_GT),
            "on_cache": ON_CACHE_DIR.name, "off_cache": OFF_CACHE_DIR.name,
            "matching": "one-to-one Hungarian on IoU >= thr",
        },
        "counts": {"ok": len(recs), "errors": len(errors), "off_fresh_docling_runs": fresh_off,
                   "pages_with_groups_on": sum(1 for r in recs if r["groups_on"] > 0),
                   "pages_with_groups_off": sum(1 for r in recs if r["groups_off"] > 0)},
        "errors": errors,
        "timing": {
            "off_docling_mean_sec": round(statistics.mean(off_times), 3) if off_times else None,
            "off_docling_median_sec": round(statistics.median(off_times), 3) if off_times else None,
            "on_docling_mean_sec": round(statistics.mean(on_times), 3) if on_times else None,
            "on_docling_median_sec": round(statistics.median(on_times), 3) if on_times else None,
            "stage_median_on_excl_first": stage_med("stage_on"),
            "stage_median_off_excl_first": stage_med("stage_off"),
            "group_function_total_sec": round(group_time, 2),
            "wall_clock_sec": round(time.time() - wall0, 1),
        },
        "regions_summary": {reg: micro_macro(recs, reg) for reg in REGIONS} if recs else {},
        "paired": {
            "R2_vs_R3_OCR_off_effect_on_grouping": paired(recs, "R2", "R3"),
            "R1_vs_R2_grouping_effect_ocr_on": paired(recs, "R1", "R2"),
            "R4_vs_R1_ocr_effect_on_raw_boxes": paired(recs, "R4", "R1"),
            "R3_vs_R1_offOCR_group_vs_onOCR_raw": paired(recs, "R3", "R1"),
        } if recs else {},
        "label_totals": label_totals,
        "breakdown_layout": group_by(recs, "layout") if recs else {},
        "breakdown_data_source": group_by(recs, "data_source", min_pages=1) if recs else {},
        "worst_R3_vs_R2": worst_rows,
        "per_page": recs,
    }
    save_json_atomic(OUT_DIR / "results.json", results)
    write_report(results, OUT_DIR / "report.md")
    log(f"DONE: ok={len(recs)} errors={len(errors)} wall={results['timing']['wall_clock_sec']}s")


# ----------------------------------------------------------------------
# Report (Vietnamese)
# ----------------------------------------------------------------------
def f2(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:.3f}"


def write_report(res: Dict[str, Any], path: Path) -> None:
    cfg, cnt, tm = res["config"], res["counts"], res["timing"]
    rs = res["regions_summary"]
    L: List[str] = []
    L.append("# Đánh giá trích xuất ảnh sơ đồ/hình: OCR bật so với OCR tắt\n")
    if not rs:
        L.append("Không có trang nào được đánh giá thành công. Xem `run.log` và mục lỗi trong `results.json`.\n")
        path.write_text("\n".join(L), encoding="utf-8")
        return
    L.append(f"> **Số trang đánh giá:** {cnt['ok']} / {cfg['selected']} chọn (lỗi: {cnt['errors']}) | "
             f"**Pool:** {cfg['pool_pages']} trang có GT figure và cache OCR-bật | **Seed:** {cfg['seed']} | "
             f"**Ngưỡng IoU chính:** {cfg['primary_iou']}")
    L.append(f"> **Docling/trang:** OCR bật {tm['on_docling_median_sec']} s (trung vị), OCR tắt {tm['off_docling_median_sec']} s (trung vị). "
             f"**Thời gian gom nhóm (group_whole_diagrams, 2 lần/trang):** {tm['group_function_total_sec']} s\n")
    L.append("---\n")

    L.append("## 1. Bảng chính: chất lượng vùng ảnh theo 4 cấu hình (IoU 0.5)\n")
    L.append("| Vùng | Mô tả | Precision (micro) | Recall (micro) | F1 (micro) | F1 (trung bình trang) | F1@0.3 | F1@0.7 | IoU khớp TB |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for reg, desc in REGIONS.items():
        s = rs[reg]
        L.append(f"| {reg} | `{desc}` | {f2(s['micro_precision'])} | {f2(s['micro_recall'])} | **{f2(s['micro_f1'])}** | "
                 f"{f2(s['macro_f1_mean'])} | {f2(s['micro_f1@0.3'])} | {f2(s['micro_f1@0.7'])} | {f2(s['mean_iou_matched'])} |")
    L.append("")

    L.append("## 2. Lỗi cấu trúc (over-merge, phân mảnh, diện tích box lớn nhất)\n")
    L.append("| Vùng | Số box | Box over-merge | Trang có over-merge | Số khối text bị nuốt | GT figure bị phân mảnh | Trang có phân mảnh | Diện tích box lớn nhất TB | Trang box > 45% trang |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for reg in REGIONS:
        s = rs[reg]
        L.append(f"| {reg} | {s['n_pred']} | {s['over_merge_boxes']} | {s['pages_with_over_merge']} / {s['n_pages']} | "
                 f"{s['text_blocks_absorbed']} | {s['fragmented_gt']} / {s['n_gt']} | {s['pages_with_fragmentation']} / {s['n_pages']} | "
                 f"{f2(s['mean_max_pred_area_ratio'])} | {s['pages_max_area_over_45pct']} |")
    L.append("")

    L.append("## 3. So sánh cặp (F1@0.5 theo trang)\n")
    L.append("| So sánh | Số trang | A thắng | Hoà | A thua | Mean Δ | Median Δ | CI 95% Mean Δ (bootstrap) |")
    L.append("|---|---|---|---|---|---|---|---|")
    labels = {
        "R2_vs_R3_OCR_off_effect_on_grouping": "R2 − R3: tắt OCR làm đổi kết quả gom (A = R2)",
        "R1_vs_R2_grouping_effect_ocr_on": "R1 − R2: gom nhóm có giúp so với box thô (OCR bật)",
        "R4_vs_R1_ocr_effect_on_raw_boxes": "R4 − R1: tắt OCR làm đổi box thô (A = R4)",
        "R3_vs_R1_offOCR_group_vs_onOCR_raw": "R3 − R1: gom OCR tắt so với box thô OCR bật",
    }
    for key, title in labels.items():
        p = res["paired"][key]
        ci = p["bootstrap_ci95_mean_delta"]
        ci_s = "—" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
        md = "—" if p["mean_delta_f1"] is None else f"{p['mean_delta_f1']:+.3f}"
        mdd = "—" if p["median_delta_f1"] is None else f"{p['median_delta_f1']:+.3f}"
        L.append(f"| {title} | {p['n_pages']} | {p['a_wins']} | {p['ties']} | {p['a_losses']} | {md} | {mdd} | {ci_s} |")
    L.append("")

    L.append("## 4. Tác động của OCR lên nhãn layout (tổng trên các trang)\n")
    L.append("| Nhãn | Tổng OCR bật | Tổng OCR tắt | Số trang có nhãn (bật) | Số trang có nhãn (tắt) |")
    L.append("|---|---|---|---|---|")
    for k, v in sorted(res["label_totals"].items(), key=lambda kv: -kv[1]["on_total"]):
        L.append(f"| `{k}` | {v['on_total']} | {v['off_total']} | {v['on_pages_with']} | {v['off_pages_with']} |")
    L.append("")

    L.append("## 5. Trang mà OCR tắt làm F1 tụt nhiều nhất (R3 so với R2)\n")
    L.append("| Trang | Layout | Nguồn | F1 R2 | F1 R3 | Box R2 | Box R3 | GT figure | text ON | text OFF | section_header ON | section_header OFF | over-merge R2 | over-merge R3 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for w in res["worst_R3_vs_R2"]:
        L.append(f"| `{w['stem']}` | {w['layout']} | {w['data_source']} | {f2(w['f1_R2'])} | {f2(w['f1_R3'])} | {w['n_pred_R2']} | "
                 f"{w['n_pred_R3']} | {w['n_fig_gt']} | {w['text_on']} | {w['text_off']} | {w['section_header_on']} | "
                 f"{w['section_header_off']} | {w['over_merge_R2']} | {w['over_merge_R3']} |")
    L.append("")

    L.append("## 6. Phân rã theo layout (micro F1)\n")
    L.append("| Layout | Số trang | F1 R1 | F1 R2 | F1 R3 | F1 R4 |")
    L.append("|---|---|---|---|---|---|")
    for g, row in res["breakdown_layout"].items():
        L.append(f"| `{g}` | {row['n_pages']} | {f2(row['R1']['micro_f1'])} | {f2(row['R2']['micro_f1'])} | "
                 f"{f2(row['R3']['micro_f1'])} | {f2(row['R4']['micro_f1'])} |")
    L.append("")
    L.append("## 7. Phân rã theo nguồn dữ liệu (micro F1)\n")
    L.append("| Nguồn | Số trang | F1 R1 | F1 R2 | F1 R3 | F1 R4 |")
    L.append("|---|---|---|---|---|---|")
    for g, row in res["breakdown_data_source"].items():
        L.append(f"| `{g}` | {row['n_pages']} | {f2(row['R1']['micro_f1'])} | {f2(row['R2']['micro_f1'])} | "
                 f"{f2(row['R3']['micro_f1'])} | {f2(row['R4']['micro_f1'])} |")
    L.append("")

    L.append("## 8. Thời gian mỗi trang (bỏ trang đầu, trung vị, giây)\n")
    sO, sF = tm["stage_median_on_excl_first"], tm["stage_median_off_excl_first"]
    L.append("| Stage | OCR bật | OCR tắt |")
    L.append("|---|---|---|")
    for k in sorted(set(sO) | set(sF)):
        L.append(f"| `{k}` | {sO.get(k, '—')} | {sF.get(k, '—')} |")
    L.append("")

    L.append("## 9. Giới hạn và lưu ý\n")
    L.append("- GT `figure` của OmniDocBench gồm cả ảnh chụp, biểu đồ và sơ đồ, không chỉ sơ đồ khối. Đây là proxy cho mục tiêu trích xuất ảnh sơ đồ, không phải đo trực tiếp trên sơ đồ.")
    L.append(f"- Mẫu: {cnt['ok']} trang, lấy phân tầng layout × data_source từ pool {cfg['pool_pages']} trang có GT figure và cache OCR-bật. Không có tập giữ lại (held-out), và `group_whole_diagrams_image` cùng các guard đã được phát triển khi nhìn vào các trang OmniDocBench.")
    L.append("- Ngưỡng guard (diện tích 35%, >=2 section_header, khoảng cách 160 px...) được đặt sẵn trong hàm, không được tối ưu lại trong đánh giá này.")
    L.append("- Thứ tự đọc và `order` không được dùng. Các GT figure có `order=None` vẫn được tính.")
    L.append("- Định nghĩa over-merge: box dự đoán phủ >=50% diện tích của >=2 GT figure, hoặc >=2 khối text-like (liệt kê ở `results.json`). Phân mảnh: GT figure bị >=2 box có IoU trong [0.1, 0.5).")
    L.append("- OCR tắt chỉ giữ layout và cấu trúc, nên text rỗng. Các guard dựa vào nhãn (section_header, list_item, title) sẽ hoạt động khác nếu layout model vẫn xuất các nhãn này.")
    L.append(f"- Thời gian: OCR tắt {tm['off_docling_median_sec']} s/trang (trung vị), OCR bật {tm['on_docling_median_sec']} s/trang (trung vị), cả hai trên GPU. Trang đầu của mỗi lần chạy có warm-up nên đã bỏ khi tính trung vị theo stage.")
    L.append("- Lỗi trong `results.json` > `errors`; không tính vào thống kê.\n")
    path.write_text("\n".join(L).strip() + "\n", encoding="utf-8")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Diagram/figure crop evaluation: OCR on vs off.")
    ap.add_argument("--n-pages", type=int, default=100, help="number of pages to sample (default 100)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="keep first N selected pages (smoke test)")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    run(parse_args(argv))


if __name__ == "__main__":
    main()
