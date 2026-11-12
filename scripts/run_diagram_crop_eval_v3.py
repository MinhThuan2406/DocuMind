"""
Diagram / picture cluster grouping v2: leave-source-out evaluation (OCR OFF cache, no Docling run).

Regions compared on every OmniDocBench page with >= 1 GT 'figure' (613 pages, 1385 GT):
  R1 = raw Docling picture boxes (baseline, no grouping)
  R3 = old production grouping group_whole_diagrams_image (via run_diagram_crop_eval_v2.group_call)
  R5 = diagram_grouping_v2.group_whole_diagrams_image_v2 with one GroupParamsV2 configuration

Protocol (no tuning on test data):
  * Pages are assigned to K=5 folds by data_source (never by page). The assignment is a fixed greedy
    balance on GT figure counts and broken counts, see assign_folds().
  * For each held-out fold f, the configuration is chosen on the other 4 folds only:
      maximise whole@0.5 on broken GT, subject to
        harm   : intact GT lost by R5 (R1 whole@0.5 true, R5 false) <= HARM_FRAC x intact GT (train)
        near   : box-level merge rate on near-distinct pages <= R1 rate + NEAR_SLACK (train)
      ties: fewer harms, then lower near merge rate, then grid order.
      If no configuration is feasible, the one with the fewest harms is taken and flagged.
  * The reported R5 numbers are out-of-fold: every page is scored with the configuration chosen
    for its own fold. A configuration chosen on all pages is also reported, for deployment only.

Metric definitions follow run_diagram_crop_eval_v2 (whole-capture, split, merge, near-distinct pairs
with edge gap <= 80 px). The 80 px gap is an evaluation definition kept for comparability; the
algorithm itself uses only relative distances.

Usage:
  .venv\\Scripts\\python scripts\\run_diagram_crop_eval_v3.py
  .venv\\Scripts\\python scripts\\run_diagram_crop_eval_v3.py --limit 40 --no-viz
"""

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from eval_utils import (  # noqa: E402
    boxes_covering_gts,
    classify_gt_completeness,
    split_count,
    whole_capture,
)
from diagram_grouping_v2 import GroupParamsV2, group_whole_diagrams_image_v2  # noqa: E402
from ir_postprocessor import load_font  # noqa: E402
from run_diagram_crop_eval import (  # noqa: E402
    DATASET_JSON,
    IMAGES_DIR,
    OFF_CACHE_DIR,
    PRED_FIG_LABELS,
    ROOT,
    gt_boxes_by_kind,
    item_boxes,
)
from run_diagram_crop_eval_v2 import group_call  # noqa: E402
from run_e2e_reading_order_eval import Logger, load_cache, save_json_atomic  # noqa: E402
from test_diagram_grouping import detect_visual_containers_opencv  # noqa: E402

OUT_DIR = ROOT / "outputs/05_diagram_crop_eval"
VIZ_DIR = OUT_DIR / "viz_v3"
THRS = (0.3, 0.5, 0.7)
PIECE_THR = 0.7
SPLIT_CONT_THR = 0.5
COVER_THR = 0.5
NEAR_GAP_PX = 80.0
N_FOLDS = 5
HARM_FRAC = 0.01
NEAR_SLACK = 0.01
SEED = 42
SUBSETS = ("broken", "intact", "all")

GRID = {
    "k_gap": (0.1, 0.25, 0.5, 1.0),
    "tau_align": (0.0, 0.5),
    "tau_size": (0.0, 0.2),
    "evidence": ("tight", "container", "both"),
    "big_frac": (0.05, 2.0),          # 2.0 = anti-engulf rule disabled
    "max_group_frac": (0.35, 0.6),
    "pad_rel": (0.0, 0.005),
}


def combo_list() -> List[Dict[str, Any]]:
    keys = list(GRID)
    return [dict(zip(keys, vals)) for vals in product(*[GRID[k] for k in keys])]


def combo_id(c: Dict[str, Any]) -> str:
    return (f"kg{c['k_gap']}_al{c['tau_align']}_sz{c['tau_size']}_ev{c['evidence']}"
            f"_bg{c['big_frac']}_mg{c['max_group_frac']}_pd{c['pad_rel']}")


def rate(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den else None


def wilson(k: int, n: int, z: float = 1.96) -> Optional[Tuple[float, float]]:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)


# ----------------------------------------------------------------------
# Page preparation (OCR-off cache, GT, R1, R3, containers)
# ----------------------------------------------------------------------
def near_pairs(fig: List[List[float]]) -> List[List[int]]:
    out = []
    for a in range(len(fig)):
        for b in range(a + 1, len(fig)):
            A, B = fig[a], fig[b]
            ix = min(A[2], B[2]) - max(A[0], B[0])
            iy = min(A[3], B[3]) - max(A[1], B[1])
            if ix > 0 and iy > 0:
                continue
            dx = max(0.0, max(A[0], B[0]) - min(A[2], B[2]))
            dy = max(0.0, max(A[1], B[1]) - min(A[3], B[3]))
            if math.hypot(dx, dy) <= NEAR_GAP_PX:
                out.append([a, b])
    return out


def eval_boxes(pg: Dict[str, Any], boxes: List[List[float]]) -> Dict[str, Any]:
    fig = pg["fig_gt"]
    return {
        "whole": [[whole_capture(g, boxes, t) for t in THRS] for g in fig],
        "split": [split_count(g, boxes, SPLIT_CONT_THR) for g in fig],
        "cover": boxes_covering_gts(boxes, fig, COVER_THR) if boxes and fig else [[] for _ in boxes],
    }


def prepare_pages(log: Logger, limit: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    data = json.loads(DATASET_JSON.read_text(encoding="utf-8"))
    pool = []
    for i, it in enumerate(data):
        fig, _ = gt_boxes_by_kind(it)
        if fig and (IMAGES_DIR / it["page_info"]["image_path"]).exists():
            pool.append((i, it))
    if limit:
        pool = pool[:limit]
    log(f"Pages with >= 1 GT figure: {len(pool)}")

    pages: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for pos, (idx, item) in enumerate(pool, 1):
        p = item["page_info"]
        attr = p.get("page_attribute", {})
        img_rel = p["image_path"]
        stem = Path(img_rel).stem
        cache = load_cache(OFF_CACHE_DIR / f"{stem}.json")
        if cache is None or cache.get("status") != "ok":
            errors.append({"stem": stem, "error": "missing OCR-off cache"})
            log(f"[{pos}/{len(pool)}] {stem[:24]} MISSING OFF CACHE")
            continue
        try:
            items = [it for it in cache["items"] if it.get("bbox") and len(it["bbox"]) == 4]
            W, H = cache["image_size"]
            fig_gt, _ = gt_boxes_by_kind(item)
            r1 = item_boxes(items, PRED_FIG_LABELS)
            others = [it["bbox"] for it in items if it.get("label") not in PRED_FIG_LABELS]
            img_path = IMAGES_DIR / img_rel
            containers = detect_visual_containers_opencv(str(img_path))
            r3 = group_call(items, img_path, (W, H))["_r3_boxes"]
            gts = []
            for g in fig_gt:
                cls, n_p = classify_gt_completeness(g, r1, piece_thr=PIECE_THR, iou_thr=0.5)
                gts.append({"bbox": [round(v, 1) for v in g], "cls": cls, "n_pieces": n_p})
            pages.append({
                "idx": idx, "stem": stem, "img_rel": img_rel, "data_source": attr.get("data_source", "unknown"),
                "layout": attr.get("layout", "unknown"), "W": W, "H": H, "page_area": float(W * H),
                "items": items, "others": others, "containers": containers,
                "fig_gt": fig_gt, "gts": gts, "near": near_pairs(fig_gt),
                "r1_boxes": r1, "r3_boxes": r3,
            })
        except Exception as e:  # keep going, record
            errors.append({"stem": stem, "error": f"{type(e).__name__}: {str(e)[:300]}"})
            log(f"[{pos}/{len(pool)}] {stem[:24]} ERROR {type(e).__name__}: {str(e)[:200]}")
        if pos % 100 == 0 or pos == len(pool):
            log(f"  prepared {pos}/{len(pool)}")
    return pages, errors


# ----------------------------------------------------------------------
# Aggregation over page subsets
# ----------------------------------------------------------------------
def summarize(pages, res_list, page_ids: Sequence[int], subset: str) -> Dict[str, Any]:
    n = 0
    wh = {f"{t:.1f}": 0 for t in THRS}
    split2 = 0
    cands = merged = 0
    for pi in page_ids:
        pg, r = pages[pi], res_list[pi]
        sel = [k for k, g in enumerate(pg["gts"]) if subset == "all" or g["cls"] == subset]
        if not sel:
            continue
        for k in sel:
            n += 1
            for ti, t in enumerate(THRS):
                wh[f"{t:.1f}"] += int(r["whole"][k][ti])
            split2 += int(r["split"][k] >= 2)
        selset = set(sel)
        for cov in r["cover"]:
            if any(c in selset for c in cov):
                cands += 1
                if len(cov) >= 2:
                    merged += 1
    out: Dict[str, Any] = {"n_gt": n, "split_ge2": split2, "split_rate": rate(split2, n),
                           "merge_cands": cands, "merged": merged, "merge_rate": rate(merged, cands)}
    for key, v in wh.items():
        out[f"whole@{key}"] = v
        out[f"whole_rate@{key}"] = rate(v, n)
    return out


def near_stats(pages, res_list, page_ids: Sequence[int]) -> Dict[str, Any]:
    pairs = pairs_merged = box_c = box_m = 0
    for pi in page_ids:
        pg, r = pages[pi], res_list[pi]
        if not pg["near"]:
            continue
        pairs += len(pg["near"])
        for a, b in pg["near"]:
            if any(a in cov and b in cov for cov in r["cover"]):
                pairs_merged += 1
        for cov in r["cover"]:
            if cov:
                box_c += 1
                if len(cov) >= 2:
                    box_m += 1
    return {"pairs": pairs, "pairs_merged": pairs_merged, "pair_rate": rate(pairs_merged, pairs),
            "box_cands": box_c, "box_merged": box_m, "box_rate": rate(box_m, box_c)}


def harm_stats(pages, res_list, r1_list, page_ids: Sequence[int]) -> Dict[str, int]:
    n_int = harm = gain_broken = 0
    for pi in page_ids:
        pg, r, r1 = pages[pi], res_list[pi], r1_list[pi]
        for k, g in enumerate(pg["gts"]):
            if g["cls"] == "intact":
                n_int += 1
                if r1["whole"][k][1] and not r["whole"][k][1]:
                    harm += 1
            if g["cls"] == "broken" and r["whole"][k][1]:
                gain_broken += 1
    return {"n_intact": n_int, "harm_intact": harm, "broken_whole05": gain_broken}


def train_score(pages, res_c, r1_list, page_ids: Sequence[int]) -> Dict[str, Any]:
    h = harm_stats(pages, res_c, r1_list, page_ids)
    nc = near_stats(pages, res_c, page_ids)
    n1 = near_stats(pages, r1_list, page_ids)
    near_rate = nc["box_rate"] or 0.0
    near_r1 = n1["box_rate"] or 0.0
    feasible = h["harm_intact"] <= HARM_FRAC * h["n_intact"] and near_rate <= near_r1 + NEAR_SLACK
    return {**h, "near_rate": near_rate, "near_rate_R1": near_r1, "feasible": feasible}


def select_config(pages, res_all, r1_list, page_ids: Sequence[int]) -> Tuple[int, Dict[str, Any], str]:
    scores = [(ci, train_score(pages, res_all[ci], r1_list, page_ids)) for ci in range(len(res_all))]
    feas = [(ci, s) for ci, s in scores if s["feasible"]]
    if feas:
        ci, s = min(feas, key=lambda t: (-t[1]["broken_whole05"], t[1]["harm_intact"], t[1]["near_rate"], t[0]))
        return ci, s, "feasible"
    ci, s = min(scores, key=lambda t: (t[1]["harm_intact"], -t[1]["broken_whole05"], t[1]["near_rate"], t[0]))
    return ci, s, "no_feasible_config"


def assign_folds(pages) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    by_src: Dict[str, List[int]] = defaultdict(lambda: [0, 0])  # [n_broken, n_gt]
    for pg in pages:
        for g in pg["gts"]:
            by_src[pg["data_source"]][1] += 1
            by_src[pg["data_source"]][0] += int(g["cls"] == "broken")
    srcs = sorted(by_src, key=lambda s: (-by_src[s][0], -by_src[s][1], s))
    stats = [{"fold": f, "sources": [], "n_broken": 0, "n_gt": 0} for f in range(N_FOLDS)]
    fold_of: Dict[str, int] = {}
    for s in srcs:
        f = min(range(N_FOLDS), key=lambda k: (stats[k]["n_broken"], stats[k]["n_gt"], k))
        fold_of[s] = f
        stats[f]["sources"].append(s)
        stats[f]["n_broken"] += by_src[s][0]
        stats[f]["n_gt"] += by_src[s][1]
    return fold_of, stats


# ----------------------------------------------------------------------
# Analysis of the missed GT (no algorithm change)
# ----------------------------------------------------------------------
def _inter_area(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def missed_profile(pages) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = []
    for pi, pg in enumerate(pages):
        for k, g in enumerate(pg["gts"]):
            if g["cls"] != "missed":
                continue
            gb = g["bbox"]
            ga = _area(gb)
            ratio = ga / pg["page_area"]
            in_cont = max((_inter_area(gb, c) / ga for c in pg["containers"]), default=0.0) if ga > 0 else 0.0
            cont_iou = 0.0
            for c in pg["containers"]:
                inter = _inter_area(gb, c)
                uni = ga + _area(c) - inter
                if uni > 0:
                    cont_iou = max(cont_iou, inter / uni)
            other_ov = max((_inter_area(gb, o) / ga for o in pg["others"]), default=0.0) if ga > 0 else 0.0
            rows.append({"pi": pi, "gid_local": k, "stem": pg["stem"], "data_source": pg["data_source"],
                         "layout": pg["layout"], "area_ratio": ratio, "in_container": in_cont >= 0.9,
                         "container_iou_ge_0.5": cont_iou >= 0.5, "other_box_covers_ge_0.5": other_ov >= 0.5})

    def bucket(r: float) -> str:
        if r < 0.01:
            return "<1%"
        if r < 0.05:
            return "1-5%"
        if r < 0.2:
            return "5-20%"
        return ">=20%"

    summary = {
        "n_missed": len(rows),
        "area_bucket": dict(Counter(bucket(r["area_ratio"]) for r in rows)),
        "in_container_ge_0.9": sum(r["in_container"] for r in rows),
        "container_iou_ge_0.5": sum(r["container_iou_ge_0.5"] for r in rows),
        "other_box_covers_ge_0.5": sum(r["other_box_covers_ge_0.5"] for r in rows),
        "by_data_source": dict(Counter(r["data_source"] for r in rows).most_common()),
        "by_layout": dict(Counter(r["layout"] for r in rows).most_common()),
    }
    return rows, summary


# ----------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------
def draw_case(img_rel: str, title: str, note: str, gts: List[List[float]], focus: int,
              pred: List[List[float]], out_path: Path) -> None:
    img = Image.open(IMAGES_DIR / img_rel).convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size
    font = load_font(max(18, W // 60))
    for k, b in enumerate(gts):
        draw.rectangle(b, outline=(30, 90, 230), width=6 if k == focus else 3)
    for pb in pred:
        draw.rectangle(pb, outline=(220, 30, 30), width=4)
    banner = int(getattr(font, "size", 18) * 2.6)
    draw.rectangle([0, 0, W, banner], fill=(20, 20, 20))
    draw.text((12, 6), title, fill=(255, 255, 255), font=font)
    draw.text((12, banner // 2 + 2), note, fill=(255, 215, 0), font=font)
    if W > 1600:
        img = img.resize((1600, int(H * 1600 / W)))
    img.save(out_path, "PNG")


def make_viz(pages, res_oof_boxes, gt_rows, log) -> List[Dict[str, Any]]:
    rng = random.Random(SEED)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    groups = {
        "broken_fixed_by_R5": [r for r in gt_rows if r["cls"] == "broken" and r["R5_whole@0.5"]],
        "broken_still_wrong_R5": [r for r in gt_rows if r["cls"] == "broken" and not r["R5_whole@0.5"]],
        "intact_harmed_by_R5": [r for r in gt_rows if r["cls"] == "intact" and r["R1_whole@0.5"]
                                and not r["R5_whole@0.5"]],
    }
    out = []
    for key, pool_rows in groups.items():
        pick = rng.sample(pool_rows, min(6, len(pool_rows)))
        for n, r in enumerate(sorted(pick, key=lambda x: (x["pi"], x["gt_k"])), 1):
            pg = pages[r["pi"]]
            pred = res_oof_boxes[r["pi"]]
            name = f"{key}_{n:02d}.png"
            title = f"{key} | {pg['stem'][:34]} | GT blue, R5(v2, out-of-fold) red | cls={r['cls']}"
            note = f"GT figs={len(pg['fig_gt'])} R5 boxes={len(pred)} | focus GT={[round(v) for v in r['bbox']]}"
            draw_case(pg["img_rel"], title, note, pg["fig_gt"], r["gt_k"], pred, VIZ_DIR / name)
            out.append({"case": key, "file": f"viz_v3/{name}", "stem": pg["stem"], "cls": r["cls"]})
    log(f"Saved {len(out)} visualisations to {VIZ_DIR}")
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = Logger(OUT_DIR / "run_v3.log")
    t0 = time.time()
    log("=" * 80)
    log(f"Diagram grouping v2 evaluation start | argv={' '.join(sys.argv[1:])}")

    pages, errors = prepare_pages(log, args.limit)
    N = len(pages)
    fig_total = sum(len(pg["gts"]) for pg in pages)
    log(f"Prepared pages={N} GT={fig_total} errors={len(errors)}")

    r1_list = [eval_boxes(pg, pg["r1_boxes"]) for pg in pages]
    r3_list = [eval_boxes(pg, pg["r3_boxes"]) for pg in pages]

    combos = combo_list()
    log(f"Grid: {len(combos)} configurations")
    res_all: List[List[Dict[str, Any]]] = []
    for ci, c in enumerate(combos):
        params = GroupParamsV2(**c)
        res_c = []
        for pg in pages:
            boxes = group_whole_diagrams_image_v2(pg["items"], (pg["W"], pg["H"]),
                                                  containers=pg["containers"], params=params)["boxes"]
            res_c.append(eval_boxes(pg, boxes))
        res_all.append(res_c)
        if (ci + 1) % 32 == 0 or ci + 1 == len(combos):
            log(f"  grid {ci + 1}/{len(combos)} configurations evaluated")

    fold_of_src, fold_stats = assign_folds(pages)
    fold_of_page = [fold_of_src[pg["data_source"]] for pg in pages]
    log("Folds: " + "; ".join(f"f{s['fold']}={s['sources']} (broken {s['n_broken']}, GT {s['n_gt']})"
                              for s in fold_stats))

    # Leave-source-out selection and out-of-fold scoring
    fold_rows = []
    chosen_fold: List[int] = []
    res_oof: List[Dict[str, Any]] = [None] * N
    for f in range(N_FOLDS):
        train_ids = [i for i in range(N) if fold_of_page[i] != f]
        test_ids = [i for i in range(N) if fold_of_page[i] == f]
        ci, s_train, status = select_config(pages, res_all, r1_list, train_ids)
        chosen_fold.append(ci)
        for i in test_ids:
            res_oof[i] = res_all[ci][i]
        held = summarize(pages, res_all[ci], test_ids, "broken")
        held_all = harm_stats(pages, res_all[ci], r1_list, test_ids)
        fold_rows.append({"fold": f, "config_id": combo_id(combos[ci]), "params": combos[ci], "status": status,
                          "train": s_train, "heldout_broken_whole@0.5": held["whole@0.5"], "heldout_broken_n": held["n_gt"],
                          "heldout_harm_intact": held_all["harm_intact"], "heldout_n_intact": held_all["n_intact"],
                          "heldout_pages": len(test_ids)})
        log(f"Fold {f}: config={combo_id(combos[ci])} status={status} "
            f"train broken whole@0.5={s_train['broken_whole05']} harm={s_train['harm_intact']}/{s_train['n_intact']} "
            f"| held-out broken whole@0.5={held['whole@0.5']}/{held['n_gt']} harm={held_all['harm_intact']}")

    subsets = {}
    all_ids = list(range(N))
    for s in SUBSETS:
        subsets[s] = {"R1": summarize(pages, r1_list, all_ids, s), "R3": summarize(pages, r3_list, all_ids, s),
                      "R5_oof": summarize(pages, res_oof, all_ids, s)}
    near = {"R1": near_stats(pages, r1_list, all_ids), "R3": near_stats(pages, r3_list, all_ids),
            "R5_oof": near_stats(pages, res_oof, all_ids)}
    harm = {"R3": harm_stats(pages, r3_list, r1_list, all_ids), "R5_oof": harm_stats(pages, res_oof, r1_list, all_ids)}

    # Deployment configuration (chosen on all pages) and in-sample sensitivity
    ci_dep, s_dep, status_dep = select_config(pages, res_all, r1_list, all_ids)
    dep_params = combos[ci_dep]
    sens = []
    for key, vals in GRID.items():
        for v in vals:
            cc = dict(dep_params, **{key: v})
            cid = combos.index(cc)
            row = train_score(pages, res_all[cid], r1_list, all_ids)
            sens.append({"param": key, "value": v, "is_deployed": v == dep_params[key],
                         "broken_whole05_insample": row["broken_whole05"], "harm_intact": row["harm_intact"],
                         "near_rate": row["near_rate"], "feasible": row["feasible"]})

    all_scores = [(ci, train_score(pages, res_all[ci], r1_list, all_ids)) for ci in range(len(combos))]
    n_feasible_insample = sum(1 for _, s_ in all_scores if s_["feasible"])
    top = sorted(all_scores, key=lambda t: (not t[1]["feasible"], -t[1]["broken_whole05"],
                                             t[1]["harm_intact"], t[1]["near_rate"], t[0]))[:10]
    top_rows = [{"config_id": combo_id(combos[ci]), "params": combos[ci], "feasible": s["feasible"],
                 "broken_whole05": s["broken_whole05"], "harm_intact": s["harm_intact"],
                 "near_rate": s["near_rate"]} for ci, s in top]

    # Per-GT rows (out-of-fold R5)
    gt_rows: List[Dict[str, Any]] = []
    for pi, pg in enumerate(pages):
        for k, g in enumerate(pg["gts"]):
            gt_rows.append({"pi": pi, "gt_k": k, "stem": pg["stem"], "data_source": pg["data_source"],
                            "layout": pg["layout"], "bbox": g["bbox"], "cls": g["cls"],
                            "area_ratio": round(_area(g["bbox"]) / pg["page_area"], 5),
                            "R1_whole@0.5": bool(r1_list[pi]["whole"][k][1]),
                            "R3_whole@0.5": bool(r3_list[pi]["whole"][k][1]),
                            "R5_whole@0.5": bool(res_oof[pi]["whole"][k][1]),
                            "R5_config": combo_id(combos[chosen_fold[fold_of_page[pi]]])})

    # Causes for broken GT still wrong under R5 (out-of-fold)
    cause_counts: Counter = Counter()
    for r in gt_rows:
        if r["cls"] != "broken" or r["R5_whole@0.5"]:
            continue
        pi, k = r["pi"], r["gt_k"]
        boxes = res_oof_boxes_for(pages, res_oof, pi, combos, chosen_fold, fold_of_page)
        g = pages[pi]["fig_gt"][k]
        covering = [b for b in boxes if _inter_area(g, b) / max(1e-9, _area(g)) >= COVER_THR]
        if not covering:
            cat = "no_box_covers_gt"
        elif any(_area(b) > 1.5 * _area(g) for b in covering):
            cat = "box_too_large"
        else:
            cat = "box_too_small_or_shifted"
        r["R5_fail_cause"] = cat
        cause_counts[cat] += 1

    missed_rows, missed_sum = missed_profile(pages)

    viz = []
    if not args.no_viz:
        oof_boxes = [res_oof_boxes_for(pages, res_oof, pi, combos, chosen_fold, fold_of_page) for pi in range(N)]
        viz = make_viz(pages, oof_boxes, gt_rows, log)

    results: Dict[str, Any] = {
        "config": {"pages": N, "gt_total": fig_total, "errors": len(errors), "folds": N_FOLDS,
                   "grid_size": len(combos), "grid": {k: list(v) for k, v in GRID.items()},
                   "harm_frac": HARM_FRAC, "near_slack": NEAR_SLACK, "thresholds": list(THRS),
                   "piece_thr": PIECE_THR, "split_thr": SPLIT_CONT_THR, "cover_thr": COVER_THR,
                   "near_gap_px_eval": NEAR_GAP_PX, "cache": OFF_CACHE_DIR.name},
        "errors": errors,
        "fold_assignment": fold_stats,
        "folds": fold_rows,
        "subsets_oof": subsets,
        "near_distinct": near,
        "harm_vs_R1": harm,
        "deployment": {"config_id": combo_id(dep_params), "params": dep_params, "status": status_dep,
                       "insample": s_dep},
        "sensitivity_insample": sens,
        "top10_insample": top_rows,
        "n_feasible_insample": n_feasible_insample,
        "n_configs": len(combos),
        "r5_fail_causes_oof": dict(cause_counts),
        "missed_summary": missed_sum,
        "missed_rows": missed_rows,
        "per_gt": gt_rows,
        "viz": viz,
    }
    save_json_atomic(OUT_DIR / "results_v3.json", results)
    log(f"Saved {OUT_DIR / 'results_v3.json'}")
    write_report(results, OUT_DIR / "report_v3.md")
    log(f"DONE: pages={N} gt={fig_total} errors={len(errors)} wall={round(time.time() - t0, 1)}s")


def res_oof_boxes_for(pages, res_oof, pi, combos, chosen_fold, fold_of_page) -> List[List[float]]:
    """Recomputes the out-of-fold R5 boxes of one page with the configuration chosen for its fold."""
    pg = pages[pi]
    params = GroupParamsV2(**combos[chosen_fold[fold_of_page[pi]]])
    return group_whole_diagrams_image_v2(pg["items"], (pg["W"], pg["H"]),
                                         containers=pg["containers"], params=params)["boxes"]


# ----------------------------------------------------------------------
# Report (Vietnamese)
# ----------------------------------------------------------------------
def _p(v: Optional[float]) -> str:
    return "—" if v is None else f"{100 * v:.1f}%"


def _ci(k: int, n: int) -> str:
    w = wilson(k, n)
    return "—" if w is None else f"[{100 * w[0]:.1f}%, {100 * w[1]:.1f}%]"


def write_report(res: Dict[str, Any], path: Path) -> None:
    cfg = res["config"]
    sub = res["subsets_oof"]
    near = res["near_distinct"]
    L: List[str] = []
    L.append("# Đánh giá gom cụm ảnh v2 (leave-source-out), OCR tắt\n")
    L.append(f"> **Phạm vi:** {cfg['pages']} trang OmniDocBench có GT figure | **GT figure:** {cfg['gt_total']} | "
             f"**Lưới:** {cfg['grid_size']} cấu hình | **Fold:** {cfg['folds']} (nhóm theo data_source) | "
             f"**Lỗi trang:** {cfg['errors']} | **Cache:** `{cfg['cache']}` (không chạy lại Docling)\n")
    L.append("---\n")

    L.append("## 1. Thuật toán v2 thay đổi gì\n")
    L.append("Module `scripts/diagram_grouping_v2.py`, hàm `group_whole_diagrams_image_v2`. Hàm cũ `group_whole_diagrams_image` không bị sửa.\n")
    L.append("- **Giữ:** box thô của Docling làm mặc định. Hình đơn không có bằng chứng gộp thì giữ nguyên. Chỉ gộp các mảnh `picture`/`figure` với nhau. Container OpenCV giữ lại làm bằng chứng và làm giới hạn kích thước (diện tích không quá 35% trang, hệ số 1.35 so với hợp các mảnh).")
    L.append("- **Bỏ:** hút `formula`/`table` vào cụm (v1: formula_absorbed 10, table_absorbed 10 ca hỏng). Chiến lược C căn hàng bằng ngưỡng pixel tuyệt đối 35 px và 140 px. Ngưỡng khoảng cách pixel tuyệt đối 45 px (pad) và 160 px (gap). Snap container lỏng (IoU > 0.6 hoặc container dưới 1.35 lần hợp, v1: container_snap 37 ca hỏng).")
    L.append("- **Đổi:** mọi khoảng cách là tương đối: khoảng hở cạnh / đường chéo nhỏ nhất của hai mảnh (`k_gap`). Yêu cầu căn chỉnh (`tau_align`: chồng lấn tối thiểu theo một trục) và cỡ tương đương (`tau_size`). Bằng chứng thứ hai là hai mảnh cùng nằm trong một container chặt chỉ chứa picture. Hai mảnh lớn (diện tích >= `big_frac` của trang) chỉ gộp khi gần chạm nhau (anti-engulf). Snap container chỉ khi IoU >= 0.85 với hợp và container không chứa box khác. Nhóm chỉ được nhận khi hợp không vượt `max_group_frac` diện tích trang. Padding tương đối `pad_rel` x max(W, H).")
    L.append("- **Hàng rào trước khi gộp:** một box formula/table nằm trong khoảng giữa hai mảnh (chồng lấn quá 50% với hợp hai mảnh và không nằm trong mảnh nào) sẽ chặn cặp ghép.\n")

    L.append("## 2. Giao thức đánh giá (không tune trên tập kiểm tra)\n")
    L.append("- Trang được gán vào 5 fold theo `data_source` (một nguồn chỉ thuộc một fold). Việc gán là tham lam cố định theo số ca vỡ và số GT, không dùng kết quả thuật toán.")
    L.append("- Với mỗi fold giữ lại: chọn cấu hình chỉ trên 4 fold còn lại, mục tiêu tối đa whole@0.5 trên ca vỡ, ràng buộc cứng: (a) số GT nguyên vẹn bị R5 làm hỏng <= 1% số GT nguyên vẹn của tập huấn luyện; (b) tỉ lệ box gộp nhầm hình riêng gần nhau <= R1 + 1 điểm %.")
    L.append("- Số báo cáo của R5 là **out-of-fold**: mỗi trang được chấm bằng cấu hình chọn cho fold của nó. Cấu hình chọn trên toàn bộ 613 trang chỉ dùng để triển khai, không phải kết quả báo cáo.\n")
    L.append("| Fold | Nguồn dữ liệu | Số GT | Số vỡ |")
    L.append("|---|---|---|---|")
    for s in res["fold_assignment"]:
        L.append(f"| {s['fold']} | {', '.join(f'`{x}`' for x in s['sources'])} | {s['n_gt']} | {s['n_broken']} |")
    L.append("")

    L.append("### 2.1 Cấu hình được chọn theo fold\n")
    L.append("| Fold | Cấu hình | Trạng thái | Huấn luyện: vỡ whole@0.5 | Huấn luyện: nguyên vẹn bị hỏng | Giữ lại: vỡ whole@0.5 | Giữ lại: nguyên vẹn bị hỏng |")
    L.append("|---|---|---|---|---|---|---|")
    for fr in res["folds"]:
        t = fr["train"]
        L.append(f"| {fr['fold']} | `{fr['config_id']}` | {fr['status']} | {t['broken_whole05']} | "
                 f"{t['harm_intact']} / {t['n_intact']} | {fr['heldout_broken_whole@0.5']} / {fr['heldout_broken_n']} | "
                 f"{fr['heldout_harm_intact']} / {fr['heldout_n_intact']} |")
    L.append("")
    dep = res["deployment"]
    L.append(f"**Cấu hình triển khai** (chọn trên toàn bộ 613 trang, trạng thái `{dep['status']}`): `{dep['config_id']}`. "
             "Số liệu trong-mẫu của nó chỉ để tham khảo, không đưa vào báo cáo tổng.\n")

    L.append("## 3. Bảng R1 / R3 / R5 (v2, out-of-fold)\n")
    L.append("Tỉ lệ whole-capture là tỉ lệ GT có đúng một box dự đoán có IoU >= ngưỡng. Khoảng tin cậy là Wilson 95% trên số GT.\n")
    L.append("| Tập | Vùng | n | whole@0.3 | whole@0.5 (95% CI) | whole@0.7 | split (>=2 mảnh) | merge (box gộp >=2 GT / box phủ GT của tập) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in ("broken", "intact", "all"):
        for reg, label in (("R1", "R1 thô"), ("R3", "R3 cũ"), ("R5_oof", "R5 v2 (OOF)")):
            d = sub[s][reg]
            n = d["n_gt"]
            k05 = d["whole@0.5"]
            L.append(f"| {s} | {label} | {n} | {_p(d['whole_rate@0.3'])} | {_p(d['whole_rate@0.5'])} {_ci(k05, n)} | "
                     f"{_p(d['whole_rate@0.7'])} | {_p(d['split_rate'])} | "
                     f"{d['merged']} / {d['merge_cands']} ({_p(d['merge_rate'])}) |")
    L.append("")
    h3, h5 = res["harm_vs_R1"]["R3"], res["harm_vs_R1"]["R5_oof"]
    L.append("**Số GT được sửa và bị hỏng so với R1 (whole@0.5, toàn bộ 613 trang):**\n")
    L.append("| Vùng | GT nguyên vẹn bị hỏng | Tỉ lệ hỏng trên nguyên vẹn (95% CI) | GT vỡ được sửa (trên tổng GT vỡ) |")
    L.append("|---|---|---|---|")
    for lab, h, reg in (("R3 cũ", h3, "R3"), ("R5 v2 (OOF)", h5, "R5_oof")):
        L.append(f"| {lab} | {h['harm_intact']} / {h['n_intact']} | {_p(rate(h['harm_intact'], h['n_intact']))} "
                 f"{_ci(h['harm_intact'], h['n_intact'])} | {h['broken_whole05']} / {sub['broken'][reg]['n_gt']} |")
    L.append("")
    ok_harm = h5["harm_intact"] <= HARM_FRAC * h5["n_intact"]
    L.append(f"Ràng buộc hại (<= 1%): R5 v2 **{'ĐẠT' if ok_harm else 'KHÔNG ĐẠT'}** "
             f"({h5['harm_intact']} / {h5['n_intact']} = {_p(rate(h5['harm_intact'], h5['n_intact']))}).\n")

    L.append("## 4. Hình riêng biệt gần nhau (gộp nhầm)\n")
    L.append("Cặp GT không chồng lấn, khoảng cách cạnh <= 80 px (định nghĩa đánh giá như v2). Box gộp nhầm là box phủ >= 0.5 của >= 2 GT.\n")
    L.append("| Vùng | Số cặp gần nhau | Cặp bị một box phủ | Tỉ lệ cặp bị gộp | Box trên trang có cặp gần nhau | Box gộp nhầm | Tỉ lệ box gộp nhầm |")
    L.append("|---|---|---|---|---|---|---|")
    for reg, lab in (("R1", "R1 thô"), ("R3", "R3 cũ"), ("R5_oof", "R5 v2 (OOF)")):
        d = near[reg]
        L.append(f"| {lab} | {d['pairs']} | {d['pairs_merged']} | {_p(d['pair_rate'])} | {d['box_cands']} | "
                 f"{d['box_merged']} | {_p(d['box_rate'])} |")
    L.append("")
    r1r, r5r = near["R1"]["box_rate"] or 0.0, near["R5_oof"]["box_rate"] or 0.0
    L.append(f"Chênh lệch tỉ lệ box gộp nhầm R5 - R1: **{100 * (r5r - r1r):+.1f} điểm %** "
             f"(ràng buộc: không quá +1.0 điểm %, {'ĐẠT' if r5r <= r1r + NEAR_SLACK else 'KHÔNG ĐẠT'}).\n")

    L.append("## 5. Độ nhạy tham số (trong-mẫu, trên toàn bộ 613 trang)\n")
    L.append("Thay từng tham số một, giữ các tham số khác bằng cấu hình triển khai. Đây là trong-mẫu, chỉ để xem hướng tác động.\n")
    L.append("| Tham số | Giá trị | Vỡ whole@0.5 | Nguyên vẹn bị hỏng | Tỉ lệ box gộp nhầm | Khả thi | Triển khai |")
    L.append("|---|---|---|---|---|---|---|")
    for row in res["sensitivity_insample"]:
        L.append(f"| `{row['param']}` | {row['value']} | {row['broken_whole05_insample']} | {row['harm_intact']} | "
                 f"{_p(row['near_rate'])} | {'có' if row['feasible'] else 'không'} | {'*' if row['is_deployed'] else ''} |")
    L.append("")
    L.append("Mười cấu hình tốt nhất trên toàn bộ dữ liệu (trong-mẫu, xếp theo khả thi rồi số ca vỡ được sửa):\n")
    L.append("| Cấu hình | Khả thi | Vỡ whole@0.5 | Nguyên vẹn bị hỏng | Tỉ lệ box gộp nhầm |")
    L.append("|---|---|---|---|---|")
    for row in res["top10_insample"]:
        L.append(f"| `{row['config_id']}` | {'có' if row['feasible'] else 'không'} | {row['broken_whole05']} | "
                 f"{row['harm_intact']} | {_p(row['near_rate'])} |")
    L.append("")

    L.append("## 6. Nguyên nhân các ca vỡ còn sai (out-of-fold)\n")
    causes = res["r5_fail_causes_oof"]
    if causes:
        L.append("| Nguyên nhân | Số ca |")
        L.append("|---|---|")
        for k, v in sorted(causes.items(), key=lambda kv: -kv[1]):
            L.append(f"| `{k}` | {v} |")
    L.append("")
    L.append("- `no_box_covers_gt`: không có box R5 nào phủ >= 0.5 GT (mảnh không được gộp hoặc bị loại vì ràng buộc).")
    L.append("- `box_too_large`: có box phủ GT nhưng nở quá lớn so với GT.")
    L.append("- `box_too_small_or_shifted`: có box phủ GT nhưng không đủ IoU (quá nhỏ hoặc lệch).\n")

    ms = res["missed_summary"]
    L.append("## 7. Phân tích 275 GT bị bỏ sót (không đổi thuật toán)\n")
    L.append(f"Số GT bị bỏ sót: **{ms['n_missed']}** (không có box thô nào chứa >= 0.7 của GT).\n")
    L.append("| Đặc điểm | Số GT | Tỉ lệ |")
    L.append("|---|---|---|")
    for b in ("<1%", "1-5%", "5-20%", ">=20%"):
        c = ms["area_bucket"].get(b, 0)
        L.append(f"| Diện tích GT / trang {b} | {c} | {_p(rate(c, ms['n_missed']))} |")
    for label, key in (("Nằm trong container OpenCV (phủ >= 0.9 diện tích GT)", "in_container_ge_0.9"),
                       ("Có container OpenCV khớp GT (IoU >= 0.5)", "container_iou_ge_0.5"),
                       ("Có box formula/table phủ >= 0.5 GT (Docling đã nhận là vùng khác)", "other_box_covers_ge_0.5")):
        c = ms[key]
        L.append(f"| {label} | {c} | {_p(rate(c, ms['n_missed']))} |")
    L.append("")
    L.append("Phân bố theo data_source: " + ", ".join(f"`{k}` {v}" for k, v in ms["by_data_source"].items()) + ".")
    L.append("")
    L.append("Phân bố theo layout: " + ", ".join(f"`{k}` {v}" for k, v in ms["by_layout"].items()) + ".\n")

    L.append("## 8. Ảnh minh hoạ\n")
    if res["viz"]:
        L.append("Thư mục `viz_v3/`. GT xanh, R5 đỏ (cấu hình out-of-fold). Nguyên nhân và ca được ghi trên tiêu đề.\n")
        for v in res["viz"]:
            L.append(f"- `{v['file']}`: {v['case']} | `{v['stem'][:40]}` | cls={v['cls']}")
    else:
        L.append("Không tạo ảnh (--no-viz).")
    L.append("")

    L.append("## 9. Giới hạn\n")
    L.append("- Đây là ảnh phẳng với OCR tắt. Chưa đo trên PDF vector kiểu AUTOSAR, mục tiêu dữ liệu thực.")
    L.append("- GT `figure` của OmniDocBench gồm ảnh, biểu đồ và sơ đồ. Không phải đo riêng trên sơ đồ khối.")
    L.append("- Chỉ có 38 ca vỡ. Khoảng tin cậy rộng, và chọn cấu hình trên 4 fold có thể không ổn định (xem mục 2.1).")
    L.append("- Ngưỡng cạnh 80 px để định nghĩa cặp gần nhau là định nghĩa đánh giá cũ, giữ để so sánh. Thuật toán không dùng ngưỡng pixel tuyệt đối.")
    L.append("- Số GT bỏ sót (275) không thể sửa bằng gom cụm vì không có mảnh thô. Phát hiện hình từ pixel nằm ngoài phạm vi này.")
    L.append("- Ràng buộc hại và gộp nhầm là ràng buộc cứng do người dùng đặt. Không có cấu hình nào thoả ràng buộc thì báo cáo nêu rõ.\n")

    path.write_text("\n".join(L).strip() + "\n", encoding="utf-8")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Leave-source-out evaluation of diagram grouping v2 (OCR-off cache).")
    ap.add_argument("--limit", type=int, default=0, help="first N pages only (debugging)")
    ap.add_argument("--no-viz", action="store_true", help="skip visualisations")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    run(parse_args(argv))


if __name__ == "__main__":
    main()
