"""
Analysis only (no change to the grouping algorithm): can features of two picture pieces separate
"pieces of one broken figure" from "pieces of two different figures" (or of an intact figure)?

Data : OCR-off cache, 613 pages, same page preparation and same leave-source-out folds as
       run_diagram_crop_eval_v3 (fold assignment depends only on GT counts).
Pairs: candidate pairs are all picture-piece pairs whose edge gap <= 1.0 x min(diagonal). This is the
       loosest 'tight' setting of the v2 grid (k_gap = 1.0), so no positive pair is removed by the
       candidate filter. Positive pair = both pieces have containment >= 0.7 in the same broken GT.
       Everything else is negative (different GT, or one piece belongs to an intact/missed figure).
Output:
  - oracle ceiling: broken GT whose pieces (containment >= 0.7), unioned, reach IoU >= 0.5
  - univariate AUC per feature (rank-based), CI by bootstrap over PAGES
  - depth-2 decision tree and standardised logistic regression (numpy IRLS), out-of-fold by source fold
  - trade-off in GT units: for each threshold, merge pairs whose score >= threshold (union-find per
    page, union box per component), then count broken GT fixed (whole@0.5) and intact GT harmed.
    Reported two ways: (a) the OOF score curve (threshold picked in-sample, optimistic) and
    (b) nested: threshold chosen on the other folds' pages, applied to the held-out fold.

Usage:
  .venv\\Scripts\\python scripts\\analyze_pair_features.py
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import rankdata

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from eval_utils import piece_containment, whole_capture  # noqa: E402
from diagram_grouping_v2 import (  # noqa: E402
    CONT_MAX_AREA_FRAC,
    CONT_MULT,
    _area,
    _inside,
    _inter,
    _pure_container,
    _separated_by_other,
    _union,
)
from run_diagram_crop_eval_v3 import OUT_DIR, assign_folds, prepare_pages  # noqa: E402

CAND_GAP = 1.0
PIECE_THR = 0.7
N_BOOT = 400
SEED = 42
HARM_FRAC = 0.01
FEATURES = [
    "gap_rel", "gap_abs_px", "gap_over_img_diag", "align_share", "align_x", "align_y",
    "size_ratio", "aspect_similarity", "min_area_rel", "max_area_rel", "union_area_rel",
    "same_tight_container", "separator_between", "others_overlap_union", "horizontal_arrangement",
    "n_pieces_on_page",
]
LOG_FEATURES = {"gap_abs_px", "min_area_rel", "max_area_rel", "union_area_rel", "n_pieces_on_page"}


def pair_feats(a, b, others, containers, W, H, page_area, n_pieces) -> Dict[str, float]:
    area_a, area_b = _area(a), _area(b)
    wa, ha = a[2] - a[0], a[3] - a[1]
    wb, hb = b[2] - b[0], b[3] - b[1]
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    gap = math.hypot(dx, dy)
    s = min(math.hypot(wa, ha), math.hypot(wb, hb))
    u = _union([a, b])
    u_area = _area(u)
    ov_x = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) / max(1e-9, min(wa, wb))
    ov_y = max(0.0, min(a[3], b[3]) - max(a[1], b[1])) / max(1e-9, min(ha, hb))
    same_cont = 0.0
    for c in containers:
        if _area(c) <= CONT_MAX_AREA_FRAC * page_area and _area(c) <= CONT_MULT * u_area \
                and _inside(a, c) and _inside(b, c) and _pure_container(c, others):
            same_cont = 1.0
            break
    n_other = 0
    for o in others:
        if _inside(o, a) or _inside(o, b):
            continue
        oa = _area(o)
        if oa > 0 and _inter(o, u) / oa >= 0.5:
            n_other += 1
    ar_a = wa / max(1e-9, ha)
    ar_b = wb / max(1e-9, hb)
    return {
        "gap_rel": gap / max(1e-9, s),
        "gap_abs_px": gap,
        "gap_over_img_diag": gap / max(1e-9, math.hypot(W, H)),
        "align_share": max(ov_x, ov_y),
        "align_x": ov_x,
        "align_y": ov_y,
        "size_ratio": min(area_a, area_b) / max(1e-9, max(area_a, area_b)),
        "aspect_similarity": min(ar_a, ar_b) / max(1e-9, max(ar_a, ar_b)),
        "min_area_rel": min(area_a, area_b) / page_area,
        "max_area_rel": max(area_a, area_b) / page_area,
        "union_area_rel": u_area / page_area,
        "same_tight_container": same_cont,
        "separator_between": 1.0 if _separated_by_other(a, b, others) else 0.0,
        "others_overlap_union": float(n_other),
        "horizontal_arrangement": 1.0 if dx >= dy else 0.0,
        "n_pieces_on_page": float(n_pieces),
    }


def auc_rank(scores: np.ndarray, y: np.ndarray) -> Optional[float]:
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(scores, method="average")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def gini(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    p = y.mean()
    return 2 * p * (1 - p)


def best_split(X: np.ndarray, y: np.ndarray, idx: np.ndarray) -> Optional[Tuple[float, int, float, np.ndarray, np.ndarray]]:
    best = None
    for f in range(X.shape[1]):
        vals = X[idx, f]
        uniq = np.unique(vals)
        if len(uniq) < 2:
            continue
        cands = (uniq[:-1] + uniq[1:]) / 2.0
        if len(cands) > 60:
            cands = np.quantile(vals, np.linspace(0.02, 0.98, 60))
        for t in cands:
            m = vals <= t
            left, right = idx[m], idx[~m]
            if len(left) < 5 or len(right) < 5:
                continue
            g = (len(left) * gini(y[left]) + len(right) * gini(y[right])) / len(idx)
            if best is None or g < best[0]:
                best = (g, f, float(t), left, right)
    return best


def fit_tree(X, y, idx, depth) -> Dict[str, Any]:
    leaf = {"leaf": True, "p": float(y[idx].mean()) if len(idx) else 0.5, "n": int(len(idx))}
    if depth == 0 or len(idx) < 20 or gini(y[idx]) == 0:
        return leaf
    bs = best_split(X, y, idx)
    if bs is None:
        return leaf
    _, f, t, left, right = bs
    return {"leaf": False, "feature": f, "thr": t,
            "left": fit_tree(X, y, left, depth - 1), "right": fit_tree(X, y, right, depth - 1)}


def predict_tree(node, x) -> float:
    while not node["leaf"]:
        node = node["left"] if x[node["feature"]] <= node["thr"] else node["right"]
    return node["p"]


def fit_logit(X: np.ndarray, y: np.ndarray, lam: float = 1e-2, iters: int = 60) -> np.ndarray:
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    reg = np.eye(Xb.shape[1]) * lam
    reg[0, 0] = 0.0
    for _ in range(iters):
        z = np.clip(Xb @ w, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-z))
        g = Xb.T @ (y - mu) - reg @ w
        H = Xb.T @ (Xb * (mu * (1 - mu))[:, None]) + reg + 1e-9 * np.eye(Xb.shape[1])
        step = np.linalg.solve(H, g)
        w += step
        if np.linalg.norm(step) < 1e-8:
            break
    return w


def predict_logit(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    z = np.clip(np.hstack([np.ones((len(X), 1)), X]) @ w, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def components(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    comp: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        comp[find(i)].append(i)
    return list(comp.values())


def tradeoff_page(pg, pieces, pair_list, scores, thr, r1_whole05) -> Tuple[List[int], List[int]]:
    """Per page: fixed broken GT and harmed intact GT when pairs with score >= thr are merged."""
    edges = []
    for (pi_idx, i, j), s in zip(pair_list, scores):
        if s >= thr:
            edges.append((i, j))
    if not edges:
        boxes = pieces
    else:
        comps = components(len(pieces), edges)
        boxes = []
        for c in comps:
            if len(c) >= 2:
                boxes.append(_union([pieces[k] for k in c]))
            else:
                boxes.append(pieces[c[0]])
    fixed, harmed = [], []
    for k, g in enumerate(pg["gts"]):
        gb = pg["fig_gt"][k]
        new = whole_capture(gb, boxes, 0.5)
        if g["cls"] == "broken" and new:
            fixed.append(k)
        if g["cls"] == "intact" and r1_whole05[k] and not new:
            harmed.append(k)
    return fixed, harmed


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    pages, errors = prepare_pages(lambda m: None, 0)
    N = len(pages)
    fold_of_src, fold_stats = assign_folds(pages)
    fold_of_page = [fold_of_src[pg["data_source"]] for pg in pages]
    print(f"pages={N} errors={len(errors)} folds={[s['sources'] for s in fold_stats]}", flush=True)

    rows: List[Dict[str, Any]] = []
    pair_index: List[Tuple[int, int, int]] = []  # (page, i, j)
    y_all: List[int] = []
    feat_all: List[List[float]] = []
    pieces_by_page: List[List[List[float]]] = []
    r1_whole05_by_page: List[List[bool]] = []
    oracle_fixable = 0
    broken_total = 0
    pos_outside_cand = 0
    pos_pairs_total = 0
    broken_with_pos = set()

    for pi, pg in enumerate(pages):
        pieces = [list(map(float, b)) for b in pg["r1_boxes"]]
        pieces_by_page.append(pieces)
        r1_whole05_by_page.append([bool(whole_capture(g, pieces, 0.5)) for g in pg["fig_gt"]])
        n_p = len(pieces)
        for k, g in enumerate(pg["gts"]):
            if g["cls"] != "broken":
                continue
            broken_total += 1
            gb = pg["fig_gt"][k]
            sel = [q for q in pieces if piece_containment(q, gb) >= PIECE_THR]
            if len(sel) >= 2:
                u = _union(sel)
                inter = _inter(u, gb)
                iou = inter / max(1e-9, _area(u) + _area(gb) - inter)
                if iou >= 0.5:
                    oracle_fixable += 1
        for i in range(n_p):
            for j in range(i + 1, n_p):
                a, b = pieces[i], pieces[j]
                wa, ha = a[2] - a[0], a[3] - a[1]
                wb, hb = b[2] - b[0], b[3] - b[1]
                dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
                dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
                s = min(math.hypot(wa, ha), math.hypot(wb, hb))
                g_rel = math.hypot(dx, dy) / max(1e-9, s)
                pos_gt = None
                for k, g in enumerate(pg["gts"]):
                    if g["cls"] != "broken":
                        continue
                    gb = pg["fig_gt"][k]
                    if piece_containment(a, gb) >= PIECE_THR and piece_containment(b, gb) >= PIECE_THR:
                        pos_gt = k
                        break
                is_pos = pos_gt is not None
                if is_pos:
                    pos_pairs_total += 1
                    broken_with_pos.add((pi, pos_gt))
                    if g_rel > CAND_GAP:
                        pos_outside_cand += 1
                if g_rel > CAND_GAP:
                    continue
                f = pair_feats(a, b, pg["others"], pg["containers"], pg["W"], pg["H"], pg["page_area"], n_p)
                pair_index.append((pi, i, j))
                y_all.append(1 if is_pos else 0)
                feat_all.append([f[k] for k in FEATURES])
        if (pi + 1) % 200 == 0:
            print(f"  features {pi + 1}/{N}", flush=True)

    X = np.array(feat_all, dtype=float)
    y = np.array(y_all, dtype=int)
    pg_of_pair = np.array([p for p, _, _ in pair_index], dtype=int)
    fold_of_pair = np.array([fold_of_page[p] for p in pg_of_pair], dtype=int)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    print(f"candidate pairs={len(y)} positives={n_pos} negatives={n_neg} "
          f"broken_GT={broken_total} oracle_fixable={oracle_fixable} "
          f"positives_outside_candidates={pos_outside_cand} broken_GT_with_positive_pair={len(broken_with_pos)}", flush=True)

    Xt = X.copy()
    for k, name in enumerate(FEATURES):
        if name in LOG_FEATURES:
            Xt[:, k] = np.log1p(np.maximum(Xt[:, k], 0.0))

    # Univariate AUC with page-level bootstrap
    rng = np.random.default_rng(SEED)
    page_ids = np.unique(pg_of_pair)
    rows_by_page = {p: np.where(pg_of_pair == p)[0] for p in page_ids}
    auc_rows = []
    for k, name in enumerate(FEATURES):
        a = auc_rank(Xt[:, k], y)
        if a is None:
            auc_rows.append({"feature": name, "auc": None})
            continue
        boots = []
        for _ in range(N_BOOT):
            samp = rng.choice(page_ids, size=len(page_ids), replace=True)
            idx = np.concatenate([rows_by_page[p] for p in samp])
            ab = auc_rank(Xt[idx, k], y[idx])
            if ab is not None:
                boots.append(ab)
        lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (None, None))
        auc_rows.append({"feature": name, "auc": round(a, 4),
                         "auc_dir": round(max(a, 1 - a), 4),
                         "direction": "higher=positive" if a >= 0.5 else "lower=positive",
                         "ci95_boot_pages": [round(float(lo), 4), round(float(hi), 4)] if boots else None,
                         "median_pos": round(float(np.median(Xt[y == 1, k])), 4),
                         "median_neg": round(float(np.median(Xt[y == 0, k])), 4)})
    auc_rows.sort(key=lambda r: -(r.get("auc_dir") or 0))

    # Out-of-fold models
    oof_tree = np.zeros(len(y))
    oof_logit = np.zeros(len(y))
    tree_rules = None
    for f in range(5):
        tr = np.where(fold_of_pair != f)[0]
        te = np.where(fold_of_pair == f)[0]
        if len(tr) == 0 or len(te) == 0:
            continue
        tree = fit_tree(Xt, y, tr, depth=2)
        oof_tree[te] = [predict_tree(tree, Xt[i]) for i in te]
        mu, sd = Xt[tr].mean(axis=0), Xt[tr].std(axis=0) + 1e-9
        w = fit_logit((Xt[tr] - mu) / sd, y[tr])
        oof_logit[te] = predict_logit(w, (Xt[te] - mu) / sd)
    full_tree = fit_tree(Xt, y, np.arange(len(y)), depth=2)
    mu_all, sd_all = Xt.mean(axis=0), Xt.std(axis=0) + 1e-9
    w_all = fit_logit((Xt - mu_all) / sd_all, y)
    logit_coef = {name: round(float(w_all[k + 1]), 4) for k, name in enumerate(FEATURES)}

    def tree_text(node, depth=0) -> List[str]:
        if node["leaf"]:
            return [("  " * depth) + f"-> p(pos)={node['p']:.3f} (n={node['n']})"]
        name = FEATURES[node["feature"]]
        return ([("  " * depth) + f"if {name} <= {node['thr']:.4f}:"] + tree_text(node["left"], depth + 1)
                + [("  " * depth) + "else:"] + tree_text(node["right"], depth + 1))

    tree_rules = tree_text(full_tree)

    def pair_auc(scores):
        a = auc_rank(scores, y)
        return None if a is None else round(a, 4)

    model_auc = {"depth2_tree_oof": pair_auc(oof_tree), "logistic_oof": pair_auc(oof_logit)}

    # Trade-off in GT units
    score_sets = {"depth2_tree_oof": oof_tree, "logistic_oof": oof_logit, "best_single_feature": None}
    best_feat = auc_rows[0]["feature"]
    bk = FEATURES.index(best_feat)
    sign = 1.0 if auc_rows[0]["auc"] >= 0.5 else -1.0
    score_sets["best_single_feature"] = sign * Xt[:, bk]

    intact_total = sum(1 for pg in pages for g in pg["gts"] if g["cls"] == "intact")
    curves = {}
    for name, sc in score_sets.items():
        thr_list = np.unique(np.quantile(sc, np.linspace(0.0, 0.999, 60)))
        curve = []
        page_results: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
        for thr in thr_list:
            fx = np.zeros(N, dtype=int)
            hm = np.zeros(N, dtype=int)
            for pi, pg in enumerate(pages):
                # pair rows of one page are contiguous (pair_index is built page by page)
                start = np.searchsorted(pg_of_pair, pi, side="left")
                end = np.searchsorted(pg_of_pair, pi, side="right")
                pl = [pair_index[r] for r in range(start, end)]
                fixed, harmed = tradeoff_page(pg, pieces_by_page[pi], pl, sc[start:end], thr,
                                              r1_whole05_by_page[pi])
                fx[pi] = len(fixed)
                hm[pi] = len(harmed)
            page_results[float(thr)] = (fx, hm)
            curve.append({"thr": float(thr), "fixed": int(fx.sum()), "harmed": int(hm.sum())})
        curves[name] = curve
        # nested threshold selection: choose on other folds, apply to held-out fold
        nested_fixed = nested_harm = 0
        chosen = []
        page_fold = np.array(fold_of_page)
        intact_pg = np.array([sum(1 for g in pg["gts"] if g["cls"] == "intact") for pg in pages])
        for f in range(5):
            mask_tr = page_fold != f
            best = None
            for thr, (fx, hm) in page_results.items():
                fx_tr = int(fx[mask_tr].sum())
                hm_tr = int(hm[mask_tr].sum())
                n_int_tr = int(intact_pg[mask_tr].sum())
                if hm_tr <= HARM_FRAC * n_int_tr:
                    key = (fx_tr, -hm_tr)
                    if best is None or key > best[0]:
                        best = (key, thr)
            if best is None:
                chosen.append({"fold": f, "thr": None, "feasible": False})
                continue
            thr = best[1]
            fx, hm = page_results[thr]
            nested_fixed += int(fx[~mask_tr].sum())
            nested_harm += int(hm[~mask_tr].sum())
            chosen.append({"fold": f, "thr": round(thr, 4), "feasible": True,
                           "fixed_heldout": int(fx[~mask_tr].sum()), "harmed_heldout": int(hm[~mask_tr].sum())})
        curves[name + "_nested"] = {"fixed": nested_fixed, "harmed": nested_harm, "folds": chosen,
                                    "n_intact": int(intact_total)}

    out = {
        "config": {"pages": N, "page_errors": len(errors), "candidate_gap_rel": CAND_GAP,
                   "piece_containment_thr": PIECE_THR, "features": FEATURES,
                   "n_candidate_pairs": int(len(y)), "n_positive_pairs": n_pos, "n_negative_pairs": n_neg,
                   "n_broken_gt": broken_total, "n_broken_gt_with_positive_pair": len(broken_with_pos),
                   "positive_pairs_outside_candidates": pos_outside_cand,
                   "oracle_fixable_broken_gt": oracle_fixable, "n_intact_gt": intact_total,
                   "boot_resamples": N_BOOT, "harm_frac": HARM_FRAC},
        "auc_univariate": auc_rows,
        "model_auc_oof": model_auc,
        "tree_depth2_full_fit": tree_rules,
        "logistic_standardised_coef": logit_coef,
        "tradeoff_curves": curves,
    }
    (OUT_DIR / "pair_features.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("AUC (univariate, sorted):", flush=True)
    for r in auc_rows:
        print(f"  {r['feature']:24s} auc={r.get('auc')} dir={r.get('direction')} ci={r.get('ci95_boot_pages')}", flush=True)
    print("model AUC OOF:", model_auc, flush=True)
    for name in ("depth2_tree_oof", "logistic_oof", "best_single_feature"):
        nes = curves[name + "_nested"]
        print(f"{name}: nested fixed={nes['fixed']}/{broken_total} harmed={nes['harmed']}/{intact_total}", flush=True)
    print("saved", OUT_DIR / "pair_features.json", flush=True)


if __name__ == "__main__":
    main()
