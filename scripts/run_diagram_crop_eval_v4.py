"""
Over-merge vetoes on top of the aggressive v2 merger (OCR OFF cache, no Docling run).

Question: can a veto on a candidate merge keep most of the broken-figure fixes while removing the
harm that diagram_grouping_v2 does on intact figures?  Pages/GT/folds/metrics are those of
run_diagram_crop_eval_v3 (613 pages, 1385 GT figures, 5 leave-source-out folds).

Base merger : group_whole_diagrams_image_v2 with GroupParamsV2() defaults
              (v3: fixes 13/38 broken, harms 197/1057 intact in-sample).
Vetoes (a veto only blocks a pair that v2 would have merged; the deployed config is NOT used as
base because it forms only 7 groups in 613 pages, so a veto there has nothing to block):
  cap  : ORACLE caption veto. Uses OmniDocBench GT 'figure_caption' boxes (upper bound, not a Docling
         result: the OCR-off cache has no caption boxes; the OCR-on cache covers 107/613 pages).
         Each caption is assigned to the nearest picture piece within CAP_REACH x that piece's diagonal.
         Pair vetoed when both pieces own >= 1 caption and the caption sets are disjoint.
  ws   : self-normalising whitespace veto (idea 3). On the axis of the larger edge gap,
         ratio = gap / max(longest ink-free run inside either piece along that axis, WS_FLOOR_REL x
         max(W, H)). Vetoed when ratio > t. t is chosen inside the training folds only.
  both : cap or ws.

PRE-REGISTERED SUCCESS BAR (written before the first full run). A veto counts as effective only if,
out-of-fold over all 613 pages, ALL of:
  (i)   broken GT fixed (whole@0.5 gained vs R1) >= 5 / 38
  (ii)  intact GT harmed <= 1% of 1057  (<= 10)
  (iii) near-pair box merge rate <= R1 rate + 1 pp
If no veto clears this, the result is reported as negative and the work stops there.

Usage:
  .venv\\Scripts\\python scripts\\run_diagram_crop_eval_v4.py
  .venv\\Scripts\\python scripts\\run_diagram_crop_eval_v4.py --limit 40
Outputs: outputs/05_diagram_crop_eval/results_v4.json, report_v4.md, run_v4.log (v3 files untouched).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from diagram_grouping_v2 import (  # noqa: E402
    CONT_MAX_AREA_FRAC,
    PICTURE_LABELS,
    GroupParamsV2,
    _area,
    _diag,
    _edge_gap,
    _pure_container,
    group_whole_diagrams_image_v2,
    pair_eligible,
)
from eval_utils import piece_containment  # noqa: E402
from run_diagram_crop_eval import DATASET_JSON, IMAGES_DIR, poly_to_bbox  # noqa: E402
from run_diagram_crop_eval_v3 import (  # noqa: E402
    HARM_FRAC,
    N_FOLDS,
    NEAR_SLACK,
    OUT_DIR,
    assign_folds,
    eval_boxes,
    harm_stats,
    near_stats,
    prepare_pages,
    rate,
    select_config,
    summarize,
    wilson,
)
from run_e2e_reading_order_eval import Logger  # noqa: E402
from test_diagram_grouping import read_image_bgr  # noqa: E402

CAP_REACH = 1.0          # caption -> piece assignment reach, x piece diagonal (fixed, not tuned)
WS_FLOOR_REL = 0.005     # floor of the inner-whitespace denominator, share of max(W, H)
INK_DELTA = 30           # |gray - page background| above this counts as ink
WS_GRID = (1.0, 2.0, 3.0, 5.0, 8.0, 13.0)
BAR = {"fixed_min": 5, "harm_max_frac": 0.01, "near_slack": 0.01}


# ----------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------
def gt_caption_boxes(item: Dict[str, Any]) -> List[List[float]]:
    out = []
    for d in item.get("layout_dets", []):
        if d.get("ignore", False) or d.get("category_type") != "figure_caption":
            continue
        b = poly_to_bbox(d.get("poly") or d.get("polygon") or d.get("bbox") or [])
        if b:
            out.append([float(v) for v in b])
    return out


def edge_dist(a: Sequence[float], b: Sequence[float]) -> float:
    dx, dy = _edge_gap(a, b)
    return math.hypot(dx, dy)


def caption_owners(pieces: List[List[float]], caps: List[List[float]]) -> List[frozenset]:
    """For each piece the set of caption indices whose nearest piece it is (within CAP_REACH x diag)."""
    owned: List[set] = [set() for _ in pieces]
    for ci, c in enumerate(caps):
        if not pieces:
            break
        ds = [edge_dist(c, p) for p in pieces]
        k = int(np.argmin(ds))
        if ds[k] <= CAP_REACH * _diag(pieces[k]):
            owned[k].add(ci)
    return [frozenset(s) for s in owned]


def load_ink(img_rel: str) -> np.ndarray:
    bgr = read_image_bgr(str(IMAGES_DIR / img_rel))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bg = int(np.bincount(gray.ravel(), minlength=256).argmax())
    return np.abs(gray.astype(np.int16) - bg) > INK_DELTA


def inner_ws_run(ink: np.ndarray, box: Sequence[float], axis: str) -> int:
    """Longest ink-free run of columns (axis 'x') or rows ('y') strictly inside the piece's ink extent."""
    h, w = ink.shape
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    x1, y1 = min(w, int(math.ceil(box[2]))), min(h, int(math.ceil(box[3])))
    if x1 <= x0 or y1 <= y0:
        return 0
    reg = ink[y0:y1, x0:x1]
    prof = reg.any(axis=0) if axis == "x" else reg.any(axis=1)
    idx = np.flatnonzero(prof)
    if idx.size < 2:
        return 0
    inner = prof[idx[0]: idx[-1] + 1]
    best = cur = 0
    for v in inner:
        if v:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return best


def ws_ratio(ink: np.ndarray, a: Sequence[float], b: Sequence[float], W: int, H: int) -> float:
    dx, dy = _edge_gap(a, b)
    if dx <= 0 and dy <= 0:
        return 0.0
    axis, gap = ("x", dx) if dx >= dy else ("y", dy)
    inner = max(inner_ws_run(ink, a, axis), inner_ws_run(ink, b, axis))
    return gap / max(float(inner), WS_FLOOR_REL * max(W, H))


def prepare_extra(pages: List[Dict[str, Any]], log: Logger) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Per page: picture pieces, caption owners, pairwise features. Also a table of pairs v2 would merge."""
    data = json.loads(DATASET_JSON.read_text(encoding="utf-8"))
    defaults = GroupParamsV2()
    pair_rows: List[Dict[str, Any]] = []
    for pos, pg in enumerate(pages, 1):
        item = data[pg["idx"]]
        caps = gt_caption_boxes(item)
        pieces = [[float(v) for v in it["bbox"]] for it in pg["items"]
                  if it.get("label") in PICTURE_LABELS and _area(it["bbox"]) > 0]
        others = [[float(v) for v in it["bbox"]] for it in pg["items"]
                  if it.get("label") not in PICTURE_LABELS and it.get("bbox") and _area(it["bbox"]) > 0]
        conts = [list(c) for c in pg["containers"]
                 if _area(c) <= CONT_MAX_AREA_FRAC * pg["page_area"] and _pure_container(c, others)]
        owners = caption_owners(pieces, caps)
        feats: Dict[Tuple[Tuple[float, ...], Tuple[float, ...]], Dict[str, Any]] = {}
        if len(pieces) >= 2:
            ink = load_ink(pg["img_rel"])
            for i in range(len(pieces)):
                for j in range(i + 1, len(pieces)):
                    a, b = pieces[i], pieces[j]
                    cap_sep = bool(owners[i] and owners[j] and not (owners[i] & owners[j]))
                    wr = ws_ratio(ink, a, b, pg["W"], pg["H"])
                    f = {"cap_sep": cap_sep, "ws": wr}
                    feats[(tuple(a), tuple(b))] = f
                    feats[(tuple(b), tuple(a))] = f
                    if pair_eligible(a, b, others, conts, pg["page_area"], defaults):
                        is_pos = any(
                            g["cls"] == "broken"
                            and piece_containment(a, pg["fig_gt"][k]) >= 0.7
                            and piece_containment(b, pg["fig_gt"][k]) >= 0.7
                            for k, g in enumerate(pg["gts"])
                        )
                        pair_rows.append({"pi": pos - 1, "pos": is_pos, "cap_sep": cap_sep, "ws": wr,
                                          "has_caps": bool(caps)})
        pg["feats"] = feats
        pg["n_caps"] = len(caps)
        if pos % 100 == 0 or pos == len(pages):
            log(f"  features {pos}/{len(pages)}")
    return pages, pair_rows


# ----------------------------------------------------------------------
# Systems
# ----------------------------------------------------------------------
def make_veto(pg: Dict[str, Any], use_cap: bool, t_ws: Optional[float]):
    feats = pg["feats"]

    def veto(a: Sequence[float], b: Sequence[float]) -> bool:
        f = feats.get((tuple(a), tuple(b)))
        if f is None:
            return False
        if use_cap and f["cap_sep"]:
            return True
        return t_ws is not None and f["ws"] > t_ws

    return veto


def run_system(pages, use_cap: bool, t_ws: Optional[float]) -> List[Dict[str, Any]]:
    params = GroupParamsV2()
    out = []
    for pg in pages:
        res = group_whole_diagrams_image_v2(pg["items"], (pg["W"], pg["H"]), containers=pg["containers"],
                                            params=params, veto=make_veto(pg, use_cap, t_ws))
        out.append(eval_boxes(pg, res["boxes"]))
    return out


def n_groups(pages, use_cap: bool, t_ws: Optional[float]) -> int:
    params = GroupParamsV2()
    return sum(len(group_whole_diagrams_image_v2(pg["items"], (pg["W"], pg["H"]), containers=pg["containers"],
                                                 params=params, veto=make_veto(pg, use_cap, t_ws))["diagram_groups"])
               for pg in pages)


def metrics(pages, res_list, r1_list, ids) -> Dict[str, Any]:
    h = harm_stats(pages, res_list, r1_list, ids)
    nc = near_stats(pages, res_list, ids)
    all_ = summarize(pages, res_list, ids, "all")
    return {"fixed_broken": h["broken_whole05"], "harm_intact": h["harm_intact"], "n_intact": h["n_intact"],
            "near_box_rate": nc["box_rate"], "near_pair_rate": nc["pair_rate"],
            "whole@0.5_all": all_["whole@0.5"], "whole_rate@0.5_all": all_["whole_rate@0.5"], "n_gt": all_["n_gt"]}


def oof(pages, res_grid, r1_list, fold_of_page, log, name) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    N = len(pages)
    res_oof: List[Optional[Dict[str, Any]]] = [None] * N
    rows = []
    for f in range(N_FOLDS):
        tr = [i for i in range(N) if fold_of_page[i] != f]
        te = [i for i in range(N) if fold_of_page[i] == f]
        ci, s, status = select_config(pages, res_grid, r1_list, tr)
        for i in te:
            res_oof[i] = res_grid[ci][i]
        rows.append({"fold": f, "choice_index": ci, "status": status, "train_fixed": s["broken_whole05"],
                     "train_harm": s["harm_intact"], "train_near": s["near_rate"]})
        log(f"  [{name}] fold {f}: choice={ci} status={status} train fixed={s['broken_whole05']} harm={s['harm_intact']}")
    return res_oof, rows  # type: ignore[return-value]


def verdict(m: Dict[str, Any], near_r1: float) -> Dict[str, Any]:
    ok_fixed = m["fixed_broken"] >= BAR["fixed_min"]
    ok_harm = m["harm_intact"] <= BAR["harm_max_frac"] * m["n_intact"]
    ok_near = (m["near_box_rate"] or 0.0) <= near_r1 + BAR["near_slack"]
    return {"fixed>=5": ok_fixed, "harm<=1%": ok_harm, "near<=R1+1pp": ok_near,
            "effective": ok_fixed and ok_harm and ok_near}


def run(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = Logger(OUT_DIR / "run_v4.log")
    t0 = time.time()
    log("=" * 80)
    log(f"Veto evaluation v4 start | argv={' '.join(sys.argv[1:])}")
    pages, errors = prepare_pages(log, args.limit)
    pages, pair_rows = prepare_extra(pages, log)
    N = len(pages)
    ids = list(range(N))
    r1_list = [eval_boxes(pg, pg["r1_boxes"]) for pg in pages]
    fold_of_src, fold_stats = assign_folds(pages)
    fold_of_page = [fold_of_src[pg["data_source"]] for pg in pages]
    log(f"Pages={N} errors={len(errors)} pages with GT captions={sum(1 for p in pages if p['n_caps'])}")

    base = metrics(pages, r1_list, r1_list, ids)
    near_r1 = base["near_box_rate"] or 0.0
    systems: Dict[str, Any] = {"R1_raw_boxes": {**base, "groups": 0}}

    res_v2 = run_system(pages, False, None)
    systems["v2_default_no_veto"] = {**metrics(pages, res_v2, r1_list, ids), "groups": n_groups(pages, False, None)}
    res_cap = run_system(pages, True, None)
    systems["v2_default+cap_oracle"] = {**metrics(pages, res_cap, r1_list, ids), "groups": n_groups(pages, True, None)}

    # ws veto: threshold grid (no-veto = last candidate), chosen per fold on training folds only
    ws_grid = [run_system(pages, False, t) for t in WS_GRID] + [res_v2]
    both_grid = [run_system(pages, True, t) for t in WS_GRID] + [res_cap]
    insample = []
    for name, grid in (("ws", ws_grid), ("cap+ws", both_grid)):
        for ci, res in enumerate(grid):
            t = WS_GRID[ci] if ci < len(WS_GRID) else None
            insample.append({"system": name, "t_ws": t, **metrics(pages, res, r1_list, ids)})
    for name, grid in (("v2_default+ws", ws_grid), ("v2_default+cap_oracle+ws", both_grid)):
        res_oof, fold_rows = oof(pages, grid, r1_list, fold_of_page, log, name)
        systems[name + "_OOF"] = {**metrics(pages, res_oof, r1_list, ids), "folds": fold_rows,
                                  "groups": None}
    for name in ("v2_default+cap_oracle",):
        systems[name]["note"] = "no tuned parameter (CAP_REACH fixed), so in-sample == out-of-fold"

    for k, m in systems.items():
        if k == "R1_raw_boxes":
            continue
        m["verdict"] = verdict(m, near_r1)

    # Pair-level view over pairs that v2-default would merge (what the veto actually has to separate)
    pos = [r for r in pair_rows if r["pos"]]
    neg = [r for r in pair_rows if not r["pos"]]
    pair_view = {
        "merged_pairs": len(pair_rows), "pos": len(pos), "neg": len(neg),
        "cap_sep": {"pos_vetoed": sum(r["cap_sep"] for r in pos), "neg_vetoed": sum(r["cap_sep"] for r in neg),
                    "pairs_on_pages_with_caps": sum(r["has_caps"] for r in pair_rows)},
        "ws": [{"t": t, "pos_vetoed": sum(r["ws"] > t for r in pos), "neg_vetoed": sum(r["ws"] > t for r in neg)}
               for t in WS_GRID],
    }

    ceiling = {"n_gt": base["n_gt"], "R1_whole@0.5": base["whole@0.5_all"],
               "oracle_fixable_broken": 31, "broken": 38, "missed": 275,
               "perfect_grouping_gain_pp": round(100 * 31 / base["n_gt"], 2),
               "missed_pp": round(100 * 275 / base["n_gt"], 2)}
    out = {"config": {"pages": N, "errors": len(errors), "bar": BAR, "cap_reach": CAP_REACH,
                      "ws_floor_rel": WS_FLOOR_REL, "ink_delta": INK_DELTA, "ws_grid": list(WS_GRID),
                      "R1_near_box_rate": near_r1},
           "systems": systems, "insample_by_threshold": insample, "pair_view": pair_view, "ceiling": ceiling,
           "elapsed_s": round(time.time() - t0, 1)}
    (OUT_DIR / "results_v4.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out)
    log(f"Done in {time.time() - t0:.0f}s -> results_v4.json, report_v4.md")


def write_report(out: Dict[str, Any]) -> None:
    c = out["config"]
    L = ["# Báo cáo v4: veto chống over-merge trên nền v2 default", "",
         "Tiêu chí hiệu quả đăng ký trước: OOF fixed >= 5/38, harm <= 1% intact, near box_rate <= R1 + 1pp.", "",
         "| Hệ thống | fixed broken | harm intact | near box_rate | whole@0.5 | Đạt? |", "|---|---|---|---|---|---|"]
    for k, m in out["systems"].items():
        v = m.get("verdict")
        near = m["near_box_rate"]
        L.append(f"| {k} | {m['fixed_broken']}/38 | {m['harm_intact']}/{m['n_intact']} | "
                 f"{near if near is None else f'{near:.3f}'} | {m['whole@0.5_all']}/{m['n_gt']} | "
                 f"{'có' if v and v['effective'] else ('-' if v is None else 'không')} |")
    L += ["", f"Mốc R1 near box_rate = {c['R1_near_box_rate']:.3f}", "", "## Cặp mà v2 default gộp (veto phải tách được)", ""]
    pv = out["pair_view"]
    L.append(f"Cặp bị gộp: {pv['merged_pairs']} (đúng: {pv['pos']}, sai: {pv['neg']}).")
    L.append(f"Caption oracle: chặn {pv['cap_sep']['pos_vetoed']} cặp đúng và {pv['cap_sep']['neg_vetoed']} cặp sai.")
    L += ["", "| ngưỡng ws | chặn cặp đúng | chặn cặp sai |", "|---|---|---|"]
    for r in pv["ws"]:
        L.append(f"| {r['t']} | {r['pos_vetoed']} | {r['neg_vetoed']} |")
    ce = out["ceiling"]
    L += ["", "## Trần lý thuyết", "",
          f"Gom hoàn hảo chỉ cứu tối đa {ce['oracle_fixable_broken']}/{ce['n_gt']} hình (+{ce['perfect_grouping_gain_pp']} pp whole@0.5), "
          f"trong khi {ce['missed']} hình ({ce['missed_pp']}%) bị detector bỏ sót hoàn toàn."]
    (OUT_DIR / "report_v4.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    run(ap.parse_args())
