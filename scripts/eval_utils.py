"""
Shared evaluation helpers for DocuMind reading-order benchmarks.

Used by:
  - scripts/run_dataset_stress_test.py  (Tier 1 / Tier 2 Kendall's Tau)
  - scripts/run_e2e_reading_order_eval.py (end-to-end Docling evaluation)
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# GT categories that are never scored for reading order.
EXCLUDED_GT_TYPES = ("abandon", "text_mask", "table_mask", "chart_mask", "equation_semantic", "list_group")


def compute_kendall_tau(ranks_a: Sequence[Optional[int]], ranks_b: Sequence[Optional[int]]) -> Optional[float]:
    """
    Kendall's Tau-a rank correlation in [-1.0, 1.0].

    Pairs where either rank is None are dropped. Ties count as 0 in the numerator
    but still occupy the n*(n-1)/2 denominator (tau-a).

    Returns None when fewer than 2 valid pairs exist: the score is undefined there,
    and must not be treated as a perfect 1.0.
    """
    valid_pairs = [(a, b) for a, b in zip(ranks_a, ranks_b) if a is not None and b is not None]
    n = len(valid_pairs)
    if n < 2:
        return None
    r_a = [p[0] for p in valid_pairs]
    r_b = [p[1] for p in valid_pairs]
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff_a = r_a[i] - r_a[j]
            diff_b = r_b[i] - r_b[j]
            prod = diff_a * diff_b
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    total_pairs = n * (n - 1) / 2.0
    return (concordant - discordant) / total_pairs


def poly_to_bbox(box: Sequence[float]) -> Optional[List[float]]:
    """Converts an OmniDocBench bbox (4 values) or polygon (>=8 values) to [xmin, ymin, xmax, ymax]."""
    if len(box) < 4:
        return None
    if len(box) == 4:
        return [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
    xs = [float(box[k]) for k in range(0, len(box), 2)]
    ys = [float(box[k]) for k in range(1, len(box), 2)]
    return [min(xs), min(ys), max(xs), max(ys)]


def gt_elements_from_page(page_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Ground-truth elements of one OmniDocBench page, filtered exactly like Tier 1:
    skip ignore=True, order=None, excluded categories, and boxes that cannot be parsed.
    Element order follows layout_dets order.
    """
    out: List[Dict[str, Any]] = []
    for d_idx, d in enumerate(page_item.get("layout_dets", [])):
        if d.get("ignore", False):
            continue
        if d.get("order") is None:
            continue
        cat = d.get("category_type", "text")
        if cat in EXCLUDED_GT_TYPES:
            continue
        box = d.get("polygon") or d.get("poly") or d.get("bbox") or []
        bbox = poly_to_bbox(box)
        if not bbox:
            continue
        out.append({
            "id": f"gt_{d_idx}",
            "type": cat,
            "bbox": bbox,
            "order": int(d["order"]),
            "text": (d.get("text") or "")[:40],
        })
    return out


def docling_bbox_to_pixel(
    box_top_left_pt: Sequence[float],
    scale_x: float,
    scale_y: float,
) -> List[float]:
    """
    Docling boxes are in page units (points, derived from image DPI) after
    to_top_left_origin(). Multiply by (image_px / page_pt) to get image pixels.
    """
    l, t, r, b = box_top_left_pt
    return [round(l * scale_x, 2), round(t * scale_y, 2), round(r * scale_x, 2), round(b * scale_y, 2)]


def iou_matrix(boxes_a: Sequence[Sequence[float]], boxes_b: Sequence[Sequence[float]]) -> np.ndarray:
    """Pairwise IoU matrix of shape (len(a), len(b)) for [xmin, ymin, xmax, ymax] boxes."""
    a = np.asarray(boxes_a, dtype=float).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=float).reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def match_one_to_one(
    pred_boxes: Sequence[Sequence[float]],
    gt_boxes: Sequence[Sequence[float]],
    iou_thr: float = 0.3,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    One-to-one matching between predicted and GT boxes (Hungarian, maximising total IoU).
    Pairs with IoU < iou_thr are never matched.

    Returns (gt_to_pred, pred_to_gt) index maps.
    """
    gt_to_pred: Dict[int, int] = {}
    pred_to_gt: Dict[int, int] = {}
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return gt_to_pred, pred_to_gt
    m = iou_matrix(pred_boxes, gt_boxes)
    w = np.where(m >= iou_thr, m, 0.0)
    rows, cols = linear_sum_assignment(w, maximize=True)
    for r, c in zip(rows, cols):
        if w[r, c] > 0:
            pred_to_gt[int(r)] = int(c)
            gt_to_pred[int(c)] = int(r)
    return gt_to_pred, pred_to_gt


# ----------------------------------------------------------------------
# Box-detection metrics (used by the diagram/figure crop evaluation)
# ----------------------------------------------------------------------
def box_area(b: Sequence[float]) -> float:
    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))


def area_coverage(gt_box: Sequence[float], pred_box: Sequence[float]) -> float:
    """Fraction of the GT box's area that lies inside the predicted box."""
    ga = box_area(gt_box)
    if ga <= 0:
        return 0.0
    ix = max(0.0, min(gt_box[2], pred_box[2]) - max(gt_box[0], pred_box[0]))
    iy = max(0.0, min(gt_box[3], pred_box[3]) - max(gt_box[1], pred_box[1]))
    return (ix * iy) / ga


def detection_prf(
    pred_boxes: Sequence[Sequence[float]],
    gt_boxes: Sequence[Sequence[float]],
    iou_thr: float,
) -> Dict[str, Any]:
    """
    One-to-one precision / recall / F1 at an IoU threshold (Hungarian matching).
    Conventions: precision = 0 when there are no predictions; recall = 0 when there is no GT;
    F1 = 0 when precision + recall = 0. Returns mean IoU of matched pairs (None if no match).
    """
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)
    g2p, _ = match_one_to_one(pred_boxes, gt_boxes, iou_thr)
    tp = len(g2p)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = None
    if tp:
        m = iou_matrix([pred_boxes[g2p[g]] for g in g2p], [gt_boxes[g] for g in g2p])
        mean_iou = float(np.mean([m[k, k] for k in range(tp)]))
    return {"tp": tp, "n_pred": n_pred, "n_gt": n_gt, "precision": precision,
            "recall": recall, "f1": f1, "mean_iou_matched": mean_iou}


def over_merge_count(
    pred_boxes: Sequence[Sequence[float]],
    fig_gt_boxes: Sequence[Sequence[float]],
    other_gt_boxes: Sequence[Sequence[float]],
    cover_thr: float = 0.5,
    min_figs: int = 2,
    min_others: int = 2,
) -> Dict[str, int]:
    """
    A predicted box is 'over-merged' if it covers >= min_figs distinct GT figures, or
    >= min_others 'other' GT blocks (text/title/caption/formula). Covering means that at least
    cover_thr of the GT box's area lies inside the predicted box.
    """
    boxes_over = 0
    text_absorbed_total = 0
    for p in pred_boxes:
        figs = sum(1 for g in fig_gt_boxes if area_coverage(g, p) >= cover_thr)
        others = sum(1 for g in other_gt_boxes if area_coverage(g, p) >= cover_thr)
        text_absorbed_total += others
        if figs >= min_figs or others >= min_others:
            boxes_over += 1
    return {"over_merge_boxes": boxes_over, "text_blocks_absorbed": text_absorbed_total}


def fragmented_gt_count(
    pred_boxes: Sequence[Sequence[float]],
    fig_gt_boxes: Sequence[Sequence[float]],
    lo: float = 0.1,
    hi: float = 0.5,
) -> int:
    """Number of GT figures overlapped (IoU in [lo, hi)) by >= 2 predicted boxes: each piece is partial."""
    if not pred_boxes or not fig_gt_boxes:
        return 0
    m = iou_matrix(pred_boxes, fig_gt_boxes)  # (P, G)
    partial = (m >= lo) & (m < hi)
    return int(np.sum(partial.sum(axis=0) >= 2))


# ----------------------------------------------------------------------
# Whole-capture / split / merge definitions for GT figure clusters (v2)
# ----------------------------------------------------------------------
def piece_containment(piece: Sequence[float], container: Sequence[float]) -> float:
    """area(piece inter container) / area(piece): how much of the piece lies inside the container."""
    pa = box_area(piece)
    if pa <= 0:
        return 0.0
    ix = max(0.0, min(piece[2], container[2]) - max(piece[0], container[0]))
    iy = max(0.0, min(piece[3], container[3]) - max(piece[1], container[1]))
    return (ix * iy) / pa


def classify_gt_completeness(
    gt_box: Sequence[float],
    raw_pred_boxes: Sequence[Sequence[float]],
    piece_thr: float = 0.7,
    iou_thr: float = 0.5,
) -> Tuple[str, int]:
    """
    Classify one GT figure from the raw (un-grouped) detector boxes:
      'intact'  : some raw box has IoU >= iou_thr with the GT (checked first)
      'broken'  : no such box, and >= 2 raw boxes have containment >= piece_thr inside the GT
      'missed'  : no such box, and no raw box has containment >= piece_thr inside the GT
      'other'   : no such box, exactly one raw box is contained (a single partial piece)
    Returns (class, n_pieces) where n_pieces counts raw boxes with containment >= piece_thr.
    """
    intact = any(iou_matrix([p], [gt_box])[0, 0] >= iou_thr for p in raw_pred_boxes)
    n_pieces = sum(1 for p in raw_pred_boxes if piece_containment(p, gt_box) >= piece_thr)
    if intact:
        return "intact", n_pieces
    if n_pieces >= 2:
        return "broken", n_pieces
    if n_pieces == 0:
        return "missed", 0
    return "other", n_pieces


def whole_capture(gt_box: Sequence[float], pred_boxes: Sequence[Sequence[float]], iou_thr: float = 0.5) -> bool:
    """True when exactly one predicted box reaches IoU >= iou_thr with the GT figure."""
    if not pred_boxes:
        return False
    m = iou_matrix(pred_boxes, [gt_box])[:, 0]
    return int(np.sum(m >= iou_thr)) == 1


def split_count(gt_box: Sequence[float], pred_boxes: Sequence[Sequence[float]], cont_thr: float = 0.5) -> int:
    """Number of predicted boxes that have >= cont_thr of their own area inside the GT figure."""
    return sum(1 for p in pred_boxes if piece_containment(p, gt_box) >= cont_thr)


def boxes_covering_gts(
    pred_boxes: Sequence[Sequence[float]],
    gt_boxes: Sequence[Sequence[float]],
    cover_thr: float = 0.5,
) -> List[List[int]]:
    """For each predicted box, the indices of GT figures with >= cover_thr of the GT area inside the box."""
    out: List[List[int]] = []
    for p in pred_boxes:
        out.append([g for g, gb in enumerate(gt_boxes) if area_coverage(gb, p) >= cover_thr])
    return out
