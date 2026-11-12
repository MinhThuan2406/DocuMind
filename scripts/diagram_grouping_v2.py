"""
Picture-cluster grouping v2 (DO-NO-HARM). Separate from the production function
test_diagram_grouping.group_whole_diagrams_image, which is left unchanged.

Input : cached Docling items (pixel boxes, OCR off): label + bbox in image pixels.
Output: diagram groups (union boxes of >= 2 'picture' pieces) and standalone picture boxes.
        Raw picture boxes are kept as they are unless there is positive evidence to merge.

Changes from the v1 function (group_whole_diagrams_image):
  1. Only 'picture'/'figure' pieces are merged. formula / table boxes are never absorbed; they
     only act as separators (a pair is blocked when such a box sits in the span between them).
  2. All distances are relative to the size of the two pieces (edge gap / min diagonal). No
     absolute pixel thresholds remain (the v1 pad 45 px, gap 160 px, align 35 px, and 140 px
     are gone).
  3. Merging needs evidence: (a) tight relative gap AND alignment along one axis AND comparable
     piece sizes, OR (b) both pieces inside one tight OpenCV container that holds no other
     non-picture box. Chains are accepted only while the union stays under max_group_frac of the page.
  4. Two pieces that are each large (area >= big_frac of the page) are merged only when they
     almost touch (anti-engulf rule for separate figures).
  5. Container snap is applied only when the container is tight (IoU >= snap_iou with the union)
     and contains no non-picture box. v1 snapped whenever IoU > 0.6 or when the container was
     below 1.35x the union, which expanded boxes (container_snap harm).
  6. Container area guard (35 % of page) and the 1.35x tightness multiplier are kept from v1.
  7. Removed from v1: Strategy C (horizontal card alignment with absolute pixel thresholds) and
     the header/list separators, because the OCR-off input carries no text, section_header or
     list_item boxes.

Usage:
  from diagram_grouping_v2 import GroupParamsV2, group_whole_diagrams_image_v2
  res = group_whole_diagrams_image_v2(items, (W, H), containers=..., params=GroupParamsV2())
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

PICTURE_LABELS = ("picture", "figure")
CONT_MULT = 1.35            # container area <= CONT_MULT x union area (kept from v1)
CONT_MAX_AREA_FRAC = 0.35   # containers above this share of the page are background (kept from v1)
BIG_TOUCH_REL = 0.05        # two big pieces must be within this relative gap to merge
CONT_GAP_MULT = 2.0         # container evidence allows a gap up to this multiple of k_gap
SEP_OVERLAP = 0.5           # a non-picture box overlapping the span of a pair by this share blocks it
INSIDE_TOL = 0.15           # 'inside' means at least 1 - INSIDE_TOL of the box lies in the outer box


@dataclass(frozen=True)
class GroupParamsV2:
    k_gap: float = 0.5           # max edge gap / min(diagonal) of the two pieces (tight evidence)
    tau_align: float = 0.5       # min overlap share along x or y (0 disables the alignment need)
    tau_size: float = 0.2        # min area ratio small / large (0 disables)
    evidence: str = "both"       # 'tight' (gap + alignment), 'container' (shared tight picture-only
                                 # container), or 'both' (either one)
    big_frac: float = 0.05       # piece area share of the page regarded as 'big' (>= 1.0 disables)
    max_group_frac: float = 0.35 # reject a group whose union covers more than this share of the page
    snap_iou: float = 0.85       # snap union to a container only when IoU >= this
    pad_rel: float = 0.005       # padding of merged boxes, as a share of max(W, H)


def _area(b: Sequence[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _diag(b: Sequence[float]) -> float:
    return math.hypot(b[2] - b[0], b[3] - b[1])


def _inter(a: Sequence[float], b: Sequence[float]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def _inside(inner: Sequence[float], outer: Sequence[float], tol: float = INSIDE_TOL) -> bool:
    ai = _area(inner)
    if ai <= 0:
        return False
    return _inter(inner, outer) / ai >= 1.0 - tol


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter = _inter(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _union(boxes: Sequence[Sequence[float]]) -> List[float]:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _pure_container(c: Sequence[float], others: Sequence[Sequence[float]]) -> bool:
    """True when the container holds no non-picture box."""
    return not any(_inside(o, c) for o in others)


def _container_covering(a, b, containers) -> Optional[List[float]]:
    """containers are pre-filtered to picture-only, background-guarded candidates for this page."""
    u_area = _area(_union([a, b]))
    for c in containers:
        if _area(c) > CONT_MULT * u_area:
            continue
        if _inside(a, c) and _inside(b, c):
            return c
    return None


def _separated_by_other(a, b, others) -> bool:
    """A non-picture box that lies in the span between a and b (not inside either piece) blocks the pair."""
    u = _union([a, b])
    for o in others:
        if _inside(o, a) or _inside(o, b):
            continue
        oa = _area(o)
        if oa > 0 and _inter(o, u) / oa >= SEP_OVERLAP:
            return True
    return False


def _edge_gap(a, b) -> Tuple[float, float]:
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return dx, dy


def _overlap_share(a, b) -> float:
    ox = min(a[2], b[2]) - max(a[0], b[0])
    oy = min(a[3], b[3]) - max(a[1], b[1])
    wa, ha = a[2] - a[0], a[3] - a[1]
    wb, hb = b[2] - b[0], b[3] - b[1]
    share_x = max(0.0, ox) / max(1e-9, min(wa, wb))
    share_y = max(0.0, oy) / max(1e-9, min(ha, hb))
    return max(share_x, share_y)


def pair_eligible(a, b, others, containers, page_area: float, p: GroupParamsV2) -> bool:
    """containers: already filtered to picture-only, background-guarded candidates for this page."""
    area_a, area_b = _area(a), _area(b)
    if area_a <= 0 or area_b <= 0:
        return False
    s = min(_diag(a), _diag(b))
    dx, dy = _edge_gap(a, b)
    g_rel = math.hypot(dx, dy) / s if s > 0 else float("inf")

    if p.big_frac < 1.0 and min(area_a, area_b) >= p.big_frac * page_area and g_rel > BIG_TOUCH_REL:
        return False  # anti-engulf: two large pieces that are not touching stay separate figures
    if p.tau_size > 0 and min(area_a, area_b) / max(area_a, area_b) < p.tau_size:
        return False
    if _separated_by_other(a, b, others):
        return False

    if p.evidence not in ("tight", "container", "both"):
        raise ValueError(f"unknown evidence mode: {p.evidence}")
    if p.evidence in ("tight", "both") and g_rel <= p.k_gap and _overlap_share(a, b) >= p.tau_align:
        return True
    if p.evidence in ("container", "both") and g_rel <= CONT_GAP_MULT * p.k_gap:
        return _container_covering(a, b, containers) is not None
    return False


def group_whole_diagrams_image_v2(
    items: Sequence[Dict[str, Any]],
    image_size: Tuple[int, int],
    containers: Optional[Sequence[Sequence[float]]] = None,
    image_path: Optional[str] = None,
    params: GroupParamsV2 = GroupParamsV2(),
    veto: Optional[Callable[[Sequence[float], Sequence[float]], bool]] = None,
) -> Dict[str, Any]:
    """
    items: dicts with 'label' and pixel 'bbox' [x0, y0, x1, y1] (cached Docling items).
    containers: OpenCV container boxes for the same image. If None and image_path is given they are
    computed with test_diagram_grouping.detect_visual_containers_opencv (lazy import).
    Returns: diagram_groups (bbox, member piece indices), standalone (picture boxes), boxes (both).
    """
    W, H = image_size
    page_area = float(W) * float(H)
    if containers is None:
        containers = []
        if image_path is not None:
            from test_diagram_grouping import detect_visual_containers_opencv
            containers = detect_visual_containers_opencv(image_path)

    pieces: List[List[float]] = []
    others: List[List[float]] = []
    for it in items:
        b = it.get("bbox")
        if not b or len(b) != 4 or _area(b) <= 0:
            continue
        if it.get("label") in PICTURE_LABELS:
            pieces.append([float(v) for v in b])
        else:
            others.append([float(v) for v in b])

    # Picture-only containers under the background guard; the same list serves evidence and snap.
    containers = [list(c) for c in containers
                  if _area(c) <= CONT_MAX_AREA_FRAC * page_area and _pure_container(c, others)]

    n = len(pieces)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if pair_eligible(pieces[i], pieces[j], others, containers, page_area, params) and (
                    veto is None or not veto(pieces[i], pieces[j])):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    comps: Dict[int, List[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)

    groups: List[Dict[str, Any]] = []
    used = set()
    pad = params.pad_rel * max(W, H)
    for _, idxs in sorted(comps.items(), key=lambda kv: kv[1][0]):
        if len(idxs) < 2:
            continue
        u = _union([pieces[i] for i in idxs])
        if _area(u) > params.max_group_frac * page_area:
            continue
        box = u
        for c in containers:
            if _iou(c, u) >= params.snap_iou:
                box = list(c)
                break
        final = [
            max(0.0, round(box[0] - pad, 1)),
            max(0.0, round(box[1] - pad, 1)),
            min(float(W), round(box[2] + pad, 1)),
            min(float(H), round(box[3] + pad, 1)),
        ]
        groups.append({"bbox": final, "merged_element_indices": list(idxs), "n_pieces": len(idxs)})
        used.update(idxs)

    standalone = [pieces[i] for i in range(n) if i not in used]
    return {
        "diagram_groups": groups,
        "standalone": standalone,
        "boxes": [g["bbox"] for g in groups] + standalone,
        "pieces": pieces,
        "others": others,
    }
