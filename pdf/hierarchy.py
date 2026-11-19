from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ._pdf_io import extract_toc

_IGNORED_ROLES = frozenset({"page_header", "page_footer"})
_NON_HEADING_LABELS = frozenset(
    {
        "page_header",
        "page_footer",
        "caption",
        "footnote",
        "list_item",
        "code",
        "formula",
        "table",
        "picture",
    }
)
_HEADING_LABELS = frozenset({"title", "section_header"})

_CAPTION_PREFIX_RE = re.compile(
    r"^(?P<kind>figure|fig\.?|diagram|diag\.?|chart|listing|abb\.?|abbildung|grafik|graph|plate|exhibit|"
    r"table|tab\.?|tbl\.?|tabelle)"
    r"(?:(?<=\.)\s*|\s+)"
    r"(?P<num>[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*)"
    r"(?:\s*(?P<sep>[:.\-\u2013\u2014])\s*(?P<rest>.*)|\s+(?P<bare>.*))?$",
    re.IGNORECASE,
)
_TABLE_CAPTION_KINDS = frozenset({"table", "tab", "tbl", "tabelle"})
_CAPTION_PROSE_START_RE = re.compile(
    r"^(?:shows?|illustrates?|depicts?|presents?|lists?|contains?|describes?|summarizes?|summarises?|"
    r"gives?|provides?|demonstrates?|compares?|displays?|exhibits?|indicates?|explains?|highlights?|"
    r"represents?|outlines?|is|are|was|were|can|will|has|have|in|on|above|below|"
    r"zeigt|stellt|beschreibt|gibt|enthält|ist|sind)\b",
    re.IGNORECASE,
)

_CHAPTER_RE = re.compile(r"^(?:chapter|part)\s+([0-9]+|[IVXLCDM]+)\b[:.\-\s]*(.*)$", re.IGNORECASE)
_APPENDIX_RE = re.compile(r"^appendix\s+([A-Z])(?:\.([0-9]+(?:\.[0-9]+)*))?\b[:.\-\s]*(.*)$", re.IGNORECASE)
_APPENDIX_SUB_RE = re.compile(r"^([A-Z]\.[0-9]+(?:\.[0-9]+)*)\.?\s+(.+)$")
_NUMERIC_HEADING_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)*)(\.?)\s+(.+)$")
_LEADING_NUMBER_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)*)(?:\.)?\s+(\S.*)$")
_SOFT_HYPHEN_WRAP_RE = re.compile(r"([A-Za-z])-\s*\n\s*([A-Za-z])")
_REQ_ID_RE = re.compile(r"^\[[A-Za-z0-9_.:-]{2,}\]")
_DECORATIVE_RUN_RE = re.compile(r"[.\-_=•·]{4,}")
_UML_MULTIPLICITY_RE = re.compile(r"\b\d+\.\.(?:\d+\b|\*)")
_BULLET_RE = re.compile(r"^[•\-\*\u2022\u25e6\u25aa\u25b3\u25bd]")
_MULTI_LIST_LINE_RE = re.compile(r"(?:^|\n)\s*(?:[0-9]+[.)]|[•\-\*\u2022])\s+\S")
_MAX_HEADING_LEVEL = 6
_MIN_STYLE_TIER_COUNT = 5
_SLIDE_TITLE_BAND = 0.2
_TITLE_BAND_MAX_WORDS = 24
_DEFAULT_SIZE_TOLERANCE = 0.05
_SLIDE_BOOKMARK_RE = re.compile(r"^(?:foli[ea]|slide|page)\s+\d+(?:\b|:)", re.IGNORECASE)
_TOC_HEADER_RE = re.compile(r"^(?:table\s+of\s+contents|contents)$", re.IGNORECASE)


def _clean_text(text: str) -> str:
    return " ".join(text.split())


def _join_wrapped_text(first: str, second: str) -> str:
    t1 = _clean_text(first)
    t2 = _clean_text(second)
    if not t1:
        return t2
    if not t2:
        return t1
    if len(t1) >= 2 and t1.endswith("-") and t1[-2].isalpha() and t2[0].islower():
        return t1[:-1] + t2
    return f"{t1} {t2}"


def _norm_title(text: str) -> str:
    dehyphenated = _SOFT_HYPHEN_WRAP_RE.sub(r"\1\2", text)
    return re.sub(r"[^a-z0-9]+", " ", _clean_text(dehyphenated).lower()).strip()


def _filter_meaningful_toc(toc: Sequence[tuple[int, str, int]]) -> list[tuple[int, str, int]]:
    cleaned = [(int(lvl), _clean_text(str(title)), int(page)) for lvl, title, page in toc if int(page) >= 1]
    if not cleaned:
        return []
    slide_like = sum(1 for _, title, _ in cleaned if _SLIDE_BOOKMARK_RE.match(title))
    return [] if slide_like >= 0.8 * len(cleaned) else cleaned


def _load_pdf_toc(pdf_path: str | Path) -> list[tuple[int, str, int]]:
    return _filter_meaningful_toc(extract_toc(pdf_path))


def _match_toc_entries(
    ordered: list[tuple[int, dict[str, Any]]],
    toc: Sequence[tuple[int, str, int]],
) -> tuple[dict[str, int], set[int]]:
    if not toc:
        return {}, set()
    first_target_page = min(page for _, _, page in toc)
    toc_pages: set[int] = set()
    in_toc = False
    for _, elem in ordered:
        if elem.get("type") != "text" or _is_ignored_element(elem):
            continue
        page = int(elem.get("page", 0))
        if page >= first_target_page:
            break
        if _TOC_HEADER_RE.match(_clean_text(elem.get("text", ""))):
            in_toc = True
        if in_toc:
            toc_pages.add(page)

    by_page: dict[int, list[tuple[int, str, str]]] = {}
    exact_idx: dict[tuple[int, str], list[tuple[int, str]]] = {}
    for idx, (lvl, title, page) in enumerate(toc):
        norm = _norm_title(title)
        if not norm:
            continue
        uid = f"{idx}:{norm}"
        by_page.setdefault(page, []).append((lvl, norm, uid))
        exact_idx.setdefault((page, norm), []).append((lvl, uid))

    matched_levels: dict[str, int] = {}
    used_toc: set[str] = set()
    for _, elem in ordered:
        if elem.get("type") != "text" or _is_ignored_element(elem):
            continue
        page = int(elem.get("page", 0))
        if page in toc_pages:
            continue
        norm_elem = _norm_title(elem.get("text", ""))
        if not norm_elem or len(norm_elem) > 160:
            continue

        exact_hit: tuple[int, str] | None = None
        for cand_page in (page, page - 1, page + 1):
            for lvl, uid in exact_idx.get((cand_page, norm_elem), ()):
                if uid not in used_toc:
                    exact_hit = (lvl, uid)
                    break
            if exact_hit is not None:
                break
        if exact_hit is not None:
            matched_levels[elem["id"]] = exact_hit[0]
            used_toc.add(exact_hit[1])
            continue

        best: tuple[float, int, str] | None = None
        n_elem = len(norm_elem)
        for cand_page in (page, page - 1, page + 1):
            for lvl, norm_toc, uid in by_page.get(cand_page, ()):
                if uid in used_toc:
                    continue
                n_toc = len(norm_toc)
                if min(n_elem, n_toc) < 0.70 * max(n_elem, n_toc):
                    continue
                if n_elem >= 6 and (norm_elem in norm_toc or norm_toc in norm_elem):
                    score = min(n_elem, n_toc) / max(n_elem, n_toc)
                else:
                    score = SequenceMatcher(None, norm_elem, norm_toc).ratio()
                if cand_page != page:
                    score -= 0.05
                if score >= 0.85 and (best is None or score > best[0]):
                    best = (score, lvl, uid)
        if best is not None:
            matched_levels[elem["id"]] = best[1]
            used_toc.add(best[2])

    return matched_levels, toc_pages


def _is_ignored_element(elem: dict[str, Any]) -> bool:
    if elem.get("level", 1) == 2:
        return True
    meta = elem.get("metadata") or {}
    return meta.get("role") in _IGNORED_ROLES or meta.get("layout_label") in _IGNORED_ROLES


def _has_alpha(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


def _sentence_count(text: str) -> int:
    return len(re.findall(r"[.!?](?:\s|$)", text))


def _ends_sentence(text: str) -> bool:
    return bool(text) and text[-1] in ".?!;,"


def _weighted_median_font_size(elements: Sequence[dict[str, Any]]) -> float | None:
    samples: list[tuple[float, int]] = []
    for elem in elements:
        if _is_ignored_element(elem) or elem.get("type") != "text":
            continue
        meta = elem.get("metadata") or {}
        if meta.get("role") == "caption" or meta.get("layout_label") == "caption":
            continue
        fs = meta.get("font_size")
        text = _clean_text(elem.get("text", ""))
        if isinstance(fs, (int, float)) and fs > 0 and len(text) > 0:
            samples.append((float(fs), len(text)))
    if not samples:
        return None
    samples.sort(key=lambda pair: pair[0])
    half = sum(w for _, w in samples) / 2.0
    acc = 0
    for fs, w in samples:
        acc += w
        if acc >= half:
            return fs
    return samples[-1][0]


def _project_semantic_regions(
    elements: Sequence[dict[str, Any]],
    semantic_regions: Mapping[int, Sequence[Any]] | None,
) -> None:
    if not semantic_regions:
        return
    for elem in elements:
        if elem.get("type") != "text" or elem.get("level", 1) != 1:
            continue
        if elem.get("source", "text_layer") != "text_layer":
            continue
        meta = elem.setdefault("metadata", {})
        if meta.get("layout_label") is not None:
            continue
        bbox = elem.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        ex0, ey0, ex1, ey1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        elem_area = max(0.0, ex1 - ex0) * max(0.0, ey1 - ey0)
        if elem_area <= 0.0:
            continue
        regions = semantic_regions.get(elem.get("page", 0))
        if not regions:
            continue
        best_score: tuple[float, float] | None = None
        best_label: str | None = None
        for reg in regions:
            if isinstance(reg, Mapping):
                rb = reg.get("bbox")
                label = reg.get("label")
                conf = float(reg.get("confidence", 1.0))
            else:
                rb = getattr(reg, "bbox", None)
                label = getattr(reg, "label", None)
                conf = float(getattr(reg, "confidence", 1.0))
            if not rb or len(rb) != 4 or not label:
                continue
            ix = max(0.0, min(ex1, float(rb[2])) - max(ex0, float(rb[0])))
            iy = max(0.0, min(ey1, float(rb[3])) - max(ey0, float(rb[1])))
            ioa = (ix * iy) / elem_area
            if ioa >= 0.60:
                score = (ioa, conf)
                if best_score is None or score > best_score:
                    best_score = score
                    best_label = str(label)
        if best_label is not None:
            meta["layout_label"] = best_label


def _parse_caption_prefix(text: str) -> tuple[bool, str | None]:
    cleaned = _clean_text(text)
    if not cleaned or len(cleaned) > 300:
        return False, None
    m = _CAPTION_PREFIX_RE.match(cleaned)
    if not m:
        return False, None
    kind_raw = m.group("kind").lower().rstrip(".")
    target_kind = "table" if kind_raw in _TABLE_CAPTION_KINDS else "figure"
    sep = m.group("sep")
    rest = m.group("rest")
    bare = m.group("bare")

    if sep is not None:
        if sep == "." and rest and _CAPTION_PROSE_START_RE.match(rest):
            return False, None
        return True, target_kind
    if bare is None or not bare.strip():
        return True, target_kind
    if _CAPTION_PROSE_START_RE.match(bare) or len(cleaned) > 140 or _sentence_count(cleaned) >= 2:
        return False, None
    return True, target_kind


def _bbox_metrics(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    h_overlap = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    acy = (ay0 + ay1) * 0.5
    bcy = (by0 + by1) * 0.5
    if acy <= bcy and ay1 <= by0 + 8.0:
        dy = max(0.0, by0 - ay1)
    elif acy > bcy and ay0 >= by1 - 8.0:
        dy = max(0.0, ay0 - by1)
    else:
        dy = min(abs(by0 - ay1), abs(ay0 - by1)) + 50.0
    return dy, h_overlap


def _horizontal_gap(a: Sequence[float], b: Sequence[float]) -> float:
    return max(0.0, max(a[0], b[0]) - min(a[2], b[2]))


def _merge_two_elements(primary: dict[str, Any], secondary: dict[str, Any]) -> None:
    primary["text"] = _join_wrapped_text(primary.get("text", ""), secondary.get("text", ""))
    b1 = primary.get("bbox")
    b2 = secondary.get("bbox")
    if b1 and b2 and len(b1) == 4 and len(b2) == 4:
        merged_box = [min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])]
        primary["bbox"] = merged_box
        src_ref = primary.get("source_ref")
        if isinstance(src_ref, dict) and "bbox" in src_ref:
            src_ref["bbox"] = list(merged_box)


def _vertically_adjacent(
    b1: Sequence[float] | None,
    b2: Sequence[float] | None,
    *,
    gap_scale: float,
    max_cap: float,
    align_tol: float,
) -> bool:
    if not b1 or not b2 or len(b1) != 4 or len(b2) != 4:
        return False
    h1 = max(1.0, float(b1[3]) - float(b1[1]))
    h2 = max(1.0, float(b2[3]) - float(b2[1]))
    dy = float(b2[1]) - float(b1[3])
    max_gap = max(6.0, min(max_cap, gap_scale * min(h1, h2)))
    if dy < -3.0 or dy > max_gap:
        return False
    if min(float(b1[2]), float(b2[2])) <= max(float(b1[0]), float(b2[0])):
        return False
    cx1 = (float(b1[0]) + float(b1[2])) * 0.5
    cx2 = (float(b2[0]) + float(b2[2])) * 0.5
    return abs(float(b1[0]) - float(b2[0])) <= align_tol or abs(cx1 - cx2) <= align_tol


def _is_adjacent_caption_candidate(
    elem: dict[str, Any],
    targets: Sequence[dict[str, Any]],
    body_font: float | None,
) -> bool:
    if elem.get("type") != "text" or _is_ignored_element(elem):
        return False
    meta = elem.get("metadata") or {}
    label = meta.get("layout_label")
    if label in _HEADING_LABELS or label in _NON_HEADING_LABELS:
        return False
    raw = elem.get("text", "")
    if _BULLET_RE.match(raw.strip()) or len(_MULTI_LIST_LINE_RE.findall(raw)) >= 2:
        return False
    cleaned = _clean_text(raw)
    if not cleaned or not _has_alpha(cleaned) or len(cleaned) > 120 or len(cleaned.split()) > 18:
        return False
    if cleaned[-1] in ":;?" or _sentence_count(cleaned) >= 2:
        return False
    if _parse_numbered_heading(cleaned, meta, body_font) is not None:
        return False
    fs = meta.get("font_size")
    if body_font and isinstance(fs, (int, float)) and fs >= body_font * 1.15:
        return False
    bbox = elem.get("bbox")
    if not bbox or len(bbox) != 4:
        return False
    for tgt in targets:
        tb = tgt.get("bbox")
        if not tb or len(tb) != 4:
            continue
        dy, h_ov = _bbox_metrics(bbox, tb)
        if dy <= 18.0 and h_ov > 0.0:
            return True
    return False


def _merge_multiline_captions(
    elements: list[dict[str, Any]],
    ordered: list[tuple[int, dict[str, Any]]],
    body_font: float | None,
) -> list[tuple[int, dict[str, Any]]]:
    dropped_ids: set[str] = set()
    n = len(ordered)
    i = 0
    while i + 1 < n:
        _, c1 = ordered[i]
        _, c2 = ordered[i + 1]
        if (
            c1.get("type") != "text"
            or c2.get("type") != "text"
            or c1.get("level", 1) != 1
            or c2.get("level", 1) != 1
            or c1.get("page") != c2.get("page")
            or _is_ignored_element(c1)
            or _is_ignored_element(c2)
        ):
            i += 1
            continue

        m1 = c1.get("metadata") or {}
        m2 = c2.get("metadata") or {}
        l1 = m1.get("layout_label")
        l2 = m2.get("layout_label")
        is_cap1 = l1 == "caption" or m1.get("role") == "caption" or _parse_caption_prefix(c1.get("text", ""))[0]
        if not is_cap1 or l2 in _HEADING_LABELS or (l2 in _NON_HEADING_LABELS and l2 != "caption"):
            i += 1
            continue

        raw2 = c2.get("text", "")
        if _BULLET_RE.match(raw2.strip()) or len(_MULTI_LIST_LINE_RE.findall(raw2)) >= 2:
            i += 1
            continue
        t2 = _clean_text(raw2)
        if (
            not t2
            or not _has_alpha(t2)
            or len(t2) > 140
            or len(t2.split()) > 20
            or _sentence_count(t2) >= 2
            or _parse_caption_prefix(t2)[0]
            or _parse_numbered_heading(t2, m2, body_font) is not None
        ):
            i += 1
            continue

        fs1 = m1.get("font_size")
        fs2 = m2.get("font_size")
        has_fs1 = isinstance(fs1, (int, float)) and fs1 > 0
        has_fs2 = isinstance(fs2, (int, float)) and fs2 > 0
        if has_fs1 and has_fs2:
            if abs(float(fs1) - float(fs2)) > max(0.5, 0.05 * max(float(fs1), float(fs2))):
                i += 1
                continue
        elif l2 != "caption":
            i += 1
            continue
        if bool(m1.get("bold")) != bool(m2.get("bold")) and l2 != "caption":
            i += 1
            continue

        if not _vertically_adjacent(c1.get("bbox"), c2.get("bbox"), gap_scale=0.6, max_cap=12.0, align_tol=40.0):
            i += 1
            continue

        merged_text = _join_wrapped_text(c1.get("text", ""), raw2)
        if len(merged_text) > 260 or _sentence_count(merged_text) > 2:
            i += 1
            continue

        _merge_two_elements(c1, c2)
        c1.setdefault("metadata", {})["role"] = "caption"
        dropped_ids.add(c2["id"])
        i += 2

    if not dropped_ids:
        return ordered
    elements[:] = [e for e in elements if e.get("id") not in dropped_ids]
    return [pair for pair in ordered if pair[1].get("id") not in dropped_ids]


def _link_captions(
    elements: list[dict[str, Any]],
    body_font: float | None,
    protected_ids: set[str] = frozenset(),
) -> None:
    pages: dict[int, list[dict[str, Any]]] = {}
    for elem in elements:
        pages.setdefault(elem.get("page", 0), []).append(elem)

    for page_elems in pages.values():
        targets = [e for e in page_elems if e.get("level", 1) == 1 and e.get("type") in {"figure", "table"}]
        explicit_caps: list[tuple[dict[str, Any], str | None, float]] = []
        for elem in page_elems:
            if elem.get("type") != "text" or _is_ignored_element(elem):
                continue
            meta = elem.setdefault("metadata", {})
            label = meta.get("layout_label")
            if label in _HEADING_LABELS or (label in _NON_HEADING_LABELS and label != "caption"):
                continue
            is_prefixed, pref_kind = _parse_caption_prefix(elem.get("text", ""))
            if label == "caption" or meta.get("role") == "caption" or is_prefixed:
                meta["role"] = "caption"
                explicit_caps.append((elem, pref_kind, 120.0))

        claimed_caps: set[str] = set()
        claimed_targets: set[str] = set()
        _match_caption_pool(explicit_caps, targets, claimed_caps, claimed_targets)

        remaining_targets = [t for t in targets if t["id"] not in claimed_targets]
        if not remaining_targets:
            continue
        adj_caps: list[tuple[dict[str, Any], str | None, float]] = []
        for elem in page_elems:
            if elem["id"] in claimed_caps or elem["id"] in protected_ids:
                continue
            if _is_adjacent_caption_candidate(elem, remaining_targets, body_font):
                adj_caps.append((elem, None, 18.0))
        _match_caption_pool(adj_caps, remaining_targets, claimed_caps, claimed_targets)


def _min_cost_bipartite_match(
    n_left: int,
    n_right: int,
    edges: list[tuple[int, int, float]],
) -> list[tuple[int, int]]:
    # Successive shortest augmenting path (SPFA) yields max-cardinality min-cost matching.
    if n_left == 0 or n_right == 0 or not edges:
        return []
    src = 0
    sink = 1 + n_left + n_right
    n_nodes = sink + 1
    adj: list[list[list[Any]]] = [[] for _ in range(n_nodes)]

    def add_edge(u: int, v: int, cap: int, cost: float) -> None:
        fwd = [v, cap, cost, len(adj[v])]
        rev = [u, 0, -cost, len(adj[u])]
        adj[u].append(fwd)
        adj[v].append(rev)

    for i in range(n_left):
        add_edge(src, 1 + i, 1, 0.0)
    for j in range(n_right):
        add_edge(1 + n_left + j, sink, 1, 0.0)
    for i, j, cost in edges:
        add_edge(1 + i, 1 + n_left + j, 1, cost)

    inf = float("inf")
    while True:
        dist = [inf] * n_nodes
        parent_u = [-1] * n_nodes
        parent_e = [-1] * n_nodes
        in_q = [False] * n_nodes
        queue = [src]
        q_idx = 0
        dist[src] = 0.0
        in_q[src] = True
        while q_idx < len(queue):
            u = queue[q_idx]
            q_idx += 1
            in_q[u] = False
            du = dist[u]
            for e_idx, (v, cap, cost, _) in enumerate(adj[u]):
                if cap > 0 and du + cost + 1e-9 < dist[v]:
                    dist[v] = du + cost
                    parent_u[v] = u
                    parent_e[v] = e_idx
                    if not in_q[v]:
                        queue.append(v)
                        in_q[v] = True
        if dist[sink] == inf:
            break
        v = sink
        while v != src:
            u = parent_u[v]
            e = adj[u][parent_e[v]]
            e[1] -= 1
            adj[v][e[3]][1] += 1
            v = u

    matched: list[tuple[int, int]] = []
    for i in range(n_left):
        u = 1 + i
        for v, cap, _, _ in adj[u]:
            if 1 + n_left <= v < sink and cap == 0:
                matched.append((i, v - 1 - n_left))
                break
    return matched


def _match_caption_pool(
    captions: list[tuple[dict[str, Any], str | None, float]],
    targets: list[dict[str, Any]],
    claimed_caps: set[str],
    claimed_targets: set[str],
) -> None:
    if not captions or not targets:
        return
    edges: list[tuple[int, int, float]] = []
    for i, (cap, pref_kind, max_gap) in enumerate(captions):
        if cap["id"] in claimed_caps:
            continue
        cb = cap.get("bbox")
        if not cb or len(cb) != 4:
            continue
        cb_cx = (cb[0] + cb[2]) * 0.5
        for j, tgt in enumerate(targets):
            if tgt["id"] in claimed_targets:
                continue
            if pref_kind is not None and tgt.get("type") != pref_kind:
                continue
            tb = tgt.get("bbox")
            if not tb or len(tb) != 4:
                continue
            dy, h_ov = _bbox_metrics(cb, tb)
            if dy > max_gap:
                continue
            h_gap = _horizontal_gap(cb, tb)
            if h_ov <= 0.0 and (max_gap <= 24.0 or h_gap > 24.0):
                continue
            overlap_penalty = 1000.0 if h_ov <= 0.0 else 0.0
            min_w = max(1.0, min(cb[2] - cb[0], tb[2] - tb[0]))
            dx_center = abs(cb_cx - (tb[0] + tb[2]) * 0.5)
            cost = overlap_penalty + dy + h_gap + 0.1 * dx_center - 5.0 * (h_ov / min_w)
            edges.append((i, j, cost))

    for i, j in _min_cost_bipartite_match(len(captions), len(targets), edges):
        cap = captions[i][0]
        tgt = targets[j]
        cid = cap["id"]
        tid = tgt["id"]
        if cid in claimed_caps or tid in claimed_targets:
            continue
        claimed_caps.add(cid)
        claimed_targets.add(tid)
        cmeta = cap.setdefault("metadata", {})
        tmeta = tgt.setdefault("metadata", {})
        cmeta["role"] = "caption"
        cmeta["caption_of"] = tid
        tmeta["caption_id"] = cid
        tmeta["caption_text"] = cap.get("text", "")


def _propagate_captions_to_ir(out: dict[str, Any], elements: Sequence[dict[str, Any]]) -> None:
    by_id: dict[str, dict[str, Any]] = {}
    by_table_id: dict[str, dict[str, Any]] = {}
    by_diagram_id: dict[str, dict[str, Any]] = {}
    for elem in elements:
        eid = elem.get("id")
        if eid:
            by_id[eid] = elem
        meta = elem.get("metadata") or {}
        if meta.get("table_id"):
            by_table_id[meta["table_id"]] = elem
        if meta.get("diagram_id"):
            by_diagram_id[meta["diagram_id"]] = elem

    for key, alt_map in (("tables", by_table_id), ("diagrams", by_diagram_id)):
        for entry in out.get(key) or ():
            if not isinstance(entry, dict):
                continue
            meta = entry.setdefault("metadata", {})
            elem_id = entry.get("element_id") or meta.get("element_id")
            elem = by_id.get(elem_id) if elem_id else alt_map.get(entry.get("id"))
            if elem is None:
                continue
            emeta = elem.get("metadata") or {}
            cap_id = emeta.get("caption_id")
            if cap_id is None:
                continue
            cap_text = emeta.get("caption_text", "")
            entry["caption_id"] = cap_id
            entry["caption_text"] = cap_text
            meta["caption_id"] = cap_id
            meta["caption_text"] = cap_text


def _parse_numbered_heading(
    cleaned: str,
    meta: dict[str, Any],
    body_font: float | None,
) -> int | None:
    if len(cleaned) > 140 or len(cleaned.split()) > 18 or _sentence_count(cleaned) >= 2:
        return None

    fs = meta.get("font_size")
    styled = bool(meta.get("bold")) or (
        body_font is not None and isinstance(fs, (int, float)) and fs >= body_font * 1.05
    )
    explicit_label = meta.get("layout_label") in _HEADING_LABELS

    m_chap = _CHAPTER_RE.match(cleaned)
    if m_chap:
        rest = m_chap.group(2).strip()
        if (rest and _has_alpha(rest) and _title_starts_well(rest)) or (not rest and (styled or explicit_label)):
            return 1

    m_app = _APPENDIX_RE.match(cleaned)
    if m_app:
        sub = m_app.group(2)
        rest = m_app.group(3).strip()
        if (rest and _has_alpha(rest) and _title_starts_well(rest)) or (not rest and (styled or explicit_label)):
            return 1 if not sub else min(6, 1 + len(sub.split(".")))

    m_asub = _APPENDIX_SUB_RE.match(cleaned)
    if m_asub:
        parts = m_asub.group(1).split(".")
        rest = m_asub.group(2).strip()
        if _has_alpha(rest) and _title_starts_well(rest):
            return min(6, len(parts))

    m_num = _NUMERIC_HEADING_RE.match(cleaned)
    if m_num:
        num_str, dot, rest = m_num.groups()
        parts = num_str.split(".")
        first = int(parts[0])
        if first < 1 or first > 99 or len(parts) > 6:
            return None
        allow_camel = len(parts) >= 2 and (styled or explicit_label)
        if not _has_alpha(rest) or not _title_starts_well(rest, allow_camel=allow_camel):
            return None
        if len(rest.split()) == 1 and any(ch.isdigit() for ch in rest) and "-" in rest:
            return None

        if len(parts) == 1 and not dot and not (styled or explicit_label):
            return None
        if not (styled or explicit_label) and _ends_sentence(cleaned):
            return None
        return len(parts)

    return None


def _title_starts_well(rest: str, *, allow_camel: bool = False) -> bool:
    stripped = rest.lstrip("\"'([")
    if not stripped:
        return False
    first = stripped[0]
    if first.isupper() or (not first.isascii() and first.isalpha()):
        return True
    if allow_camel and first.islower():
        first_tok = stripped.split()[0]
        return any(ch.isupper() for ch in first_tok[1:])
    return False


def _is_heading_candidate(
    elem: dict[str, Any],
    body_font: float | None,
    title_band: bool = False,
) -> tuple[bool, int | None]:
    if elem.get("type") != "text" or _is_ignored_element(elem):
        return False, None
    meta = elem.get("metadata") or {}
    if meta.get("role") == "caption":
        return False, None
    label = meta.get("layout_label")
    if label in _NON_HEADING_LABELS:
        return False, None

    raw = elem.get("text", "")
    if _BULLET_RE.match(raw.strip()) or len(_MULTI_LIST_LINE_RE.findall(raw)) >= 2:
        return False, None

    cleaned = _clean_text(raw)
    if not cleaned or not _has_alpha(cleaned) or len(cleaned) <= 1:
        return False, None

    num_depth = _parse_numbered_heading(cleaned, meta, body_font)

    if label == "title":
        return True, 1
    if label == "section_header":
        return True, num_depth
    if num_depth is not None:
        return True, num_depth

    if (
        _REQ_ID_RE.match(cleaned)
        or cleaned.endswith(":")
        or _DECORATIVE_RUN_RE.search(cleaned)
        or _UML_MULTIPLICITY_RE.search(cleaned)
    ):
        return False, None

    n_words = len(cleaned.split())
    if len(cleaned) > 140 or n_words > (_TITLE_BAND_MAX_WORDS if title_band else 16) or _sentence_count(cleaned) >= 2:
        return False, None

    fs = meta.get("font_size")
    bold = bool(meta.get("bold"))
    has_fs = isinstance(fs, (int, float)) and fs > 0
    bold_ok = bold and not _ends_sentence(cleaned) and n_words <= (_TITLE_BAND_MAX_WORDS if title_band else 12)

    if body_font is not None and has_fs:
        return fs >= body_font * 1.15 or (bold_ok and fs >= body_font * 0.98), None
    return bold_ok, None


def _style_key(elem: dict[str, Any]) -> tuple[float, bool] | None:
    meta = elem.get("metadata") or {}
    fs = meta.get("font_size")
    if not isinstance(fs, (int, float)) or fs <= 0:
        return None
    return (round(float(fs), 2), bool(meta.get("bold")))


def _cluster_styles(
    styles: list[tuple[float, bool]],
    counts: dict[tuple[float, bool], int],
    max_levels: int = _MAX_HEADING_LEVEL,
    size_tolerance: float = _DEFAULT_SIZE_TOLERANCE,
) -> dict[tuple[float, bool], int]:
    # Group sizes within relative tolerance first, split by bold, then merge rare tiers down to <= max_levels.
    if not styles:
        return {}
    buckets: list[list[tuple[float, bool]]] = []
    anchor = styles[0][0]
    cur: list[tuple[float, bool]] = []
    for st in styles:
        if cur and anchor > 0 and (anchor - st[0]) / anchor > size_tolerance:
            buckets.append(cur)
            cur = [st]
            anchor = st[0]
        else:
            cur.append(st)
    if cur:
        buckets.append(cur)

    groups: list[list[tuple[float, bool]]] = []
    for bucket in buckets:
        bold_grp = [st for st in bucket if st[1]]
        reg_grp = [st for st in bucket if not st[1]]
        if bold_grp:
            groups.append(bold_grp)
        if reg_grp:
            groups.append(reg_grp)

    while len(groups) > max_levels:
        def _merge_cost(i: int) -> tuple[bool, float, int, int]:
            c_hi = sum(counts[s] for s in groups[i])
            c_lo = sum(counts[s] for s in groups[i + 1])
            gap = groups[i][-1][0] - groups[i + 1][0][0]
            return (min(c_hi, c_lo) >= _MIN_STYLE_TIER_COUNT, gap, min(c_hi, c_lo), c_hi + c_lo)

        best_i = min(range(len(groups) - 1), key=_merge_cost)
        groups[best_i].extend(groups[best_i + 1])
        del groups[best_i + 1]
    return {style: rank for rank, grp in enumerate(groups, start=1) for style in grp}


def _slide_title_ids(
    headings: list[tuple[dict[str, Any], int | None]],
    landscape_heights: dict[int, float],
) -> set[str]:
    # On landscape pages the topmost unnumbered heading inside the title band is the slide title.
    topmost: dict[Any, tuple[dict[str, Any], int | None]] = {}
    for elem, depth in headings:
        bbox = elem.get("bbox")
        if not bbox or len(bbox) != 4 or elem.get("page") not in landscape_heights:
            continue
        current = topmost.get(elem["page"])
        if current is None or bbox[1] < current[0]["bbox"][1]:
            topmost[elem["page"]] = (elem, depth)
    return {
        elem["id"]
        for page, (elem, depth) in topmost.items()
        if depth is None
        and (elem.get("metadata") or {}).get("layout_label") != "title"
        and elem["bbox"][1] < _SLIDE_TITLE_BAND * landscape_heights[page]
    }


def _merge_multiline_headings(
    elements: list[dict[str, Any]],
    ordered: list[tuple[int, dict[str, Any]]],
    body_font: float | None,
    top_band_ids: set[str],
) -> list[tuple[int, dict[str, Any]]]:
    dropped_ids: set[str] = set()
    n = len(ordered)
    i = 0
    while i + 1 < n:
        _, e1 = ordered[i]
        _, e2 = ordered[i + 1]
        if (
            e1.get("type") != "text"
            or e2.get("type") != "text"
            or e1.get("level", 1) != 1
            or e2.get("level", 1) != 1
            or e1.get("page") != e2.get("page")
            or _is_ignored_element(e1)
            or _is_ignored_element(e2)
        ):
            i += 1
            continue

        m1 = e1.get("metadata") or {}
        m2 = e2.get("metadata") or {}
        l1 = m1.get("layout_label")
        l2 = m2.get("layout_label")
        if (
            m1.get("role") == "caption"
            or m2.get("role") == "caption"
            or l1 in _NON_HEADING_LABELS
            or l2 in _NON_HEADING_LABELS
            or (l1 is not None and l2 is not None and l1 != l2)
        ):
            i += 1
            continue

        raw1 = e1.get("text", "")
        raw2 = e2.get("text", "")
        if _parse_caption_prefix(raw1)[0] or _parse_caption_prefix(raw2)[0]:
            i += 1
            continue

        fs1 = m1.get("font_size")
        fs2 = m2.get("font_size")
        has_fs1 = isinstance(fs1, (int, float)) and fs1 > 0
        has_fs2 = isinstance(fs2, (int, float)) and fs2 > 0
        if has_fs1 and has_fs2:
            if bool(m1.get("bold")) != bool(m2.get("bold")):
                i += 1
                continue
            if abs(float(fs1) - float(fs2)) > max(0.3, 0.02 * max(float(fs1), float(fs2))):
                i += 1
                continue
        elif not (l1 in _HEADING_LABELS and l1 == l2):
            i += 1
            continue

        if not _vertically_adjacent(e1.get("bbox"), e2.get("bbox"), gap_scale=0.55, max_cap=14.0, align_tol=30.0):
            i += 1
            continue

        t1 = _clean_text(raw1)
        t2 = _clean_text(raw2)
        if not t1 or not t2 or t1[-1] in ".?!;:":
            i += 1
            continue
        if (
            _BULLET_RE.match(raw2.strip())
            or len(_MULTI_LIST_LINE_RE.findall(raw2)) >= 2
            or _REQ_ID_RE.match(t2)
            or t2.endswith(":")
            or _DECORATIVE_RUN_RE.search(t2)
            or _UML_MULTIPLICITY_RE.search(t2)
            or _CHAPTER_RE.match(t2)
            or _APPENDIX_RE.match(t2)
            or _APPENDIX_SUB_RE.match(t2)
            or _LEADING_NUMBER_RE.match(t2)
            or _parse_numbered_heading(t2, m2, body_font) is not None
        ):
            i += 1
            continue

        in_band = e1["id"] in top_band_ids
        if not _is_heading_candidate(e1, body_font, in_band)[0]:
            i += 1
            continue
        probe = {"type": "text", "text": _join_wrapped_text(raw1, raw2), "metadata": m1}
        if not _is_heading_candidate(probe, body_font, in_band)[0]:
            i += 1
            continue

        _merge_two_elements(e1, e2)
        dropped_ids.add(e2["id"])
        i += 2

    if not dropped_ids:
        return ordered
    elements[:] = [e for e in elements if e.get("id") not in dropped_ids]
    return [pair for pair in ordered if pair[1].get("id") not in dropped_ids]


def _validate_heading_sequence(
    headings: list[tuple[dict[str, Any], int | None]],
    toc_matches: dict[str, int],
    slide_titles: set[str],
) -> list[tuple[dict[str, Any], int | None]]:
    parsed: list[tuple[dict[str, Any], int | None, tuple[int, ...] | None, bool, bool]] = []
    top_counts: dict[int, int] = {}
    for elem, explicit_depth in headings:
        eid = elem["id"]
        has_toc = eid in toc_matches
        is_title = (elem.get("metadata") or {}).get("layout_label") == "title" or eid in slide_titles
        m = _LEADING_NUMBER_RE.match(_clean_text(elem.get("text", "")))
        parts: tuple[int, ...] | None = None
        if m:
            raw_parts = m.group(1).split(".")
            if len(raw_parts) <= 6:
                parts = tuple(int(p) for p in raw_parts)
                if 1 <= parts[0] <= 99:
                    top_counts[parts[0]] = top_counts.get(parts[0], 0) + 1
        parsed.append((elem, explicit_depth, parts, has_toc, is_title))

    validated: list[tuple[dict[str, Any], int | None]] = []
    prev_top = 0
    for elem, explicit_depth, parts, has_toc, is_title in parsed:
        if parts is None or has_toc or is_title:
            if parts is not None and has_toc and 1 <= parts[0] <= 99:
                prev_top = max(prev_top, parts[0])
            validated.append((elem, explicit_depth))
            continue

        first = parts[0]
        if first < 1 or first > 99:
            continue
        has_seq_support = top_counts.get(first, 0) >= 2 or (first + 1) in top_counts
        if prev_top == 0:
            ok = first <= 3 or (len(parts) >= 2 and first <= 20 and explicit_depth is not None) or has_seq_support
        else:
            ok = first <= prev_top + 3 or (first <= prev_top + 10 and has_seq_support)
        if not ok:
            continue
        prev_top = max(prev_top, first)
        validated.append((elem, explicit_depth))
    return validated


def _assign_heading_levels(
    headings: list[tuple[dict[str, Any], int | None]],
    slide_titles: set[str] = frozenset(),
    size_tolerance: float = _DEFAULT_SIZE_TOLERANCE,
    toc_ids: set[str] | Mapping[str, int] = frozenset(),
    body_font: float | None = None,
) -> dict[str, int]:
    slide_pages = {elem.get("page") for elem, _ in headings if elem["id"] in slide_titles}
    has_title = any((elem.get("metadata") or {}).get("layout_label") == "title" for elem, _ in headings)

    anchor_votes: dict[tuple[float, bool], dict[int, int]] = {}
    toc_anchor_keys: set[tuple[float, bool]] = set()
    unanchored_counts: dict[tuple[float, bool], int] = {}
    for elem, explicit_depth in headings:
        eid = elem["id"]
        if eid in slide_titles or (elem.get("metadata") or {}).get("layout_label") == "title":
            continue
        key = _style_key(elem)
        if key is None:
            continue
        if explicit_depth is not None:
            in_toc = eid in toc_ids
            if not in_toc and (
                elem.get("page") in slide_pages
                or (body_font is not None and key[0] < body_font * 0.98)
            ):
                continue
            if in_toc:
                toc_anchor_keys.add(key)
            bucket = anchor_votes.setdefault(key, {})
            bucket[explicit_depth] = bucket.get(explicit_depth, 0) + 1
        else:
            unanchored_counts[key] = unanchored_counts.get(key, 0) + 1

    anchor_levels: dict[tuple[float, bool], tuple[int, int]] = {}
    if not (len(slide_pages) >= 2 and not toc_anchor_keys):
        for key, votes in anchor_votes.items():
            total_anchor = sum(votes.values())
            if key in toc_anchor_keys or total_anchor >= unanchored_counts.get(key, 0):
                best_depth = min(votes, key=lambda d: (-votes[d], d))
                anchor_levels[key] = (best_depth, total_anchor)

    style_rank: dict[tuple[float, bool], int] = {}
    if not anchor_levels:
        styles = sorted(unanchored_counts, reverse=True)
        style_rank = _cluster_styles(
            styles,
            unanchored_counts,
            _MAX_HEADING_LEVEL - int(has_title or bool(slide_pages)),
            size_tolerance,
        )
    else:
        unmatched_counts: dict[tuple[float, bool], int] = {}
        for key, cnt in unanchored_counts.items():
            if key in anchor_levels:
                style_rank[key] = anchor_levels[key][0]
                continue
            fs, bold = key
            best: tuple[float, int, int] | None = None
            for (afs, abold), (alvl, acnt) in anchor_levels.items():
                if abold == bold and afs > 0 and abs(fs - afs) / max(fs, afs) <= size_tolerance:
                    cand = (abs(fs - afs), -acnt, alvl)
                    if best is None or cand < best:
                        best = cand
            if best is not None:
                style_rank[key] = best[2]
            else:
                unmatched_counts[key] = cnt

        if unmatched_counts:
            unmatched_styles = sorted(unmatched_counts, reverse=True)
            rel_ranks = _cluster_styles(unmatched_styles, unmatched_counts, _MAX_HEADING_LEVEL, size_tolerance)
            anchors_desc = sorted(anchor_levels.items(), key=lambda kv: kv[0], reverse=True)
            prev_assigned: dict[int, int] = {}
            for st in unmatched_styles:
                rank_idx = rel_ranks[st]
                if rank_idx in prev_assigned:
                    style_rank[st] = prev_assigned[rank_idx]
                    continue
                above = [(ak, alvl) for ak, (alvl, _) in anchors_desc if ak > st]
                if not above:
                    lvl = 1
                else:
                    lvl = min(_MAX_HEADING_LEVEL, above[-1][1] + 1)
                    if rank_idx > 1 and (rank_idx - 1) in prev_assigned and prev_assigned[rank_idx - 1] >= above[-1][1]:
                        lvl = min(_MAX_HEADING_LEVEL, prev_assigned[rank_idx - 1] + 1)
                prev_assigned[rank_idx] = lvl
                style_rank[st] = lvl

    levels: dict[str, int] = {}
    for elem, explicit_depth in headings:
        if explicit_depth is not None:
            levels[elem["id"]] = explicit_depth
            continue
        meta = elem.get("metadata") or {}
        if meta.get("layout_label") == "title" or elem["id"] in slide_titles:
            levels[elem["id"]] = 1
            continue
        key = _style_key(elem)
        if key is not None:
            shift = int(not anchor_levels and (has_title or elem.get("page") in slide_pages))
            levels[elem["id"]] = min(_MAX_HEADING_LEVEL, style_rank[key] + shift)
        else:
            base_lvl = 2 if has_title else 1
            levels[elem["id"]] = base_lvl + int(not has_title and elem.get("page") in slide_pages)
    return levels


def apply_structure(
    ir: dict[str, Any],
    *,
    toc: Sequence[tuple[int, str, int]] = (),
    pdf_path: str | Path | None = None,
    use_bookmarks: bool = True,
    style_size_tolerance: float = _DEFAULT_SIZE_TOLERANCE,
    semantic_regions: Mapping[int, Sequence[Any]] | None = None,
    inplace: bool = False,
) -> dict[str, Any]:
    out = ir if inplace else copy.deepcopy(ir)
    elements: list[dict[str, Any]] = out.get("elements", [])
    doc_meta = out.setdefault("metadata", {})

    if not elements:
        doc_meta["outline"] = []
        return out

    _project_semantic_regions(elements, semantic_regions)

    resolved_toc: list[tuple[int, str, int]] = []
    if use_bookmarks:
        resolved_toc = _filter_meaningful_toc(toc) if toc else (_load_pdf_toc(pdf_path) if pdf_path is not None else [])

    body_font = _weighted_median_font_size(elements)
    landscape_heights = {
        p["number"]: float(p["height_pt"])
        for p in doc_meta.get("pages", ())
        if p.get("width_pt") and p.get("height_pt") and p["width_pt"] > p["height_pt"]
    }
    top_band_ids = {
        e["id"]
        for e in elements
        if e.get("page") in landscape_heights
        and len(e.get("bbox") or ()) == 4
        and e["bbox"][1] < _SLIDE_TITLE_BAND * landscape_heights[e["page"]]
    }

    ordered = sorted(
        enumerate(elements),
        key=lambda pair: (pair[1].get("page", 0), pair[1].get("reading_order", pair[0]), pair[0]),
    )
    ordered = _merge_multiline_headings(elements, ordered, body_font, top_band_ids)
    toc_matches, toc_pages = _match_toc_entries(ordered, resolved_toc) if resolved_toc else ({}, set())
    ordered = _merge_multiline_captions(elements, ordered, body_font)

    def collect_headings() -> list[tuple[dict[str, Any], int | None]]:
        found = []
        for _, elem in ordered:
            eid = elem["id"]
            if eid in toc_matches:
                found.append((elem, toc_matches[eid]))
                continue
            if elem.get("page") in toc_pages:
                continue
            is_heading, explicit_depth = _is_heading_candidate(elem, body_font, eid in top_band_ids)
            if is_heading:
                found.append((elem, explicit_depth))
        return found

    _link_captions(elements, body_font, _slide_title_ids(collect_headings(), landscape_heights))
    _propagate_captions_to_ir(out, elements)
    raw_candidates = collect_headings()
    raw_slide_titles = _slide_title_ids(raw_candidates, landscape_heights)
    heading_candidates = _validate_heading_sequence(raw_candidates, toc_matches, raw_slide_titles)
    heading_levels = _assign_heading_levels(
        heading_candidates,
        _slide_title_ids(heading_candidates, landscape_heights),
        style_size_tolerance,
        toc_matches,
        body_font,
    )
    stack: list[tuple[int, str]] = []
    outline: list[dict[str, Any]] = []

    for _, elem in ordered:
        if _is_ignored_element(elem):
            continue
        meta = elem.setdefault("metadata", {})
        eid = elem["id"]
        if eid in heading_levels:
            h_level = heading_levels[eid]
            while stack and stack[-1][0] >= h_level:
                stack.pop()
            parent_id = stack[-1][1] if stack else None
            meta["role"] = "heading"
            meta["heading_level"] = h_level
            meta["section_parent_id"] = parent_id
            stack.append((h_level, eid))
            outline.append(
                {
                    "id": eid,
                    "title": _clean_text(elem.get("text", "")),
                    "level": h_level,
                    "page": elem.get("page"),
                    "parent_id": parent_id,
                }
            )
        else:
            meta["section_parent_id"] = stack[-1][1] if stack else None

    doc_meta["outline"] = outline
    return out


def build_section_tree(elements: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    roots: list[dict[str, Any]] = []

    for elem in elements:
        if _is_ignored_element(elem):
            continue
        meta = elem.get("metadata") or {}
        if meta.get("role") == "heading":
            node: dict[str, Any] = {
                "id": elem["id"],
                "title": _clean_text(elem.get("text", "")),
                "level": meta.get("heading_level", 1),
                "page": elem.get("page"),
                "parent_id": meta.get("section_parent_id"),
                "element_ids": [],
                "children": [],
            }
            nodes[elem["id"]] = node
            parent_id = node["parent_id"]
            if parent_id is not None and parent_id in nodes:
                nodes[parent_id]["children"].append(node)
            else:
                roots.append(node)
        else:
            sec_id = meta.get("section_parent_id")
            if sec_id is not None and sec_id in nodes:
                nodes[sec_id]["element_ids"].append(elem["id"])

    return roots
