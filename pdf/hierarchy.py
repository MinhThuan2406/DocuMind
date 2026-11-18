from __future__ import annotations

import copy
import re
from collections.abc import Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pymupdf

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
    r"^(?P<kind>figure|fig\.?|table|tab\.?)\s+"
    r"(?P<num>[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*)"
    r"(?:\s*(?P<sep>[:.\-\u2013\u2014])\s*(?P<rest>.*)|\s+(?P<bare>.*))?$",
    re.IGNORECASE,
)
_CAPTION_PROSE_START_RE = re.compile(
    r"^(?:shows?|illustrates?|depicts?|presents?|lists?|contains?|describes?|summarizes?|"
    r"gives?|provides?|demonstrates?|compares?|is|are|was|were|can|will|has|have|in|on|above|below)\b",
    re.IGNORECASE,
)

_CHAPTER_RE = re.compile(r"^(?:chapter|part)\s+([0-9]+|[IVXLCDM]+)\b[:.\-\s]*(.*)$", re.IGNORECASE)
_APPENDIX_RE = re.compile(r"^appendix\s+([A-Z])(?:\.([0-9]+(?:\.[0-9]+)*))?\b[:.\-\s]*(.*)$", re.IGNORECASE)
_APPENDIX_SUB_RE = re.compile(r"^([A-Z]\.[0-9]+(?:\.[0-9]+)*)\.?\s+(.+)$")
_NUMERIC_HEADING_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)*)(\.?)\s+(.+)$")
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


def _norm_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _clean_text(text).lower()).strip()


def _filter_meaningful_toc(toc: Sequence[tuple[int, str, int]]) -> list[tuple[int, str, int]]:
    cleaned = [(int(lvl), _clean_text(str(title)), int(page)) for lvl, title, page in toc if int(page) >= 1]
    if not cleaned:
        return []
    slide_like = sum(1 for _, title, _ in cleaned if _SLIDE_BOOKMARK_RE.match(title))
    return [] if slide_like >= 0.8 * len(cleaned) else cleaned


def _load_pdf_toc(pdf_path: str | Path) -> list[tuple[int, str, int]]:
    doc = pymupdf.open(str(pdf_path))
    try:
        raw = doc.get_toc()
    finally:
        doc.close()
    return _filter_meaningful_toc(raw)


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


def _parse_caption_prefix(text: str) -> tuple[bool, str | None]:
    cleaned = _clean_text(text)
    if not cleaned or len(cleaned) > 300:
        return False, None
    m = _CAPTION_PREFIX_RE.match(cleaned)
    if not m:
        return False, None
    kind_raw = m.group("kind").lower().rstrip(".")
    target_kind = "figure" if kind_raw.startswith("fig") else "table"
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


def _match_caption_pool(
    captions: list[tuple[dict[str, Any], str | None, float]],
    targets: list[dict[str, Any]],
    claimed_caps: set[str],
    claimed_targets: set[str],
) -> None:
    candidates: list[tuple[tuple[int, float, float], dict[str, Any], dict[str, Any]]] = []
    for cap, pref_kind, max_gap in captions:
        cb = cap.get("bbox")
        if not cb or len(cb) != 4:
            continue
        for tgt in targets:
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
            overlap_penalty = 0 if h_ov > 0.0 else 1
            min_w = max(1.0, min(cb[2] - cb[0], tb[2] - tb[0]))
            score = (overlap_penalty, dy + h_gap, -(h_ov / min_w))
            candidates.append((score, cap, tgt))

    candidates.sort(key=lambda item: item[0])
    for _, cap, tgt in candidates:
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
        if not _has_alpha(rest) or not _title_starts_well(rest):
            return None
        if len(rest.split()) == 1 and any(ch.isdigit() for ch in rest) and "-" in rest:
            return None

        if len(parts) == 1 and not dot and not (styled or explicit_label):
            return None
        if not (styled or explicit_label) and _ends_sentence(cleaned):
            return None
        return len(parts)

    return None


def _title_starts_well(rest: str) -> bool:
    stripped = rest.lstrip("\"'([")
    if not stripped:
        return False
    first = stripped[0]
    return first.isupper() or (not first.isascii() and first.isalpha())


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


def _assign_heading_levels(
    headings: list[tuple[dict[str, Any], int | None]],
    slide_titles: set[str] = frozenset(),
    size_tolerance: float = _DEFAULT_SIZE_TOLERANCE,
) -> dict[str, int]:
    slide_pages = {elem.get("page") for elem, _ in headings if elem["id"] in slide_titles}
    style_counts: dict[tuple[float, bool], int] = {}
    for elem, explicit_depth in headings:
        if (
            explicit_depth is None
            and elem["id"] not in slide_titles
            and (elem.get("metadata") or {}).get("layout_label") != "title"
        ):
            key = _style_key(elem)
            if key is not None:
                style_counts[key] = style_counts.get(key, 0) + 1
    styles = sorted(style_counts, reverse=True)
    style_rank = _cluster_styles(styles, style_counts, _MAX_HEADING_LEVEL - int(bool(slide_pages)), size_tolerance)
    has_title = any((elem.get("metadata") or {}).get("layout_label") == "title" for elem, _ in headings)

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
            levels[elem["id"]] = style_rank[key] + int(elem.get("page") in slide_pages)
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
    inplace: bool = False,
) -> dict[str, Any]:
    out = ir if inplace else copy.deepcopy(ir)
    elements: list[dict[str, Any]] = out.get("elements", [])
    doc_meta = out.setdefault("metadata", {})

    if not elements:
        doc_meta["outline"] = []
        return out

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
    toc_matches, toc_pages = _match_toc_entries(ordered, resolved_toc) if resolved_toc else ({}, set())

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
    heading_candidates = collect_headings()
    heading_levels = _assign_heading_levels(
        heading_candidates,
        _slide_title_ids(heading_candidates, landscape_heights),
        style_size_tolerance,
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
