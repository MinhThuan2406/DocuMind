from __future__ import annotations

import copy
import re
import unicodedata
from collections.abc import Collection, Mapping, Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ._pdf_io import extract_toc, round_bbox

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
_SOFT_MERGE_LABELS = frozenset({"code", "formula", "footnote"})
_SOFT_MERGE_PROTECTED_ROLES = frozenset({"heading", "caption", "page_header", "page_footer"})

_CAPTION_PREFIX_RE = re.compile(
    r"^(?P<kind>(?i:figure|fig\.?|diagram|diag\.?|chart|listing|abb\.?|abbildung|grafik|graph|plate|exhibit|"
    r"table|tab\.?|tbl\.?|tabelle))"
    r"(?:(?<=\.)\s*|\s+)"
    r"(?P<num>(?=[A-Za-z0-9.\-]*\d)[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*|"
    r"(?:[IVXLCDM]+|[A-Z])(?=\s*[:.\-\u2013\u2014]))"
    r"(?:\s*(?P<sep>[:.\-\u2013\u2014])\s*(?P<rest>.*)|\s+(?P<bare>.*))?$"
)
_TABLE_CAPTION_KINDS = frozenset({"table", "tab", "tbl", "tabelle"})
_CAPTION_PROSE_START_RE = re.compile(
    r"^(?:shows?|illustrates?|depicts?|presents?|lists?|contains?|describes?|summarizes?|summarises?|"
    r"gives?|provides?|demonstrates?|compares?|displays?|exhibits?|indicates?|explains?|highlights?|"
    r"represents?|outlines?|is|are|was|were|can|will|has|have|in|on|above|below|"
    r"zeigt|stellt|beschreibt|gibt|enthält|ist|sind)\b",
    re.IGNORECASE,
)

_STRUCT_EN_TIERS = {
    "part": 0,
    "title": 0,
    "book": 0,
    "chapter": 1,
    "division": 1,
    "section": 2,
    "subpart": 2,
    "subsection": 3,
    "article": 4,
    "rule": 4,
    "clause": 4,
    "schedule": 4,
}
_STRUCT_CJK_TIERS = {"編": 0, "章": 1, "節": 2, "款": 3, "条": 4}
_STRUCT_EN_RE = re.compile(
    r"^(part|title|book|chapter|division|section|subpart|subsection|article|rule|clause|schedule)"
    r"\s+(?-i:([0-9]+(?:\.[0-9]+)*|[IVXLCDM]+|[A-Z](?=\s*[:.\-\s]|$)))(?:\b|(?=\s|$))[:.\-\s]*(.*)$",
    re.IGNORECASE,
)
_STRUCT_CJK_RE = re.compile(r"^(第\s*[0-9一二三四五六七八九十百千]+\s*(編|章|節|款|条))(.*)$")
_STRUCT_PROSE_START_RE = re.compile(
    r"^(?:specifies|shall|is|are|was|were|requires|defines|provides|states|applies|"
    r"may|must|can|will|has|have|of|to|in|for|and|or)\b",
    re.IGNORECASE,
)
_CJK_PARTICLES = ("の", "に", "を", "は", "が", "で", "と", "へ", "も", "より", "から", "まで")
_SINGLE_LETTER_APP_RE = re.compile(r"^([A-Z])\s+([A-Z].*)$")
_NATIVE_LIST_ITEM_RE = re.compile(
    r"^(?:[•●○◦▪▫‣⁃・･■□◆◇★☆△▽▲▼▶▷▸▹∙※✓✔➢➣➤]\s*\S|"
    r"[-–—*+]\s+\S|\(\d{1,3}\)\s+\S|\d{1,3}[.)]\s+\S)"
)
_APPENDIX_RE = re.compile(r"^appendix\s+([A-Z])(?:\.([0-9]+(?:\.[0-9]+)*))?\b[:.\-\s]*(.*)$", re.IGNORECASE)
_APPENDIX_SUB_RE = re.compile(r"^([A-Z]\.[0-9]+(?:\.[0-9]+)*)\.?\s+(.+)$")
_NUMERIC_HEADING_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)*)(?:(\.)\s*|\s+)(.+)$")
_LEADING_NUMBER_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)*)(?:(\.)\s*|\s+)(\S.*)$")
_SOFT_HYPHEN_WRAP_RE = re.compile(r"([A-Za-z])-\s*\n\s*([A-Za-z])")
_REQ_ID_RE = re.compile(r"^\[[A-Za-z0-9_.:-]{2,}\]")
_DECORATIVE_RUN_RE = re.compile(r"[.\-_=•·\u2026\u30fb─━]{4,}")
_UML_MULTIPLICITY_RE = re.compile(r"\b\d+\.\.(?:\d+\b|\*)")
_BULLET_RE = re.compile(r"^[•\-*◦▪▫●○■□◆◇★☆△▽▲▼▶▷▸▹‣⁃∙・※✓✔➢➣➤\u2022\u25e6\u25aa\u25b3\u25bd\u2013\u2014]")
_MULTI_LIST_LINE_RE = re.compile(
    r"^(?:\([0-9]+\)\s*\S|[0-9]+[.)]\s+\S|[0-9]+[.)](?=[^\d\s.])\S|"
    r"[•\-*◦▪▫●○■□◆◇★☆△▽▲▼▶▷▸▹‣⁃∙・※✓✔➢➣➤\u2022\u25e6\u25aa\u25b3\u25bd\u2013\u2014]\s*\S)"
)
_TOC_LEADER_RE = re.compile(
    r"^(?P<title>.+?)\s*(?:[.\u00b7\u2022\u2026\u30fb\-_─━]{3,}|\t|\s{2,})\s*(?P<pno>\d{1,4})\s*$"
)
_CJK_CHAR_CLASS = r"\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_CJK_GAP_RE = re.compile(rf"(?<=[{_CJK_CHAR_CLASS}])\s+(?=[{_CJK_CHAR_CLASS}])")
_DEFAULT_EM = 10.0
_MAX_HEADING_LEVEL = 6
_MIN_STYLE_TIER_COUNT = 5
_SLIDE_TITLE_BAND = 0.2
_TITLE_BAND_MAX_WORDS = 24
_DEFAULT_SIZE_TOLERANCE = 0.05
_SLIDE_BOOKMARK_RE = re.compile(r"^(?:foli[ea]|slide|page)\s+\d+(?:\b|:)", re.IGNORECASE)
_TOC_HEADER_RE = re.compile(r"^(?:table\s+of\s+contents|contents|目次|もくじ)$", re.IGNORECASE)


def _heading_text(elem: Mapping[str, Any]) -> str:
    meta = elem.get("metadata") or {}
    clean = meta.get("heading_clean_text")
    if isinstance(clean, str) and clean.strip():
        return clean
    return str(elem.get("text") or "")


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _em_size(body_font: float | None) -> float:
    return float(body_font) if (body_font is not None and body_font > 0) else _DEFAULT_EM


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
    normed = _nfkc(dehyphenated).casefold()
    chars = [ch if unicodedata.category(ch)[0] in ("L", "N", "M") else " " for ch in normed]
    collapsed = " ".join("".join(chars).split())
    return _CJK_GAP_RE.sub("", collapsed)


def _is_matchable_toc_title(norm: str) -> bool:
    compact = norm.replace(" ", "")
    return len(compact) >= 2 and any(ch.isalpha() for ch in compact)


def _filter_meaningful_toc(toc: Sequence[tuple[int, str, int]]) -> list[tuple[int, str, int]]:
    cleaned = [(int(lvl), _clean_text(str(title)), int(page)) for lvl, title, page in toc if int(page) >= 1]
    if not cleaned:
        return []
    slide_like = sum(1 for _, title, _ in cleaned if _SLIDE_BOOKMARK_RE.match(title))
    return [] if slide_like >= 0.8 * len(cleaned) else cleaned


def _load_pdf_toc(pdf_path: str | Path) -> list[tuple[int, str, int]]:
    return _filter_meaningful_toc(extract_toc(pdf_path))


def _is_toc_leader_line(line: str, max_page_num: int) -> bool:
    norm = _nfkc(line).strip()
    if not norm:
        return False
    m = _TOC_LEADER_RE.match(norm)
    if not m:
        return False
    title = m.group("title").strip()
    pno = int(m.group("pno"))
    return len(title) >= 2 and _has_alpha(title) and 1 <= pno <= max_page_num


def _split_toc_pair_ids(page_elems: Sequence[dict[str, Any]], em: float, max_page_num: int) -> tuple[int, set[str]]:
    num_elems: list[tuple[dict[str, Any], float, float]] = []
    text_elems: list[tuple[dict[str, Any], float]] = []
    for elem in page_elems:
        bbox = elem.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        cleaned = _clean_text(_nfkc(elem.get("text", "")))
        if not cleaned:
            continue
        cy = (float(bbox[1]) + float(bbox[3])) * 0.5
        if cleaned.isdigit() and 1 <= int(cleaned) <= max_page_num:
            num_elems.append((elem, cy, float(bbox[2])))
        elif len(cleaned) >= 2 and _has_alpha(cleaned):
            text_elems.append((elem, cy))
    if len(num_elems) < 3 or not text_elems:
        return 0, set()
    matched_pairs: list[tuple[float, str, str]] = []
    for r_elem, r_cy, r_x1 in num_elems:
        rx0 = float(r_elem["bbox"][0])
        l_match = next(
            (
                l_elem
                for l_elem, l_cy in text_elems
                if abs(l_cy - r_cy) <= 0.6 * em and float(l_elem["bbox"][2]) <= rx0 + 0.2 * em
            ),
            None,
        )
        if l_match is not None:
            matched_pairs.append((r_x1, str(l_match["id"]), str(r_elem["id"])))
    if len(matched_pairs) < 3:
        return 0, set()
    rights = sorted(item[0] for item in matched_pairs)
    median_x1 = rights[len(rights) // 2]
    aligned = [item for item in matched_pairs if abs(item[0] - median_x1) <= 2.5 * em]
    ids = {eid for _, lid, rid in aligned for eid in (lid, rid)}
    return len(aligned), ids


def _count_split_toc_pairs(page_elems: Sequence[dict[str, Any]], em: float, max_page_num: int) -> int:
    count, _ = _split_toc_pair_ids(page_elems, em, max_page_num)
    return count


def _detect_toc_pages(
    ordered: list[tuple[int, dict[str, Any]]],
    toc: Sequence[tuple[int, str, int]],
    body_font: float | None = None,
) -> tuple[set[int], set[str]]:
    by_page_elems: dict[int, list[dict[str, Any]]] = {}
    for _, elem in ordered:
        if elem.get("type") != "text" or _is_ignored_element(elem):
            continue
        pno = int(elem.get("page", 0))
        if pno >= 1:
            by_page_elems.setdefault(pno, []).append(elem)
    if not by_page_elems:
        return set(), set()

    max_page = max(by_page_elems)
    max_page_num = max(max_page + 20, 300)
    em = _em_size(body_font)
    first_target_page = min((page for _, _, page in toc), default=None)
    scan_limit = (
        max(first_target_page - 1, min(8, max(2, max_page // 3)))
        if first_target_page is not None
        else min(8, max(2, max_page // 3))
    )

    bm_entries: list[tuple[int, str, int]] = []
    for idx, (_, title, target_page) in enumerate(toc):
        norm = _norm_title(title)
        if _is_matchable_toc_title(norm):
            bm_entries.append((idx, norm, int(target_page)))

    toc_pages: set[int] = set()
    toc_elem_ids: set[str] = set()
    in_header_toc = False
    for pno in sorted(by_page_elems):
        if pno > scan_limit:
            break
        p_elems = by_page_elems[pno]
        is_toc_candidate = False
        anchor_ids: set[str] = set()

        if first_target_page is not None and pno < first_target_page:
            if any(_TOC_HEADER_RE.match(_clean_text(_nfkc(e.get("text", "")))) for e in p_elems):
                in_header_toc = True
            if in_header_toc:
                is_toc_candidate = True

        leader_lines = 0
        for e in p_elems:
            if any(_is_toc_leader_line(ln, max_page_num) for ln in e.get("text", "").splitlines()):
                leader_lines += sum(1 for ln in e.get("text", "").splitlines() if _is_toc_leader_line(ln, max_page_num))
                anchor_ids.add(e["id"])

        split_pairs, split_ids = _split_toc_pair_ids(p_elems, em, max_page_num)
        anchor_ids.update(split_ids)

        bm_hits: set[int] = set()
        if bm_entries:
            line_norm_map: dict[str, list[str]] = {}
            for e in p_elems:
                for ln in e.get("text", "").splitlines():
                    n = _norm_title(ln)
                    if _is_matchable_toc_title(n):
                        line_norm_map.setdefault(n, []).append(e["id"])
            for idx, b_norm, target_p in bm_entries:
                if target_p > pno and b_norm in line_norm_map:
                    bm_hits.add(idx)
                    anchor_ids.update(line_norm_map[b_norm])

        if leader_lines >= 3 or split_pairs >= 3 or len(bm_hits) >= 3:
            is_toc_candidate = True
            if first_target_page is not None and pno < first_target_page:
                in_header_toc = True
        elif not toc and (pno - 1) in toc_pages and (leader_lines >= 1 or split_pairs >= 1):
            is_toc_candidate = True

        if not is_toc_candidate:
            continue

        anchor_boxes = [
            e["bbox"]
            for e in p_elems
            if e["id"] in anchor_ids and isinstance(e.get("bbox"), (list, tuple)) and len(e["bbox"]) == 4
        ]
        if not anchor_boxes:
            toc_pages.add(pno)
            continue

        anchor_y0 = min(float(b[1]) for b in anchor_boxes)
        anchor_y1 = max(float(b[3]) for b in anchor_boxes)

        # Check whether this page mixes a ToC band with genuine body/heading content outside the band.
        outside_elems: list[dict[str, Any]] = []
        has_same_page_bm = False
        for e in p_elems:
            if e["id"] in anchor_ids:
                continue
            bbox = e.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            cy = 0.5 * (float(bbox[1]) + float(bbox[3]))
            if anchor_y0 - 0.5 * em <= cy <= anchor_y1 + 0.5 * em:
                continue
            cleaned = _clean_text(_nfkc(_heading_text(e)))
            if not cleaned or _TOC_HEADER_RE.match(cleaned):
                continue
            if bm_entries:
                e_norm = _norm_title(cleaned)
                if any(target_p == pno and b_norm == e_norm for _, b_norm, target_p in bm_entries):
                    has_same_page_bm = True
            if _has_alpha(cleaned) and (
                len(cleaned) >= 30 or _sentence_count(cleaned) >= 1 or _is_heading_candidate(e, body_font)[0]
            ):
                outside_elems.append(e)

        is_mixed = has_same_page_bm or len(outside_elems) >= 2
        if not is_mixed:
            toc_pages.add(pno)
            continue

        toc_band_y0 = anchor_y0
        toc_band_y1 = anchor_y1
        for e in p_elems:
            bbox = e.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            ey0, ey1 = float(bbox[1]), float(bbox[3])
            if 0.0 <= anchor_y0 - ey1 <= 4.0 * em:
                cleaned = _clean_text(_nfkc(_heading_text(e)))
                if _TOC_HEADER_RE.match(cleaned):
                    toc_band_y0 = min(toc_band_y0, ey0)

        for e in p_elems:
            if e["id"] in anchor_ids:
                toc_elem_ids.add(e["id"])
                continue
            bbox = e.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                cy = 0.5 * (float(bbox[1]) + float(bbox[3]))
                if toc_band_y0 - 0.2 * em <= cy <= toc_band_y1 + 0.2 * em:
                    toc_elem_ids.add(e["id"])

    return toc_pages, toc_elem_ids


def _match_toc_entries(
    ordered: list[tuple[int, dict[str, Any]]],
    toc: Sequence[tuple[int, str, int]],
    body_font: float | None = None,
) -> tuple[dict[str, int], set[int], set[str]]:
    toc_pages, toc_elem_ids = _detect_toc_pages(ordered, toc, body_font)
    if not toc:
        return {}, toc_pages, toc_elem_ids

    by_page: dict[int, list[tuple[int, str, str]]] = {}
    exact_idx: dict[tuple[int, str], list[tuple[int, str]]] = {}
    compact_idx: dict[tuple[int, str], list[tuple[int, str]]] = {}
    for idx, (lvl, title, page) in enumerate(toc):
        norm = _norm_title(title)
        if not _is_matchable_toc_title(norm):
            continue
        uid = f"{idx}:{norm}"
        by_page.setdefault(page, []).append((lvl, norm, uid))
        exact_idx.setdefault((page, norm), []).append((lvl, uid))
        if any(ord(ch) >= 0x3000 for ch in norm):
            compact_idx.setdefault((page, norm.replace(" ", "")), []).append((lvl, uid))

    matched_levels: dict[str, int] = {}
    used_toc: set[str] = set()
    for _, elem in ordered:
        if elem.get("type") != "text" or _is_ignored_element(elem):
            continue
        page = int(elem.get("page", 0))
        if page in toc_pages or elem["id"] in toc_elem_ids:
            continue
        norm_elem = _norm_title(_heading_text(elem))
        if not _is_matchable_toc_title(norm_elem) or len(norm_elem) > 160:
            continue

        exact_hit: tuple[int, str] | None = None
        compact_elem = norm_elem.replace(" ", "") if any(ord(ch) >= 0x3000 for ch in norm_elem) else ""
        for cand_page in (page, page - 1, page + 1):
            for lvl, uid in exact_idx.get((cand_page, norm_elem), ()):
                if uid not in used_toc:
                    exact_hit = (lvl, uid)
                    break
            if exact_hit is None and compact_elem:
                for lvl, uid in compact_idx.get((cand_page, compact_elem), ()):
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

    return matched_levels, toc_pages, toc_elem_ids


def _is_ignored_element(elem: dict[str, Any]) -> bool:
    if elem.get("level", 1) == 2:
        return True
    meta = elem.get("metadata") or {}
    return meta.get("role") in _IGNORED_ROLES or meta.get("layout_label") in _IGNORED_ROLES


def _has_alpha(text: str) -> bool:
    return any(ch.isalpha() for ch in _nfkc(text))


def _sentence_count(text: str) -> int:
    return len(re.findall(r"(?:[.!?](?:\s|$)|[。！？])", _nfkc(text)))


def _ends_sentence(text: str) -> bool:
    norm = _nfkc(text).rstrip()
    return bool(norm) and norm[-1] in ".?!;,。！？；、"


def _effective_word_count(text: str) -> float:
    total = 0.0
    for tok in _nfkc(text).split():
        cjk = 0
        sea = 0
        has_other = False
        for ch in tok:
            o = ord(ch)
            if (
                0x3040 <= o <= 0x30FF
                or 0x31F0 <= o <= 0x31FF
                or 0x3400 <= o <= 0x4DBF
                or 0x4E00 <= o <= 0x9FFF
                or 0xF900 <= o <= 0xFAFF
            ):
                cjk += 1
            elif (
                0x0E00 <= o <= 0x0EFF
                or 0x0F00 <= o <= 0x0FFF
                or 0x1000 <= o <= 0x109F
                or 0x1780 <= o <= 0x17FF
            ):
                if not unicodedata.category(ch).startswith("M"):
                    sea += 1
            elif unicodedata.category(ch)[0] in ("L", "N"):
                has_other = True
        total += cjk * 0.4 + sea * 0.25 + (1.0 if has_other or (cjk == 0 and sea == 0) else 0.0)
    return total


def _is_list_block(raw: str) -> bool:
    stripped = raw.strip()
    if not stripped:
        return False
    if _BULLET_RE.match(_nfkc(stripped)):
        return True
    list_lines = 0
    for ln in raw.splitlines():
        s_raw = ln.lstrip()
        if not s_raw:
            continue
        if unicodedata.category(s_raw[0]) == "No" and len(s_raw) > 1:
            list_lines += 1
        elif _MULTI_LIST_LINE_RE.match(_nfkc(s_raw)):
            list_lines += 1
        if list_lines >= 2:
            return True
    return False


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


def _region_fields(reg: Any) -> tuple[Sequence[float] | None, str | None, float]:
    if isinstance(reg, Mapping):
        rb = reg.get("bbox")
        label = reg.get("label")
        conf = float(reg.get("confidence", 1.0))
    else:
        rb = getattr(reg, "bbox", None)
        label = getattr(reg, "label", None)
        conf = float(getattr(reg, "confidence", 1.0))
    return rb, str(label) if label else None, conf


def _project_semantic_regions(
    elements: Sequence[dict[str, Any]],
    semantic_regions: Mapping[int, Sequence[Any]] | None,
    *,
    page_heights: Mapping[int, float] | None = None,
    body_font: float | None = None,
) -> None:
    if not semantic_regions:
        return
    for elem in elements:
        if elem.get("type") != "text" or elem.get("level", 1) != 1:
            continue
        source = elem.get("source", "text_layer")
        if source != "text_layer":
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
        pno = elem.get("page", 0)
        regions = semantic_regions.get(pno)
        if not regions:
            continue
        page_h = float(page_heights.get(pno, 0.0)) if page_heights else 0.0
        cy = 0.5 * (ey0 + ey1)
        fs = meta.get("font_size")
        best_score: tuple[float, float] | None = None
        best_label: str | None = None
        for reg in regions:
            rb, lbl_str, conf = _region_fields(reg)
            if not rb or len(rb) != 4 or not lbl_str:
                continue
            if lbl_str.lower() in _IGNORED_ROLES and page_h > 0.0:
                in_margin = cy <= 0.12 * page_h or cy >= 0.88 * page_h
                fs_ok = not (
                    body_font is not None
                    and body_font > 0.0
                    and isinstance(fs, (int, float))
                    and fs > 0.0
                    and float(fs) > body_font * 1.05
                )
                if not (in_margin and fs_ok):
                    continue
            ix = max(0.0, min(ex1, float(rb[2])) - max(ex0, float(rb[0])))
            iy = max(0.0, min(ey1, float(rb[3])) - max(ey0, float(rb[1])))
            ioa = (ix * iy) / elem_area
            if ioa >= 0.60:
                score = (ioa, conf)
                if best_score is None or score > best_score:
                    best_score = score
                    best_label = lbl_str
        if best_label is not None:
            meta["layout_label"] = best_label


def _join_soft_merged_text(run: Sequence[dict[str, Any]], label: str) -> str:
    if label == "code":
        return "\n".join(t for e in run if (t := str(e.get("text") or "").rstrip("\r\n")))
    if label == "footnote":
        acc = ""
        for e in run:
            t = str(e.get("text") or "").strip()
            if t:
                acc = _join_wrapped_text(acc, t) if acc else t
        return acc
    parts: list[str] = []
    prev_b: Sequence[float] | None = None
    for e in run:
        t = str(e.get("text") or "").strip()
        if not t:
            continue
        cur_b = e.get("bbox")
        if not parts:
            parts.append(t)
        elif prev_b is not None and cur_b is not None and len(prev_b) == 4 and len(cur_b) == 4:
            h1 = max(1.0, float(prev_b[3]) - float(prev_b[1]))
            h2 = max(1.0, float(cur_b[3]) - float(cur_b[1]))
            min_h = min(h1, h2)
            cy1 = 0.5 * (float(prev_b[1]) + float(prev_b[3]))
            cy2 = 0.5 * (float(cur_b[1]) + float(cur_b[3]))
            stacked = float(cur_b[1]) >= float(prev_b[3]) - 0.25 * min_h or abs(cy2 - cy1) > 0.5 * min_h
            parts.append(("\n" if stacked else " ") + t)
        else:
            parts.append(" " + t)
        prev_b = cur_b
    return "".join(parts)


def soft_merge_semantic_elements(
    elements: list[dict[str, Any]],
    semantic_regions: Mapping[int, Sequence[Any]] | None,
    *,
    labels: Collection[str] = _SOFT_MERGE_LABELS,
    min_ioa: float = 0.50,
) -> list[dict[str, Any]]:
    if not elements or not semantic_regions:
        return elements
    allowed = frozenset(str(lbl).lower() for lbl in labels) & _SOFT_MERGE_LABELS
    if not allowed:
        return elements

    ordered = sorted(
        enumerate(elements),
        key=lambda pair: (pair[1].get("page", 0), pair[1].get("reading_order", pair[0]), pair[0]),
    )

    def _match_key(elem: dict[str, Any]) -> tuple[int, int, str] | None:
        if elem.get("type") != "text" or elem.get("level", 1) != 1:
            return None
        if elem.get("source", "text_layer") != "text_layer":
            return None
        meta = elem.get("metadata") or {}
        if meta.get("role") in _SOFT_MERGE_PROTECTED_ROLES:
            return None
        if meta.get("table_id") or meta.get("diagram_id") or meta.get("formula_id"):
            return None
        bbox = elem.get("bbox")
        if not bbox or len(bbox) != 4:
            return None
        ex0, ey0, ex1, ey1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        elem_area = max(0.0, ex1 - ex0) * max(0.0, ey1 - ey0)
        if elem_area <= 0.0:
            return None
        pno = elem.get("page", 0)
        regions = semantic_regions.get(pno)
        if not regions:
            return None
        best_score: tuple[float, float] | None = None
        best_idx = -1
        best_lbl = ""
        for r_idx, reg in enumerate(regions):
            rb, lbl_str, conf = _region_fields(reg)
            if not rb or len(rb) != 4 or not lbl_str:
                continue
            ix = max(0.0, min(ex1, float(rb[2])) - max(ex0, float(rb[0])))
            iy = max(0.0, min(ey1, float(rb[3])) - max(ey0, float(rb[1])))
            ioa = (ix * iy) / elem_area
            if ioa >= min_ioa:
                score = (ioa, conf)
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = r_idx
                    best_lbl = lbl_str.lower()
        if best_idx < 0 or best_lbl not in allowed:
            return None
        cur_lbl = meta.get("layout_label")
        if cur_lbl is not None and str(cur_lbl).lower() != best_lbl:
            return None
        return (int(pno), best_idx, best_lbl)

    dropped_ptrs: set[int] = set()
    cur_key: tuple[int, int, str] | None = None
    cur_run: list[dict[str, Any]] = []

    def _flush_run() -> None:
        if cur_key is None or len(cur_run) < 2:
            return
        primary = cur_run[0]
        target_lbl = cur_key[2]
        primary["text"] = _join_soft_merged_text(cur_run, target_lbl)
        boxes = [e["bbox"] for e in cur_run if e.get("bbox") and len(e["bbox"]) == 4]
        if boxes:
            merged_box = [
                v + 0.0
                for v in round_bbox((
                    min(float(b[0]) for b in boxes),
                    min(float(b[1]) for b in boxes),
                    max(float(b[2]) for b in boxes),
                    max(float(b[3]) for b in boxes),
                ))
            ]
            primary["bbox"] = merged_box
            src_ref = primary.get("source_ref")
            if isinstance(src_ref, dict) and "bbox" in src_ref:
                src_ref["bbox"] = list(merged_box)
        primary["confidence"] = round(min(float(e.get("confidence", 1.0)) for e in cur_run), 4)
        pmeta = primary.setdefault("metadata", {})
        pmeta["layout_label"] = target_lbl
        for extra in cur_run[1:]:
            dropped_ptrs.add(id(extra))

    for _, elem in ordered:
        key = _match_key(elem)
        if key is not None and key == cur_key:
            cur_run.append(elem)
        else:
            _flush_run()
            cur_key = key
            cur_run = [elem] if key is not None else []
    _flush_run()

    if dropped_ptrs:
        elements[:] = [e for e in elements if id(e) not in dropped_ptrs]
        page_ro: dict[Any, int] = {}
        for _, elem in ordered:
            if id(elem) in dropped_ptrs:
                continue
            page = elem.get("page", 0)
            ro = page_ro.get(page, 0)
            elem["reading_order"] = ro
            page_ro[page] = ro + 1
    return elements


def _should_trust_section_header(elem: dict[str, Any], body_font: float | None) -> bool:
    if elem.get("source", "text_layer") != "text_layer" or body_font is None:
        return True
    fs = (elem.get("metadata") or {}).get("font_size")
    if not isinstance(fs, (int, float)) or fs <= 0:
        return True
    return float(fs) >= body_font * 0.95


def _allow_caption_check_on_label(label: Any, elem: dict[str, Any], body_font: float | None) -> bool:
    if label == "caption":
        return True
    if label == "title" or label in _NON_HEADING_LABELS:
        return False
    if label == "section_header":
        if body_font is None:
            return False
        fs = (elem.get("metadata") or {}).get("font_size")
        return isinstance(fs, (int, float)) and 0.0 < float(fs) <= body_font
    return True


def _extract_short_prefix_and_num(cleaned_nfkc: str) -> tuple[str, str | None, str] | None:
    n = len(cleaned_nfkc)
    i = 0
    letter_count = 0
    while i < n:
        cat = unicodedata.category(cleaned_nfkc[i])
        if cat[0] == "L":
            letter_count += 1
            i += 1
        elif cat[0] == "M" and letter_count > 0:
            i += 1
        else:
            break
    if letter_count == 0 or letter_count > 15:
        return None
    if i < n and cleaned_nfkc[i] == ".":
        i += 1
    prefix_key = cleaned_nfkc[:i].rstrip(".").casefold()
    while i < n and cleaned_nfkc[i].isspace():
        i += 1
    num_start = i
    while i < n and (cleaned_nfkc[i].isdigit() or (cleaned_nfkc[i] in ".-" and i + 1 < n and cleaned_nfkc[i + 1].isdigit())):
        i += 1
    if i == num_start:
        return None
    if i == n:
        return prefix_key, None, ""
    ch = cleaned_nfkc[i]
    if ch in ":.-–—、)）":
        return prefix_key, ch, cleaned_nfkc[i + 1 :].strip()
    if ch.isspace():
        rest = cleaned_nfkc[i:].strip()
        if rest and rest[0] in ":.-–—、)）":
            return prefix_key, rest[0], rest[1:].strip()
        return prefix_key, None, rest
    return None


def _learn_caption_prefixes(
    elements: Sequence[dict[str, Any]],
    body_font: float | None,
    protected_ids: set[str] | frozenset[str] = frozenset(),
) -> dict[str, str | None]:
    em = _em_size(body_font)
    adj_targets: dict[str, set[str]] = {}
    adj_kinds: dict[str, list[str]] = {}
    adj_elems: dict[str, set[str]] = {}
    non_adj_count: dict[str, int] = {}

    pages: dict[int, list[dict[str, Any]]] = {}
    for elem in elements:
        pages.setdefault(elem.get("page", 0), []).append(elem)

    for page_elems in pages.values():
        targets = [e for e in page_elems if e.get("level", 1) == 1 and e.get("type") in {"figure", "table"}]
        for elem in page_elems:
            if elem.get("type") != "text" or _is_ignored_element(elem) or elem.get("id") in protected_ids:
                continue
            meta = elem.get("metadata") or {}
            if not _allow_caption_check_on_label(meta.get("layout_label"), elem, body_font):
                continue
            raw = elem.get("text", "")
            if _is_list_block(raw):
                continue
            cleaned = _clean_text(_nfkc(raw))
            if (
                not cleaned
                or not _has_alpha(cleaned)
                or len(cleaned) > 140
                or _effective_word_count(cleaned) > 18
                or _sentence_count(cleaned) >= 2
                or _parse_caption_prefix(cleaned)[0]
                or _parse_numbered_heading(cleaned, meta, body_font) is not None
                or _parse_structural_keyword(cleaned, meta, body_font) is not None
            ):
                continue
            fs = meta.get("font_size")
            if body_font and isinstance(fs, (int, float)) and fs >= body_font * 1.15:
                continue
            parsed = _extract_short_prefix_and_num(cleaned)
            if parsed is None:
                continue
            prefix_key, sep, rest = parsed
            if sep is None and rest and _CAPTION_PROSE_START_RE.match(rest):
                continue
            bbox = elem.get("bbox")
            best_tgt: tuple[float, float, dict[str, Any]] | None = None
            if bbox and len(bbox) == 4:
                for tgt in targets:
                    tb = tgt.get("bbox")
                    if not tb or len(tb) != 4:
                        continue
                    dy, h_ov = _bbox_metrics(bbox, tb, em=em)
                    if dy <= 2.2 * em and h_ov > 0.0:
                        cand = (dy, -h_ov, tgt)
                        if best_tgt is None or cand < (best_tgt[0], best_tgt[1], None):
                            best_tgt = cand
            if best_tgt is not None:
                tgt = best_tgt[2]
                adj_targets.setdefault(prefix_key, set()).add(tgt["id"])
                adj_kinds.setdefault(prefix_key, []).append(str(tgt.get("type")))
                adj_elems.setdefault(prefix_key, set()).add(elem["id"])
            else:
                non_adj_count[prefix_key] = non_adj_count.get(prefix_key, 0) + 1

    learned: dict[str, str | None] = {}
    for prefix_key, tgt_ids in adj_targets.items():
        if len(tgt_ids) >= 2 and len(adj_elems[prefix_key]) >= non_adj_count.get(prefix_key, 0):
            kinds = set(adj_kinds[prefix_key])
            learned[prefix_key] = next(iter(kinds)) if len(kinds) == 1 else None
    return learned


def _parse_caption_prefix(
    text: str,
    learned_prefixes: Mapping[str, str | None] | None = None,
) -> tuple[bool, str | None]:
    cleaned = _clean_text(_nfkc(text))
    if not cleaned or len(cleaned) > 300:
        return False, None
    m = _CAPTION_PREFIX_RE.match(cleaned)
    if m:
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

    if learned_prefixes:
        parsed = _extract_short_prefix_and_num(cleaned)
        if parsed is not None and parsed[0] in learned_prefixes:
            prefix_key, sep, rest = parsed
            if sep is None and rest and (
                _CAPTION_PROSE_START_RE.match(rest) or len(cleaned) > 140 or _sentence_count(cleaned) >= 2
            ):
                return False, None
            return True, learned_prefixes[prefix_key]
    return False, None


def _bbox_metrics(a: Sequence[float], b: Sequence[float], *, em: float = _DEFAULT_EM) -> tuple[float, float]:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    h_overlap = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    acy = (ay0 + ay1) * 0.5
    bcy = (by0 + by1) * 0.5
    if acy <= bcy and ay1 <= by0 + 0.8 * em:
        dy = max(0.0, by0 - ay1)
    elif acy > bcy and ay0 >= by1 - 0.8 * em:
        dy = max(0.0, ay0 - by1)
    else:
        dy = min(abs(by0 - ay1), abs(ay0 - by1)) + 5.0 * em
    return dy, h_overlap


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


def _drop_merged_elements(
    elements: list[dict[str, Any]],
    ordered: list[tuple[int, dict[str, Any]]],
    dropped_ids: set[str],
) -> list[tuple[int, dict[str, Any]]]:
    if not dropped_ids:
        return ordered
    elements[:] = [e for e in elements if e.get("id") not in dropped_ids]
    remaining = [pair for pair in ordered if pair[1].get("id") not in dropped_ids]
    page_ro: dict[Any, int] = {}
    for _, elem in remaining:
        page = elem.get("page", 0)
        ro = page_ro.get(page, 0)
        elem["reading_order"] = ro
        page_ro[page] = ro + 1
    return remaining


def _vertically_adjacent(
    b1: Sequence[float] | None,
    b2: Sequence[float] | None,
    *,
    gap_scale: float,
    max_cap: float,
    align_tol: float,
    em: float = _DEFAULT_EM,
) -> bool:
    if not b1 or not b2 or len(b1) != 4 or len(b2) != 4:
        return False
    h1 = max(1.0, float(b1[3]) - float(b1[1]))
    h2 = max(1.0, float(b2[3]) - float(b2[1]))
    dy = float(b2[1]) - float(b1[3])
    max_gap = max(0.6 * em, min(max_cap, gap_scale * min(h1, h2)))
    if dy < -0.3 * em or dy > max_gap:
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
    if not _allow_caption_check_on_label(meta.get("layout_label"), elem, body_font):
        return False
    raw = elem.get("text", "")
    if _is_list_block(raw):
        return False
    cleaned = _clean_text(_nfkc(raw))
    eff_words = _effective_word_count(cleaned)
    if not cleaned or not _has_alpha(cleaned) or len(cleaned) > 120 or eff_words > 14:
        return False
    if cleaned[-1] in ":;?" or _sentence_count(cleaned) >= 2 or (eff_words > 8 and _ends_sentence(cleaned)):
        return False
    if _parse_numbered_heading(cleaned, meta, body_font) is not None or _parse_structural_keyword(cleaned, meta, body_font) is not None:
        return False
    fs = meta.get("font_size")
    if body_font and isinstance(fs, (int, float)) and fs >= body_font * 1.15:
        return False
    bbox = elem.get("bbox")
    if not bbox or len(bbox) != 4:
        return False
    em = _em_size(body_font)
    for tgt in targets:
        tb = tgt.get("bbox")
        if not tb or len(tb) != 4:
            continue
        dy, h_ov = _bbox_metrics(bbox, tb, em=em)
        if dy <= 1.8 * em and h_ov > 0.0:
            return True
    return False


def _merge_multiline_captions(
    elements: list[dict[str, Any]],
    ordered: list[tuple[int, dict[str, Any]]],
    body_font: float | None,
    learned_prefixes: Mapping[str, str | None] | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    dropped_ids: set[str] = set()
    em = _em_size(body_font)
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
        is_cap1 = (
            l1 == "caption"
            or m1.get("role") == "caption"
            or (
                _allow_caption_check_on_label(l1, c1, body_font)
                and _parse_caption_prefix(c1.get("text", ""), learned_prefixes)[0]
            )
        )
        if not is_cap1 or not _allow_caption_check_on_label(l2, c2, body_font):
            i += 1
            continue

        raw2 = c2.get("text", "")
        if _is_list_block(raw2):
            i += 1
            continue
        t2 = _clean_text(_nfkc(raw2))
        if (
            not t2
            or not _has_alpha(t2)
            or len(t2) > 140
            or _effective_word_count(t2) > 20
            or _sentence_count(t2) >= 2
            or _parse_caption_prefix(t2, learned_prefixes)[0]
            or _parse_numbered_heading(t2, m2, body_font) is not None
            or _parse_structural_keyword(t2, m2, body_font) is not None
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

        if not _vertically_adjacent(
            c1.get("bbox"),
            c2.get("bbox"),
            gap_scale=0.6,
            max_cap=1.2 * em,
            align_tol=4.0 * em,
            em=em,
        ):
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

    return _drop_merged_elements(elements, ordered, dropped_ids)


def _link_captions(
    elements: list[dict[str, Any]],
    body_font: float | None,
    protected_ids: set[str] = frozenset(),
    learned_prefixes: Mapping[str, str | None] | None = None,
) -> None:
    if learned_prefixes is None:
        learned_prefixes = _learn_caption_prefixes(elements, body_font, protected_ids)
    em = _em_size(body_font)
    pages: dict[int, list[dict[str, Any]]] = {}
    for elem in elements:
        pages.setdefault(elem.get("page", 0), []).append(elem)

    for page_elems in pages.values():
        targets = [e for e in page_elems if e.get("level", 1) == 1 and e.get("type") in {"figure", "table"}]
        explicit_caps: list[tuple[dict[str, Any], str | None, float]] = []
        for elem in page_elems:
            if elem.get("type") != "text" or _is_ignored_element(elem) or elem.get("id") in protected_ids:
                continue
            meta = elem.setdefault("metadata", {})
            label = meta.get("layout_label")
            if not _allow_caption_check_on_label(label, elem, body_font):
                continue
            is_prefixed, pref_kind = _parse_caption_prefix(elem.get("text", ""), learned_prefixes)
            if label == "caption" or meta.get("role") == "caption" or is_prefixed:
                meta["role"] = "caption"
                explicit_caps.append((elem, pref_kind, 12.0 * em))

        claimed_caps: set[str] = set()
        claimed_targets: set[str] = set()
        _match_caption_pool(explicit_caps, targets, claimed_caps, claimed_targets, em=em)

        remaining_targets = [t for t in targets if t["id"] not in claimed_targets]
        if not remaining_targets:
            continue
        adj_caps: list[tuple[dict[str, Any], str | None, float]] = []
        for elem in page_elems:
            if elem["id"] in claimed_caps or elem["id"] in protected_ids:
                continue
            if _is_adjacent_caption_candidate(elem, remaining_targets, body_font):
                adj_caps.append((elem, None, 1.8 * em))
        _match_caption_pool(adj_caps, remaining_targets, claimed_caps, claimed_targets, em=em)


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
    *,
    em: float = _DEFAULT_EM,
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
            dy, h_ov = _bbox_metrics(cb, tb, em=em)
            if dy > max_gap or h_ov <= 0.0:
                continue
            min_w = max(1.0, min(cb[2] - cb[0], tb[2] - tb[0]))
            dx_center = abs(cb_cx - (tb[0] + tb[2]) * 0.5)
            cost = dy + 0.1 * dx_center - 5.0 * (h_ov / min_w)
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


def _parse_structural_keyword(
    cleaned: str,
    meta: Mapping[str, Any],
    body_font: float | None,
    *,
    source: str = "text_layer",
) -> tuple[int, int] | None:
    norm = _clean_text(_nfkc(cleaned))
    if not norm or len(norm) > 140:
        return None
    fs = meta.get("font_size")
    styled = bool(meta.get("bold")) or (
        body_font is not None and isinstance(fs, (int, float)) and fs >= body_font * 1.05
    )
    label = meta.get("layout_label")
    explicit_label = label == "title" or (
        label == "section_header" and _should_trust_section_header({"source": source, "metadata": meta}, body_font)
    )
    eff_words = _effective_word_count(norm)
    if eff_words > (16 if (styled or explicit_label) else 12):
        return None

    m_en = _STRUCT_EN_RE.match(norm)
    if m_en:
        tier = _STRUCT_EN_TIERS[m_en.group(1).lower()]
        num_tok = m_en.group(2)
        rest = m_en.group(3).strip()
        if not rest:
            if not (styled or explicit_label):
                return None
        else:
            if _STRUCT_PROSE_START_RE.match(rest):
                return None
            if _sentence_count(rest) >= 2 or not _has_alpha(rest) or not _title_starts_well(rest):
                return None
            if not (styled or explicit_label) and _ends_sentence(norm):
                return None
        parts_len = min(6, len(num_tok.split("."))) if num_tok[0].isdigit() else 1
        return tier, parts_len

    m_cjk = _STRUCT_CJK_RE.match(norm)
    if m_cjk:
        tier = _STRUCT_CJK_TIERS[m_cjk.group(2)]
        sub_tail = re.sub(r"^(?:の|-)\d+", "", m_cjk.group(3))
        if sub_tail.startswith(_CJK_PARTICLES):
            return None
        rest = sub_tail.lstrip(" :.\t-–—")
        if not rest:
            if not (styled or explicit_label):
                return None
        else:
            if _sentence_count(rest) >= 2 or not _has_alpha(rest) or not _title_starts_well(rest):
                return None
            if not (styled or explicit_label) and _ends_sentence(norm):
                return None
        return tier, 1

    return None


def _parse_numbered_heading(
    cleaned: str,
    meta: dict[str, Any],
    body_font: float | None,
    *,
    source: str = "text_layer",
) -> int | None:
    norm = _clean_text(_nfkc(cleaned))
    if len(norm) > 140 or _effective_word_count(norm) > 18:
        return None

    fs = meta.get("font_size")
    styled = bool(meta.get("bold")) or (
        body_font is not None and isinstance(fs, (int, float)) and fs >= body_font * 1.05
    )
    label = meta.get("layout_label")
    explicit_label = label == "title" or (
        label == "section_header" and _should_trust_section_header({"source": source, "metadata": meta}, body_font)
    )

    m_app = _APPENDIX_RE.match(norm)
    if m_app:
        sub = m_app.group(2)
        rest = m_app.group(3).strip()
        if _sentence_count(rest) >= 2:
            return None
        if (rest and _has_alpha(rest) and _title_starts_well(rest)) or (not rest and (styled or explicit_label)):
            return 1 if not sub else min(6, 1 + len(sub.split(".")))

    if _sentence_count(norm) >= 2:
        return None

    m_asub = _APPENDIX_SUB_RE.match(norm)
    if m_asub:
        parts = m_asub.group(1).split(".")
        rest = m_asub.group(2).strip()
        if _has_alpha(rest) and _title_starts_well(rest):
            return min(6, len(parts))

    m_num = _NUMERIC_HEADING_RE.match(norm)
    if m_num:
        num_str, dot, rest = m_num.groups()
        parts = num_str.split(".")
        first = int(parts[0])
        if first > 99 or (first == 0 and len(parts) < 2) or len(parts) > 6:
            return None
        allow_camel = len(parts) >= 2 and (styled or explicit_label)
        if not _has_alpha(rest) or not _title_starts_well(rest, allow_camel=allow_camel):
            return None
        if len(rest.split()) == 1 and any(ch.isdigit() for ch in rest) and "-" in rest:
            return None

        if len(parts) == 1 and not dot and not (styled or explicit_label):
            return None
        if not (styled or explicit_label) and _ends_sentence(norm):
            return None
        return len(parts)

    return None


def _title_starts_well(rest: str, *, allow_camel: bool = False) -> bool:
    stripped = _nfkc(rest).lstrip("\"'([「『【〈《")
    if not stripped:
        return False
    first = stripped[0]
    if first.isupper() or (not first.isascii() and first.isalpha()):
        return True
    if allow_camel and first.islower():
        first_tok = stripped.split()[0]
        return any(ch.isupper() for ch in first_tok[1:])
    return False


def _is_native_list_item(raw: str) -> bool:
    s = _nfkc(raw).lstrip()
    if not s or s.startswith("["):
        return False
    if _NATIVE_LIST_ITEM_RE.match(s):
        return True
    return unicodedata.category(s[0]) == "No" and len(s) > 1


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

    raw = _heading_text(elem)
    if _is_list_block(raw):
        return False, None

    cleaned = _clean_text(raw)
    norm = _nfkc(cleaned)
    if not norm or not _has_alpha(norm) or len(norm) <= 1:
        return False, None

    src = str(elem.get("source", "text_layer"))
    num_depth = _parse_numbered_heading(cleaned, meta, body_font, source=src)
    struct_info = _parse_structural_keyword(cleaned, meta, body_font, source=src)

    if label == "title":
        return True, 1
    if label == "section_header" and _should_trust_section_header(elem, body_font):
        return True, num_depth
    if num_depth is not None:
        return True, num_depth
    if struct_info is not None:
        return True, None
    if _STRUCT_EN_RE.match(norm) or _STRUCT_CJK_RE.match(norm):
        return False, None

    if (
        _REQ_ID_RE.match(norm)
        or norm.endswith(":")
        or _DECORATIVE_RUN_RE.search(norm)
        or _UML_MULTIPLICITY_RE.search(norm)
    ):
        return False, None

    n_words = _effective_word_count(norm)
    if len(norm) > 140 or n_words > (_TITLE_BAND_MAX_WORDS if title_band else 16):
        return False, None

    fs = meta.get("font_size")
    bold = bool(meta.get("bold"))
    has_fs = isinstance(fs, (int, float)) and fs > 0
    styled_app = bold or (body_font is not None and has_fs and fs >= body_font * 1.05)
    if (
        _SINGLE_LETTER_APP_RE.match(norm)
        and styled_app
        and n_words <= 12
        and not _ends_sentence(norm)
        and _sentence_count(norm) < 2
    ):
        return True, None

    if _sentence_count(norm) >= 2:
        return False, None
    bold_ok = bold and not _ends_sentence(norm) and n_words <= (_TITLE_BAND_MAX_WORDS if title_band else 12)

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
    learned_prefixes: Mapping[str, str | None] | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    dropped_ids: set[str] = set()
    em = _em_size(body_font)
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
        if _parse_caption_prefix(raw1, learned_prefixes)[0] or _parse_caption_prefix(raw2, learned_prefixes)[0]:
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

        if not _vertically_adjacent(
            e1.get("bbox"),
            e2.get("bbox"),
            gap_scale=0.55,
            max_cap=1.4 * em,
            align_tol=3.0 * em,
            em=em,
        ):
            i += 1
            continue

        t1 = _nfkc(_clean_text(raw1))
        t2 = _nfkc(_clean_text(raw2))
        if not t1 or not t2 or t1[-1] in ".?!;:。！？；：":
            i += 1
            continue
        if (
            _is_list_block(raw2)
            or _REQ_ID_RE.match(t2)
            or t2.endswith(":")
            or _DECORATIVE_RUN_RE.search(t2)
            or _UML_MULTIPLICITY_RE.search(t2)
            or _APPENDIX_RE.match(t2)
            or _APPENDIX_SUB_RE.match(t2)
            or _LEADING_NUMBER_RE.match(t2)
            or _parse_numbered_heading(t2, m2, body_font) is not None
            or _parse_structural_keyword(t2, m2, body_font) is not None
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

    return _drop_merged_elements(elements, ordered, dropped_ids)


def _has_clear_heading_style(
    elem: dict[str, Any],
    body_font: float | None,
    min_top_fs: float | None,
) -> bool:
    meta = elem.get("metadata") or {}
    fs = meta.get("font_size")
    has_fs = isinstance(fs, (int, float)) and fs > 0
    if body_font is not None:
        if not has_fs or float(fs) < body_font * 1.15:
            return False
        if min_top_fs is not None and float(fs) < min_top_fs * 0.95:
            return False
        return True
    return bool(meta.get("bold"))


def _validate_heading_sequence(
    headings: list[tuple[dict[str, Any], int | None]],
    toc_matches: dict[str, int],
    slide_titles: set[str],
    body_font: float | None = None,
) -> list[tuple[dict[str, Any], int | None]]:
    parsed: list[tuple[dict[str, Any], int | None, tuple[int, ...] | None, bool, bool]] = []
    top_counts: dict[int, int] = {}
    sub_app_letters: set[str] = set()
    single_app_letters_by_style: dict[tuple[float, bool] | None, set[str]] = {}
    for elem, explicit_depth in headings:
        eid = elem["id"]
        has_toc = eid in toc_matches
        label = (elem.get("metadata") or {}).get("layout_label")
        is_labeled = (
            label == "title"
            or (label == "section_header" and _should_trust_section_header(elem, body_font))
            or eid in slide_titles
        )
        norm = _clean_text(_nfkc(elem.get("text", "")))
        m = _LEADING_NUMBER_RE.match(norm)
        parts: tuple[int, ...] | None = None
        if m:
            raw_parts = m.group(1).split(".")
            if len(raw_parts) <= 6:
                parts = tuple(int(p) for p in raw_parts)
                if (1 <= parts[0] <= 99) or (parts[0] == 0 and len(parts) >= 2):
                    top_counts[parts[0]] = top_counts.get(parts[0], 0) + 1
        m_asub = _APPENDIX_SUB_RE.match(norm)
        if m_asub:
            sub_app_letters.add(m_asub.group(1).split(".")[0])
        elif explicit_depth is None:
            m_let = _SINGLE_LETTER_APP_RE.match(norm)
            if m_let:
                single_app_letters_by_style.setdefault(_style_key(elem), set()).add(m_let.group(1))
        parsed.append((elem, explicit_depth, parts, has_toc, is_labeled))

    validated: list[tuple[dict[str, Any], int | None]] = []
    prev_top = 0
    seen_numbered = False
    max_top_fs: float | None = None
    for elem, explicit_depth, parts, has_toc, is_labeled in parsed:
        if parts is None or has_toc or is_labeled:
            if parts is not None and (has_toc or is_labeled) and 1 <= parts[0] <= 99:
                prev_top = max(prev_top, parts[0])
                seen_numbered = True
            if explicit_depth is None and not has_toc and not is_labeled:
                norm = _clean_text(_nfkc(_heading_text(elem)))
                m_let = _SINGLE_LETTER_APP_RE.match(norm)
                if m_let:
                    meta = elem.get("metadata") or {}
                    letter = m_let.group(1)
                    st = _style_key(elem)
                    has_ab_seq = st is not None and {"A", "B"}.issubset(single_app_letters_by_style.get(st, set()))
                    fs = meta.get("font_size")
                    bold = bool(meta.get("bold"))
                    has_fs = isinstance(fs, (int, float)) and fs > 0
                    styled_app = bold or (body_font is not None and has_fs and float(fs) >= body_font * 1.05)
                    if styled_app and seen_numbered and (letter in sub_app_letters or has_ab_seq):
                        explicit_depth = 1
                    else:
                        bold_ok = bold and not _ends_sentence(norm) and _effective_word_count(norm) <= 12
                        normal_ok = (
                            (float(fs) >= body_font * 1.15 or (bold_ok and float(fs) >= body_font * 0.98))
                            if (body_font is not None and has_fs)
                            else bold_ok
                        )
                        if not normal_ok:
                            continue
            validated.append((elem, explicit_depth))
            continue

        first = parts[0]
        is_preface = first == 0 and len(parts) >= 2
        if not is_preface and (first < 1 or first > 999):
            continue
        has_seq_support = top_counts.get(first, 0) >= 2 or (first + 1) in top_counts
        if is_preface:
            ok = True
        elif first <= 99:
            if prev_top == 0:
                ok = first <= 3 or (len(parts) >= 2 and first <= 20 and explicit_depth is not None) or has_seq_support
            else:
                ok = first <= prev_top + 3 or (first <= prev_top + 10 and has_seq_support)
        else:
            ok = False

        if not ok:
            if len(parts) == 1 and 1 <= first <= 999 and _has_clear_heading_style(elem, body_font, max_top_fs):
                ok = True
            else:
                continue

        seen_numbered = True
        if 1 <= first <= 99:
            prev_top = max(prev_top, first)
        if len(parts) == 1:
            fs = (elem.get("metadata") or {}).get("font_size")
            if isinstance(fs, (int, float)) and fs > 0:
                max_top_fs = float(fs) if max_top_fs is None else max(max_top_fs, float(fs))
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

    struct_items: dict[str, tuple[int, int]] = {}
    struct_pos: dict[str, int] = {}
    numeric_entries: list[tuple[int, dict[str, Any], int]] = []
    has_numeric_lvl1 = False

    anchor_votes: dict[tuple[float, bool], dict[int, int]] = {}
    toc_anchor_keys: set[tuple[float, bool]] = set()
    unanchored_counts: dict[tuple[float, bool], int] = {}
    for pos, (elem, explicit_depth) in enumerate(headings):
        eid = elem["id"]
        meta = elem.get("metadata") or {}
        if eid in slide_titles or meta.get("layout_label") == "title":
            continue
        if explicit_depth is None and eid not in toc_ids:
            s_info = _parse_structural_keyword(
                _clean_text(_heading_text(elem)),
                meta,
                body_font,
                source=str(elem.get("source", "text_layer")),
            )
            if s_info is not None:
                struct_items[eid] = s_info
                struct_pos[eid] = pos
                continue
        key = _style_key(elem)
        if key is None:
            continue
        if explicit_depth is not None:
            in_toc = eid in toc_ids
            if not in_toc:
                numeric_entries.append((pos, elem, explicit_depth))
                if explicit_depth == 1:
                    has_numeric_lvl1 = True
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

    struct_levels: dict[str, int] = {}
    if struct_items:
        struct_tiers = sorted({tier for tier, _ in struct_items.values()})
        tier_rank = {tier: idx + 1 for idx, tier in enumerate(struct_tiers)}
        base_shift = 1 if (has_title and not has_numeric_lvl1) else 0
        min_top_lvl = 1 + base_shift
        top_tier = struct_tiers[0]
        top_eids = [eid for eid, (t, _) in struct_items.items() if t == top_tier]
        first_top_eid = min(top_eids, key=lambda e: struct_pos[e])
        pos_t0 = struct_pos[first_top_eid]
        key_t0 = _style_key(headings[pos_t0][0])
        if numeric_entries and key_t0 is not None:
            fs_t0, bold_t0 = key_t0
            for n_pos, n_elem, n_depth in numeric_entries:
                if n_pos >= pos_t0:
                    continue
                n_key = _style_key(n_elem)
                if n_key is None:
                    continue
                n_fs, n_bold = n_key
                if n_fs > fs_t0 * (1.0 + size_tolerance):
                    min_top_lvl = max(min_top_lvl, n_depth + 1)
                elif abs(n_fs - fs_t0) / max(n_fs, fs_t0) <= size_tolerance and (
                    (n_bold and not bold_t0) or (n_bold == bold_t0 and top_tier >= 2)
                ):
                    min_top_lvl = max(min_top_lvl, n_depth + 1)
        for eid, (tier, parts_len) in struct_items.items():
            lvl = tier_rank[tier] + (min_top_lvl - 1) + (parts_len - 1)
            struct_levels[eid] = min(_MAX_HEADING_LEVEL, lvl)

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
        eid = elem["id"]
        if explicit_depth is not None:
            levels[eid] = explicit_depth
            continue
        if eid in struct_levels:
            levels[eid] = struct_levels[eid]
            continue
        meta = elem.get("metadata") or {}
        if meta.get("layout_label") == "title" or eid in slide_titles:
            levels[eid] = 1
            continue
        key = _style_key(elem)
        if key is not None:
            shift = int(not anchor_levels and (has_title or elem.get("page") in slide_pages))
            levels[eid] = min(_MAX_HEADING_LEVEL, style_rank[key] + shift)
        else:
            base_lvl = 2 if has_title else 1
            levels[eid] = base_lvl + int(not has_title and elem.get("page") in slide_pages)
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

    page_heights = {
        int(p["number"]): float(p["height_pt"])
        for p in doc_meta.get("pages", ())
        if isinstance(p, Mapping) and p.get("number") is not None and p.get("height_pt")
    }
    pre_body_font = _weighted_median_font_size(elements)
    _project_semantic_regions(
        elements,
        semantic_regions,
        page_heights=page_heights,
        body_font=pre_body_font,
    )

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
    learned_prefixes = _learn_caption_prefixes(elements, body_font, top_band_ids)
    ordered = _merge_multiline_headings(elements, ordered, body_font, top_band_ids, learned_prefixes)
    toc_matches, toc_pages, toc_elem_ids = _match_toc_entries(ordered, resolved_toc, body_font)
    ordered = _merge_multiline_captions(elements, ordered, body_font, learned_prefixes)

    def collect_headings() -> list[tuple[dict[str, Any], int | None]]:
        found = []
        for _, elem in ordered:
            eid = elem["id"]
            if eid in toc_matches:
                found.append((elem, toc_matches[eid]))
                continue
            if elem.get("page") in toc_pages or eid in toc_elem_ids:
                continue
            is_heading, explicit_depth = _is_heading_candidate(elem, body_font, eid in top_band_ids)
            if is_heading:
                found.append((elem, explicit_depth))
        return found

    _link_captions(elements, body_font, _slide_title_ids(collect_headings(), landscape_heights), learned_prefixes)
    _propagate_captions_to_ir(out, elements)
    raw_candidates = collect_headings()
    raw_slide_titles = _slide_title_ids(raw_candidates, landscape_heights)
    heading_candidates = _validate_heading_sequence(raw_candidates, toc_matches, raw_slide_titles, body_font)
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
                    "title": _clean_text(_heading_text(elem)),
                    "level": h_level,
                    "page": elem.get("page"),
                    "parent_id": parent_id,
                }
            )
        else:
            meta["section_parent_id"] = stack[-1][1] if stack else None
            if (
                elem.get("type") == "text"
                and eid not in toc_elem_ids
                and elem.get("page") not in toc_pages
                and meta.get("role") is None
                and meta.get("layout_label") in (None, "text")
                and _is_native_list_item(elem.get("text", ""))
            ):
                meta["layout_label"] = "list_item"

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
                "title": _clean_text(_heading_text(elem)),
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
