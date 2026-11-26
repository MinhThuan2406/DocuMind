from __future__ import annotations

import html
import re
from collections.abc import Mapping, Sequence
from typing import Any

_LIST_PREFIX_RE = re.compile(
    r"^(?:"
    r"[-*+]\s+"
    r"|\[[ xX]\]\s+"
    r"|[•●○◦▪▫‣⁃・･]\s*"
    r"|(?:\(?\d+[.)]|\(?[A-Za-z][.)]|[ivxlcdmIVXLCDM]+[.)])\s+"
    r")"
)
_MERMAID_RESERVED = frozenset(
    {"end", "subgraph", "graph", "flowchart", "style", "class", "classdef", "click", "linkstyle", "direction"}
)
_PIPE_ESCAPE_RE = re.compile(r"(?<!\\)\|")


def _single_line(text: Any) -> str:
    return " ".join(str(text or "").split())


def _is_furniture(elem: Mapping[str, Any], meta: Mapping[str, Any]) -> bool:
    role = str(meta.get("role") or "").lower()
    label = str(meta.get("layout_label") or "").lower()
    etype = str(elem.get("type") or "").lower()
    return (
        role in ("page_header", "page_footer")
        or label in ("page_header", "page_footer")
        or etype in ("header", "footer", "page_header", "page_footer")
        or bool(meta.get("is_running_furniture"))
    )


def _cell_span(cell: Mapping[str, Any], primary: str, alt: str) -> int:
    raw = cell.get(primary)
    if raw is None:
        raw = cell.get(alt)
    try:
        return max(1, int(raw or 1))
    except (TypeError, ValueError):
        return 1


def _has_merged_cells(cells: Sequence[Any]) -> bool:
    for cell in cells:
        if not isinstance(cell, Mapping):
            continue
        if _cell_span(cell, "rowspan", "row_span") > 1 or _cell_span(cell, "colspan", "col_span") > 1:
            return True
        if cell.get("is_merged"):
            return True
    return False


def _cells_to_html(cells: Sequence[Any]) -> str:
    rows: dict[int, list[Mapping[str, Any]]] = {}
    for cell in cells:
        if isinstance(cell, Mapping):
            rows.setdefault(int(cell.get("row", 0)), []).append(cell)
    if not rows:
        return ""
    lines = ["<table>"]
    for r_idx in sorted(rows):
        row_cells = sorted(rows[r_idx], key=lambda c: int(c.get("col", 0)))
        parts = ["<tr>"]
        for c in row_cells:
            tag = "th" if c.get("is_header") else "td"
            rs = _cell_span(c, "rowspan", "row_span")
            cs = _cell_span(c, "colspan", "col_span")
            attrs = ""
            if rs > 1:
                attrs += f' rowspan="{rs}"'
            if cs > 1:
                attrs += f' colspan="{cs}"'
            body = html.escape(str(c.get("text") or "").strip()).replace("\n", "<br>")
            parts.append(f"<{tag}{attrs}>{body}</{tag}>")
        parts.append("</tr>")
        lines.append("".join(parts))
    lines.append("</table>")
    return "\n".join(lines)


def _sanitize_gfm_cell(text: Any) -> str:
    norm = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not norm:
        return ""
    joined = "<br>".join(ln.strip() for ln in norm.splitlines())
    return _PIPE_ESCAPE_RE.sub(r"\\|", joined)


def _render_table(
    elem: Mapping[str, Any],
    meta: Mapping[str, Any],
    tables_by_id: Mapping[str, Mapping[str, Any]],
    tables_by_elem_id: Mapping[str, Mapping[str, Any]],
    children_by_parent: Mapping[str, list[tuple[int, int, Mapping[str, Any]]]],
) -> str | None:
    elem_id = str(elem.get("id") or "")
    tid = meta.get("table_id") or elem.get("table_id")
    tbl = tables_by_id.get(str(tid)) if tid is not None else None
    if tbl is None and elem_id:
        tbl = tables_by_elem_id.get(elem_id)

    if tbl is not None:
        cells = tbl.get("cells") or ()
        if _has_merged_cells(cells):
            raw_html = str(tbl.get("html") or "").strip()
            if raw_html:
                return "\n".join(ln for ln in raw_html.splitlines() if ln.strip())
            built = _cells_to_html(cells)
            if built:
                return built

        cell_mappings = [c for c in cells if isinstance(c, Mapping)]
        if cell_mappings:
            max_r = max(int(c.get("row", 0)) for c in cell_mappings) + 1
            max_c = max(int(c.get("col", 0)) for c in cell_mappings) + 1
            num_rows = max(int(tbl.get("num_rows") or 0), max_r)
            num_cols = max(int(tbl.get("num_cols") or 0), max_c)
            if num_rows > 0 and num_cols > 0:
                grid: list[list[str]] = [[""] * num_cols for _ in range(num_rows)]
                for c in cell_mappings:
                    r_idx = int(c.get("row", 0))
                    c_idx = int(c.get("col", 0))
                    if 0 <= r_idx < num_rows and 0 <= c_idx < num_cols:
                        val = _sanitize_gfm_cell(c.get("text"))
                        if val:
                            grid[r_idx][c_idx] = f"{grid[r_idx][c_idx]} {val}".strip() if grid[r_idx][c_idx] else val
                header = "| " + " | ".join(grid[0]) + " |"
                sep = "| " + " | ".join("---" for _ in range(num_cols)) + " |"
                body = ["| " + " | ".join(row) + " |" for row in grid[1:]]
                return "\n".join([header, sep, *body])

        raw_html = str(tbl.get("html") or "").strip()
        if raw_html:
            return "\n".join(ln for ln in raw_html.splitlines() if ln.strip())

    children = children_by_parent.get(elem_id, ())
    if children:
        child_lines = [
            str(c.get("text") or "").strip()
            for _, _, c in sorted(children, key=lambda item: (item[0], item[1]))
            if str(c.get("text") or "").strip()
        ]
        if child_lines:
            return "\n".join(child_lines)

    fallback = str(elem.get("text") or "").strip()
    return fallback or None


def _mermaid_safe_id(raw_id: Any, prefix: str, used: set[str]) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", str(raw_id or "")).strip("_")
    if not cleaned or cleaned[0].isdigit() or cleaned.lower() in _MERMAID_RESERVED:
        cleaned = f"{prefix}_{cleaned}" if cleaned else prefix
    candidate = cleaned
    suffix = 2
    while candidate in used:
        candidate = f"{cleaned}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _mermaid_label(text: Any, fallback: str = "") -> str:
    s = _single_line(text) or fallback
    return s.replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def _mermaid_node_decl(mid: str, node: Mapping[str, Any]) -> str:
    label = _mermaid_label(node.get("label"), fallback=str(node.get("id") or mid))
    shape = str(node.get("shape_type") or "rect").lower()
    if shape == "round_rect":
        return f'{mid}("{label}")'
    if shape == "diamond":
        return f'{mid}{{"{label}"}}'
    if shape in ("ellipse", "circle"):
        return f'{mid}(("{label}"))'
    return f'{mid}["{label}"]'


def _mermaid_edge_arrow(direction: str, style: str) -> str:
    dashed = style in ("dashed", "dotted")
    if direction == "bidirectional":
        return "<-.->" if dashed else "<-->"
    if direction == "undirected":
        return "-.-" if dashed else "---"
    return "-.->" if dashed else "-->"


def _render_diagram(diag: Mapping[str, Any]) -> str | None:
    nodes = [n for n in (diag.get("nodes") or ()) if isinstance(n, Mapping)]
    edges = [e for e in (diag.get("edges") or ()) if isinstance(e, Mapping)]
    groups = [g for g in (diag.get("groups") or ()) if isinstance(g, Mapping)]
    if not nodes and not edges and not groups:
        return None

    used_ids: set[str] = set()
    node_id_map: dict[str, str] = {}
    nodes_by_raw: dict[str, Mapping[str, Any]] = {}
    for idx, node in enumerate(nodes):
        raw_nid = str(node.get("id") or f"node_{idx}")
        nodes_by_raw[raw_nid] = node
        node_id_map[raw_nid] = _mermaid_safe_id(raw_nid, "n", used_ids)

    group_id_map: dict[str, str] = {}
    groups_by_raw: dict[str, Mapping[str, Any]] = {}
    group_members: dict[str, list[str]] = {}
    child_groups: dict[str, list[str]] = {}
    node_parent_group: dict[str, str] = {}

    for idx, grp in enumerate(groups):
        raw_gid = str(grp.get("id") or f"group_{idx}")
        groups_by_raw[raw_gid] = grp
        group_id_map[raw_gid] = _mermaid_safe_id(raw_gid, "g", used_ids)
        group_members.setdefault(raw_gid, [])
        child_groups.setdefault(raw_gid, [])

    for grp in groups:
        raw_gid = str(grp.get("id") or "")
        parent_gid = grp.get("parent_group_id")
        if parent_gid is not None and str(parent_gid) in groups_by_raw and str(parent_gid) != raw_gid:
            child_groups.setdefault(str(parent_gid), []).append(raw_gid)
        for member in grp.get("member_node_ids") or ():
            mid = str(member)
            if mid in nodes_by_raw and mid not in node_parent_group:
                node_parent_group[mid] = raw_gid
                group_members[raw_gid].append(mid)

    for node in nodes:
        raw_nid = str(node.get("id") or "")
        parent_gid = node.get("parent_group_id")
        if parent_gid is not None and str(parent_gid) in groups_by_raw and raw_nid not in node_parent_group:
            node_parent_group[raw_nid] = str(parent_gid)
            group_members[str(parent_gid)].append(raw_nid)

    lines: list[str] = ["```mermaid", "flowchart TD"]
    emitted_groups: set[str] = set()
    emitted_nodes: set[str] = set()

    def _emit_group(raw_gid: str, indent: str) -> None:
        if raw_gid in emitted_groups or raw_gid not in groups_by_raw:
            return
        emitted_groups.add(raw_gid)
        grp = groups_by_raw[raw_gid]
        mgid = group_id_map[raw_gid]
        glabel = _mermaid_label(grp.get("label"), fallback=mgid)
        lines.append(f'{indent}subgraph {mgid}["{glabel}"]')
        for raw_nid in group_members.get(raw_gid, ()):
            if raw_nid not in emitted_nodes:
                emitted_nodes.add(raw_nid)
                lines.append(f"{indent}    {_mermaid_node_decl(node_id_map[raw_nid], nodes_by_raw[raw_nid])}")
        for sub_gid in child_groups.get(raw_gid, ()):
            _emit_group(sub_gid, indent + "    ")
        lines.append(f"{indent}end")

    for grp in groups:
        raw_gid = str(grp.get("id") or "")
        parent_gid = grp.get("parent_group_id")
        if parent_gid is None or str(parent_gid) not in groups_by_raw:
            _emit_group(raw_gid, "    ")
    for grp in groups:
        _emit_group(str(grp.get("id") or ""), "    ")

    for node in nodes:
        raw_nid = str(node.get("id") or "")
        if raw_nid not in emitted_nodes:
            emitted_nodes.add(raw_nid)
            lines.append(f"    {_mermaid_node_decl(node_id_map[raw_nid], node)}")

    for edge in edges:
        src_raw = edge.get("source_node_id")
        dst_raw = edge.get("target_node_id")
        if src_raw is None or dst_raw is None:
            continue
        src = node_id_map.get(str(src_raw))
        dst = node_id_map.get(str(dst_raw))
        if src is None or dst is None:
            continue
        arrow = _mermaid_edge_arrow(
            str(edge.get("direction") or "directed").lower(),
            str(edge.get("style") or "solid").lower(),
        )
        edge_label = _single_line(edge.get("label"))
        if edge_label:
            safe_lbl = edge_label.replace('"', "&quot;").replace("|", "&#124;")
            lines.append(f'    {src} {arrow}|"{safe_lbl}"| {dst}')
        else:
            lines.append(f"    {src} {arrow} {dst}")

    lines.append("```")
    return "\n".join(lines)


def _render_formula(
    elem: Mapping[str, Any],
    meta: Mapping[str, Any],
    formulas_by_id: Mapping[str, Mapping[str, Any]],
    formulas_by_elem_id: Mapping[str, Mapping[str, Any]],
) -> str | None:
    latex = str(meta.get("latex") or "").strip()
    if not latex:
        fid = meta.get("formula_id") or elem.get("formula_id")
        f_entry = formulas_by_id.get(str(fid)) if fid is not None else None
        if f_entry is None:
            eid = str(elem.get("id") or "")
            if eid:
                f_entry = formulas_by_elem_id.get(eid)
        if f_entry is not None:
            latex = str(f_entry.get("latex") or "").strip()
    if latex:
        if latex.startswith("$$") and latex.endswith("$$") and len(latex) > 4:
            latex = latex[2:-2].strip()
        if latex:
            return f"$$\n{latex}\n$$"
    fallback = str(elem.get("text") or "").strip()
    return fallback or None


def _render_heading(
    elem: Mapping[str, Any],
    meta: Mapping[str, Any],
    pno: int,
    outline_by_id: Mapping[str, int],
    outline_by_page_title: Mapping[tuple[int, str], int],
) -> str | None:
    raw = meta.get("heading_clean_text") or elem.get("text") or ""
    heading_text = _single_line(raw)
    if not heading_text:
        return None
    raw_level = meta.get("heading_level")
    if raw_level is None:
        eid = str(elem.get("id") or "")
        if eid and eid in outline_by_id:
            raw_level = outline_by_id[eid]
        else:
            raw_level = outline_by_page_title.get((pno, heading_text))
            if raw_level is None:
                raw_level = outline_by_page_title.get((pno, _single_line(elem.get("text"))))
    if raw_level is None:
        label = str(meta.get("layout_label") or "").lower()
        etype = str(elem.get("type") or "").lower()
        raw_level = 1 if (label == "title" or etype == "title") else 2
    try:
        level = max(1, min(6, int(raw_level)))
    except (TypeError, ValueError):
        level = 2
    return f"{'#' * level} {heading_text}"


def _render_code(elem: Mapping[str, Any], meta: Mapping[str, Any]) -> str | None:
    code_text = str(elem.get("text") or "").strip("\r\n")
    if not code_text.strip():
        return None
    max_run = max((len(m.group(0)) for m in re.finditer(r"`+", code_text)), default=0)
    fence = "`" * max(3, max_run + 1)
    lang = str(meta.get("language") or "").strip()
    return f"{fence}{lang}\n{code_text}\n{fence}"


def _render_caption(elem: Mapping[str, Any]) -> str | None:
    lines = [ln.strip() for ln in str(elem.get("text") or "").splitlines() if ln.strip()]
    if not lines:
        return None
    inner = "\n".join(lines)
    if inner.startswith("*") and inner.endswith("*") and len(inner) >= 3:
        return inner
    return f"*{inner}*"


def _render_list_item(elem: Mapping[str, Any]) -> str | None:
    raw = str(elem.get("text") or "").strip()
    if not raw:
        return None
    lines = [ln.rstrip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return None
    first = lines[0].lstrip()
    if not _LIST_PREFIX_RE.match(first):
        first = f"- {first}"
    rest = [ln if ln.startswith(("  ", "\t")) else f"  {ln.lstrip()}" for ln in lines[1:]]
    return "\n".join([first, *rest])


def to_markdown(
    ir: Mapping[str, Any],
    *,
    include_furniture: bool = False,
    include_diagrams: bool = True,
    include_figures: bool = True,
    page_breaks: bool = False,
) -> str:
    metadata = ir.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}

    tables_by_id: dict[str, Mapping[str, Any]] = {}
    tables_by_elem_id: dict[str, Mapping[str, Any]] = {}
    for tbl in ir.get("tables") or ():
        if not isinstance(tbl, Mapping):
            continue
        if tbl.get("id") is not None:
            tables_by_id[str(tbl["id"])] = tbl
        t_eid = (tbl.get("metadata") or {}).get("element_id") if isinstance(tbl.get("metadata"), Mapping) else None
        if t_eid is not None:
            tables_by_elem_id[str(t_eid)] = tbl

    diagrams_by_id: dict[str, Mapping[str, Any]] = {}
    diagrams_by_elem_id: dict[str, Mapping[str, Any]] = {}
    for diag in ir.get("diagrams") or ():
        if not isinstance(diag, Mapping):
            continue
        if diag.get("id") is not None:
            diagrams_by_id[str(diag["id"])] = diag
        d_eid = diag.get("element_id")
        if d_eid is None and isinstance(diag.get("metadata"), Mapping):
            d_eid = diag["metadata"].get("element_id")
        if d_eid is not None:
            diagrams_by_elem_id[str(d_eid)] = diag

    formulas_by_id: dict[str, Mapping[str, Any]] = {}
    formulas_by_elem_id: dict[str, Mapping[str, Any]] = {}
    for fml in ir.get("formulas") or ():
        if not isinstance(fml, Mapping):
            continue
        if fml.get("id") is not None:
            formulas_by_id[str(fml["id"])] = fml
        f_eid = fml.get("element_id")
        if f_eid is None and isinstance(fml.get("metadata"), Mapping):
            f_eid = fml["metadata"].get("element_id")
        if f_eid is not None:
            formulas_by_elem_id[str(f_eid)] = fml

    outline_by_id: dict[str, int] = {}
    outline_by_page_title: dict[tuple[int, str], int] = {}
    for entry in metadata.get("outline") or ():
        if not isinstance(entry, Mapping) or entry.get("level") is None:
            continue
        try:
            lvl_val = int(entry["level"])
        except (TypeError, ValueError):
            continue
        if entry.get("id") is not None:
            outline_by_id[str(entry["id"])] = lvl_val
        if entry.get("page") is not None and entry.get("title"):
            outline_by_page_title[(int(entry["page"]), _single_line(entry["title"]))] = lvl_val

    pages_set: set[int] = set()
    for pmeta in metadata.get("pages") or ():
        if isinstance(pmeta, Mapping) and "number" in pmeta:
            pages_set.add(int(pmeta["number"]))

    level1_by_page: dict[int, list[tuple[int, int, Mapping[str, Any]]]] = {}
    children_by_parent: dict[str, list[tuple[int, int, Mapping[str, Any]]]] = {}
    top_elem_ids: set[str] = set()

    for idx, elem in enumerate(ir.get("elements") or ()):
        if not isinstance(elem, Mapping):
            continue
        lvl = int(elem.get("level", 1))
        ro = int(elem.get("reading_order", idx))
        if lvl == 1:
            pno = int(elem.get("page", 0))
            pages_set.add(pno)
            level1_by_page.setdefault(pno, []).append((ro, idx, elem))
            if elem.get("id") is not None:
                top_elem_ids.add(str(elem["id"]))
        elif lvl == 2 and elem.get("parent_id") is not None:
            children_by_parent.setdefault(str(elem["parent_id"]), []).append((ro, idx, elem))

    page_chunks: list[str] = []
    for pno in sorted(pages_set):
        entries = sorted(level1_by_page.get(pno, ()), key=lambda item: (item[0], item[1]))
        blocks: list[tuple[str, str]] = []

        for _, _, elem in entries:
            meta = elem.get("metadata")
            if not isinstance(meta, Mapping):
                meta = {}
            if not include_furniture and _is_furniture(elem, meta):
                continue

            etype = str(elem.get("type") or "").lower()
            role = str(meta.get("role") or "").lower()
            label = str(meta.get("layout_label") or "").lower()
            elem_id = str(elem.get("id") or "")

            if etype == "table" or label == "table":
                rendered = _render_table(elem, meta, tables_by_id, tables_by_elem_id, children_by_parent)
                if rendered:
                    blocks.append(("table", rendered))
            elif etype == "figure" or (label == "picture" and etype != "text"):
                diag_id = meta.get("diagram_id") or elem.get("diagram_id")
                diag = diagrams_by_id.get(str(diag_id)) if (diag_id is not None and include_diagrams) else None
                if diag is None and include_diagrams and diag_id is not None and elem_id:
                    diag = diagrams_by_elem_id.get(elem_id)
                mermaid = _render_diagram(diag) if diag is not None else None
                if mermaid:
                    blocks.append(("diagram", mermaid))
                elif include_figures:
                    cap_id = meta.get("caption_id")
                    cap_text = _single_line(meta.get("caption_text"))
                    if cap_text and (cap_id is None or str(cap_id) not in top_elem_ids):
                        blocks.append(("figure", f"*[Figure: {cap_text}]*"))
                    elif fig_text := _single_line(elem.get("text")):
                        blocks.append(("figure", f"*[Figure: {fig_text}]*"))
                    else:
                        blocks.append(("figure", "*[Figure]*"))
            elif etype == "formula" or label == "formula" or role == "formula":
                rendered = _render_formula(elem, meta, formulas_by_id, formulas_by_elem_id)
                if rendered:
                    blocks.append(("formula", rendered))
            elif role == "heading" or (
                role not in ("caption", "page_header", "page_footer")
                and (label in ("title", "section_header") or etype in ("heading", "title", "section_header"))
            ):
                rendered = _render_heading(elem, meta, pno, outline_by_id, outline_by_page_title)
                if rendered:
                    blocks.append(("heading", rendered))
            elif label == "code" or role == "code" or etype == "code":
                rendered = _render_code(elem, meta)
                if rendered:
                    blocks.append(("code", rendered))
            elif role == "caption" or label == "caption" or etype == "caption":
                rendered = _render_caption(elem)
                if rendered:
                    blocks.append(("caption", rendered))
            elif label == "list_item" or role == "list_item" or etype == "list_item":
                rendered = _render_list_item(elem)
                if rendered:
                    blocks.append(("list_item", rendered))
            else:
                lines = [ln.rstrip() for ln in str(elem.get("text") or "").strip().splitlines()]
                prose = "\n".join(lines).strip()
                if prose:
                    blocks.append(("prose", prose))

        if not blocks:
            continue

        page_parts: list[str] = []
        for idx_b, (btype, content) in enumerate(blocks):
            if idx_b > 0:
                prev_type = blocks[idx_b - 1][0]
                page_parts.append("\n" if (btype == "list_item" and prev_type == "list_item") else "\n\n")
            page_parts.append(content)
        page_chunks.append("".join(page_parts))

    sep = "\n\n---\n\n" if page_breaks else "\n\n"
    return sep.join(page_chunks)


extract_markdown = to_markdown
