"""
DocuMind Unified IR Postprocessor & Markdown Generator
Implements:
1. Column-Aware Reading Order Sorting (2-column layout detection and top-to-bottom column sorting)
2. Hierarchy Tree Construction (filling level and parent_id for title, section, text, list_item, caption, diagram)
3. Linking elements <-> tables <-> captions (filling table_id, non-empty table text, and caption metadata)
4. Markdown (.md) generation from normalized Unified IR
5. Re-drawing visualization overlays with updated reading order badges
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


# Color palette for layout visualization (RGB tuples)
CATEGORY_COLORS = {
    "table": (0, 168, 107),              # Emerald Green
    "table_caption": (46, 139, 87),      # Sea Green
    "caption": (46, 139, 87),
    "table_footnote": (60, 179, 113),
    "title": (30, 144, 255),             # Dodger Blue
    "section_header": (30, 144, 255),    # Dodger Blue
    "header": (70, 130, 180),
    "page_header": (70, 130, 180),
    "footer": (160, 82, 45),
    "page_footer": (160, 82, 45),
    "page_number": (0, 139, 139),
    "text_block": (105, 105, 105),       # Dim Gray
    "text": (105, 105, 105),
    "paragraph": (105, 105, 105),
    "list_group": (255, 140, 0),         # Dark Orange
    "list_item": (255, 140, 0),
    "figure": (147, 112, 219),           # Medium Purple
    "picture": (147, 112, 219),
    "diagram_group": (0, 180, 110),      # Bright Teal
    "vector_diagram_group": (0, 180, 110),
    "formula": (220, 20, 60),
    "footnote": (188, 143, 143),
}
DEFAULT_COLOR = (120, 120, 120)


def get_color(label: str) -> Tuple[int, int, int]:
    normalized = label.lower().replace(" ", "_")
    return CATEGORY_COLORS.get(normalized, DEFAULT_COLOR)


def load_font(size: int):
    font_paths = [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two bboxes [xmin, ymin, xmax, ymax]."""
    if not box1 or not box2 or len(box1) < 4 or len(box2) < 4:
        return 0.0
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


# ----------------------------------------------------------------------
# 1. Column-Aware Reading Order Sorting
# ----------------------------------------------------------------------
def sort_elements_reading_order(
    elements: List[Dict[str, Any]],
    page_w: float,
    page_h: float,
) -> List[Dict[str, Any]]:
    """
    Identifies 2-column layout and sorts elements:
    - Full-width header blocks (top of page)
    - Left column (top to bottom)
    - Full-width body blocks (if any, placed between columns)
    - Right column (top to bottom)
    - Full-width footer blocks (bottom of page)
    """
    mid_x = page_w / 2.0
    classified = []

    for e in elements:
        bbox = e.get("bbox")
        if not bbox or len(bbox) < 4:
            classified.append(("full_width", e))
            continue
        xmin, ymin, xmax, ymax = bbox
        cx = (xmin + xmax) / 2.0
        w_ratio = (xmax - xmin) / page_w

        # Thresholds per specification
        if (xmin < 0.42 * page_w and xmax > 0.58 * page_w) or w_ratio > 0.72:
            col = "full_width"
        elif xmax <= 0.54 * page_w or cx < 0.48 * page_w:
            col = "left_col"
        elif xmin >= 0.46 * page_w or cx > 0.52 * page_w:
            col = "right_col"
        else:
            col = "left_col" if cx < mid_x else "right_col"

        classified.append((col, e))

    n_left = sum(1 for c, _ in classified if c == "left_col")
    n_right = sum(1 for c, _ in classified if c == "right_col")
    is_2col = (n_left >= 2 and n_right >= 2)

    if not is_2col:
        # Single column layout: sort top-to-bottom
        return sorted(
            elements,
            key=lambda it: (it["bbox"][1] if it.get("bbox") else 0, it["bbox"][0] if it.get("bbox") else 0),
        )

    # 2-column layout separation
    left_elems = [e for c, e in classified if c == "left_col"]
    right_elems = [e for c, e in classified if c == "right_col"]
    full_elems = [e for c, e in classified if c == "full_width"]

    left_min_y = min((e["bbox"][1] for e in left_elems if e.get("bbox")), default=0)
    right_min_y = min((e["bbox"][1] for e in right_elems if e.get("bbox")), default=0)
    col_start_y = min(left_min_y, right_min_y)

    left_max_y = max((e["bbox"][3] for e in left_elems if e.get("bbox")), default=page_h)
    right_max_y = max((e["bbox"][3] for e in right_elems if e.get("bbox")), default=page_h)
    col_end_y = max(left_max_y, right_max_y)

    top_full = []
    bottom_full = []
    body_full = []
    for e in full_elems:
        bbox = e.get("bbox", [0, 0, 0, 0])
        if bbox[3] <= col_start_y + 15:
            top_full.append(e)
        elif bbox[1] >= col_end_y - 15:
            bottom_full.append(e)
        else:
            body_full.append(e)

    top_full.sort(key=lambda it: it["bbox"][1] if it.get("bbox") else 0)
    left_elems.sort(key=lambda it: it["bbox"][1] if it.get("bbox") else 0)
    right_elems.sort(key=lambda it: it["bbox"][1] if it.get("bbox") else 0)
    bottom_full.sort(key=lambda it: it["bbox"][1] if it.get("bbox") else 0)
    body_full.sort(key=lambda it: it["bbox"][1] if it.get("bbox") else 0)

    # In 2-column layout: Left column first, then Right column
    return top_full + left_elems + body_full + right_elems + bottom_full


# ----------------------------------------------------------------------
# 2. Markdown Table Renderer from Cells
# ----------------------------------------------------------------------
def render_markdown_table_from_cells(
    cells: List[Dict[str, Any]],
    num_rows: int,
    num_cols: int,
) -> str:
    """Renders a clean Markdown table string from structured table cells."""
    if not cells or num_rows <= 0 or num_cols <= 0:
        return ""

    grid = [["" for _ in range(num_cols)] for _ in range(num_rows)]
    for c in cells:
        r = c.get("row", 0)
        col = c.get("col", 0)
        txt = c.get("text", "").replace("\n", " ").replace("|", "\\|").strip()
        if r < num_rows and col < num_cols:
            grid[r][col] = txt

    lines = []
    # Header row
    header_cells = grid[0]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("| " + " | ".join(["---"] * num_cols) + " |")

    # Data rows
    for r in range(1, num_rows):
        lines.append("| " + " | ".join(grid[r]) + " |")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# 3. Main Postprocessing Function
# ----------------------------------------------------------------------
def postprocess_ir_structure(
    doc_ir: Dict[str, Any],
    page_w: Optional[float] = None,
    page_h: Optional[float] = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Applies the 3 structural fixes and builds the Markdown representation.
    Returns (updated_doc_ir, markdown_text).
    """
    meta = doc_ir.setdefault("metadata", {})
    pw = page_w or meta.get("page_width", 1000.0)
    ph = page_h or meta.get("page_height", 1000.0)

    elements = doc_ir.get("elements", [])
    tables = doc_ir.get("tables", [])
    diagrams = doc_ir.get("diagrams", [])

    # Step 1: Column-aware sorting
    sorted_elements = sort_elements_reading_order(elements, pw, ph)

    # Step 2: Re-index IDs and reading_order
    old_to_new_id = {}
    for idx, e in enumerate(sorted_elements, 1):
        old_id = e["id"]
        new_id = f"elem_{idx:04d}"
        old_to_new_id[old_id] = new_id
        e["id"] = new_id
        e["reading_order"] = idx - 1

    # Update diagram merged_element_ids if mapped from old IDs
    for dg in diagrams:
        if "merged_element_ids" in dg:
            dg["merged_element_ids"] = [
                old_to_new_id.get(mid, mid) for mid in dg["merged_element_ids"]
            ]

    # Step 3: Link Elements <-> Tables <-> Captions
    table_elements = [e for e in sorted_elements if e.get("type") == "table"]
    caption_elements = [e for e in sorted_elements if e.get("type") == "caption"]

    for tbl in tables:
        tbl_bbox = tbl.get("bbox")
        best_iou = 0.0
        best_elem = None

        if tbl_bbox:
            for te in table_elements:
                if te.get("bbox"):
                    iou = compute_iou(tbl_bbox, te["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_elem = te

        # Fallback if IoU was 0 (e.g. single table)
        if best_elem is None and table_elements:
            best_elem = table_elements[0]

        if best_elem is not None:
            tbl_id = tbl["id"]
            best_elem["table_id"] = tbl_id
            best_elem.setdefault("metadata", {})["table_id"] = tbl_id

            # Find matching caption (nearest vertically in same horizontal span)
            matched_cap = None
            min_cap_dist = float("inf")
            te_box = best_elem.get("bbox", [0, 0, 0, 0])

            for cap in caption_elements:
                c_box = cap.get("bbox", [0, 0, 0, 0])
                # Check column overlap
                if not (c_box[2] < te_box[0] - 80 or c_box[0] > te_box[2] + 80):
                    dist = min(abs(c_box[1] - te_box[3]), abs(te_box[1] - c_box[3]))
                    if dist < min_cap_dist:
                        min_cap_dist = dist
                        matched_cap = cap

            tbl_meta = tbl.setdefault("metadata", {})
            tbl_meta["element_id"] = best_elem["id"]

            if matched_cap and min_cap_dist < 300:
                cap_txt = matched_cap.get("text", "").strip()
                tbl_meta["caption_id"] = matched_cap["id"]
                tbl_meta["caption_text"] = cap_txt
                best_elem["text"] = f"[Table {tbl_id}: {cap_txt}]"
            else:
                shape_str = f"{tbl.get('num_rows', 0)}x{tbl.get('num_cols', 0)}"
                best_elem["text"] = f"[Table {tbl_id}: {shape_str}]"

    # Step 4: Hierarchy Construction (parent_id & level)
    current_title_id = None
    current_section_id = None
    current_section_level = 1
    last_text_id = None
    last_text_level = 2

    for e in sorted_elements:
        t = e.get("type", "text")

        if t == "title":
            e["level"] = 1
            e["parent_id"] = None
            current_title_id = e["id"]
            current_section_id = e["id"]
            current_section_level = 1
        elif t in ("section_header", "header", "page_header"):
            lvl = 1 if current_title_id is None else 2
            e["level"] = lvl
            e["parent_id"] = current_title_id
            current_section_id = e["id"]
            current_section_level = lvl
            last_text_id = None
        elif t == "caption":
            # Link caption to nearest table or figure
            nearest_target = None
            min_dist = float("inf")
            e_box = e.get("bbox", [0, 0, 0, 0])

            for cand in sorted_elements:
                if cand["id"] == e["id"]:
                    continue
                if cand.get("type") in ("table", "picture", "figure", "diagram_group"):
                    c_box = cand.get("bbox", [0, 0, 0, 0])
                    if not (c_box[2] < e_box[0] - 80 or c_box[0] > e_box[2] + 80):
                        dist = min(abs(c_box[1] - e_box[3]), abs(e_box[1] - c_box[3]))
                        if dist < min_dist:
                            min_dist = dist
                            nearest_target = cand

            if nearest_target and min_dist < 300:
                e["parent_id"] = nearest_target["id"]
                e["level"] = nearest_target.get("level", 2) + 1
            else:
                e["parent_id"] = current_section_id
                e["level"] = current_section_level + 1
        elif t in ("list_item", "list_group"):
            parent = last_text_id or current_section_id
            e["parent_id"] = parent
            e["level"] = (last_text_level + 1) if last_text_id else (current_section_level + 1)
        elif t in ("diagram_group", "vector_diagram_group"):
            e["level"] = current_section_level + 1
            e["parent_id"] = current_section_id
        else:
            e["level"] = current_section_level + 1
            e["parent_id"] = current_section_id
            if t in ("text", "paragraph", "text_block"):
                last_text_id = e["id"]
                last_text_level = e["level"]

    # Link elements merged into diagrams to their diagram parent
    for dg in diagrams:
        diag_id = dg["id"]
        for mid in dg.get("merged_element_ids", []):
            for e in sorted_elements:
                if e["id"] == mid:
                    e["parent_id"] = diag_id
                    e["level"] = 3

    # Update elements list
    doc_ir["elements"] = sorted_elements

    # Step 5: Build Markdown content
    md_content = generate_markdown_content(doc_ir)

    return doc_ir, md_content


# ----------------------------------------------------------------------
# 4. Markdown Document Generator
# ----------------------------------------------------------------------
def generate_markdown_content(doc_ir: Dict[str, Any]) -> str:
    """Builds a rich GitHub-Flavored Markdown string from Unified JSON IR."""
    meta = doc_ir.get("metadata", {})
    doc_id = doc_ir.get("doc_id", "document")
    src_file = doc_ir.get("source_file", "")
    category = meta.get("category", "")
    parser = meta.get("parser", "")
    proc_time = meta.get("processing_time_sec", 0)

    lines = []
    lines.append(f"# {doc_id}\n")
    lines.append(f"> **Category:** {category}  ")
    lines.append(f"> **Source File:** `{src_file}`  ")
    lines.append(f"> **Parser:** {parser} | **Processing Time:** {proc_time}s\n")
    lines.append("---\n")

    rendered_table_ids = set()

    for e in doc_ir.get("elements", []):
        t = e.get("type", "text")
        text = e.get("text", "").strip()
        level = e.get("level", 2)
        parent_id = e.get("parent_id")

        indent = "  " if parent_id and str(parent_id).startswith("diag_") else ""

        if t == "title":
            lines.append(f"# {text}\n")
        elif t in ("section_header", "header", "page_header"):
            h_prefix = "#" * min(6, max(2, level))
            lines.append(f"\n{h_prefix} {text}\n")
        elif t in ("list_item", "list_group"):
            lines.append(f"{indent}- {text}")
        elif t == "table":
            tbl_id = e.get("table_id") or e.get("metadata", {}).get("table_id")
            tbl = next((tb for tb in doc_ir.get("tables", []) if tb.get("id") == tbl_id), None)
            if tbl and tbl.get("cells"):
                tbl_md = render_markdown_table_from_cells(
                    tbl["cells"], tbl.get("num_rows", 0), tbl.get("num_cols", 0)
                )
                lines.append(f"\n{tbl_md}\n")
            elif tbl and tbl.get("html"):
                lines.append(f"\n{tbl['html']}\n")
            else:
                lines.append(f"\n*{text}*\n")

            cap_text = tbl.get("metadata", {}).get("caption_text", "") if tbl else ""
            if cap_text:
                lines.append(f"\n*{cap_text}*\n")
            if tbl_id:
                rendered_table_ids.add(tbl_id)
        elif t == "caption":
            # If caption was already attached to a table, skip re-printing
            is_rendered = any(
                tb.get("metadata", {}).get("caption_id") == e["id"] and tb.get("id") in rendered_table_ids
                for tb in doc_ir.get("tables", [])
            )
            if not is_rendered:
                lines.append(f"\n*{text}*\n")
        elif t in ("picture", "figure"):
            cap = f" *(Figure: {text})*" if text else ""
            lines.append(f"\n![{e['id']}]({src_file}){cap}\n")
        elif t == "formula":
            lines.append(f"\n$$\n{text}\n$$\n")
        else:
            if text:
                lines.append(f"\n{indent}{text}\n")

    # If diagrams exist (e.g. 02_diagram_grouping)
    if doc_ir.get("diagrams"):
        lines.append("\n---\n")
        lines.append("## Consolidated Diagram Groups\n")
        for dg in doc_ir["diagrams"]:
            d_id = dg["id"]
            crop_img = dg.get("cropped_image", "")
            summary = dg.get("text_summary", "")
            merged_ids = ", ".join(dg.get("merged_element_ids", []))
            lines.append(f"### Diagram Group `{d_id}`\n")
            if crop_img:
                lines.append(f"![Diagram {d_id}]({crop_img})\n")
            lines.append(f"- **Merged Elements:** `{merged_ids}`")
            lines.append(f"- **Bounding Box:** `{dg.get('bbox')}`")
            if summary:
                lines.append(f"- **Summary Text:** {summary}\n")

            # Flowchart node/edge rendering in Mermaid
            if dg.get("nodes") and dg.get("edges"):
                lines.append("\n```mermaid\ngraph TD")
                for nd in dg["nodes"]:
                    sub = f" ({nd['subtitle']})" if nd.get("subtitle") else ""
                    lines.append(f"  {nd['id']}[\"{nd['label']}{sub}\"]")
                for ed in dg["edges"]:
                    lbl = f"|{ed['label']}| " if ed.get("label") else ""
                    lines.append(f"  {ed['src']} --> {lbl}{ed['dst']}")
                lines.append("```\n")

    return "\n".join(lines).strip() + "\n"


# ----------------------------------------------------------------------
# 5. Image Re-drawing Helper
# ----------------------------------------------------------------------
def redraw_overlay_and_comparison(
    image_path: str,
    gt_elements: List[Dict[str, Any]],
    docling_elements: List[Dict[str, Any]],
    overlay_path: str,
    compare_path: str,
    title_overlay: str = "Docling Sorted Predictions",
):
    """Redraws the overlay and side-by-side comparison with the updated reading order."""
    base_img = Image.open(image_path).convert("RGB")
    w, h = base_img.size

    font_size = max(14, int(w / 75))
    font = load_font(font_size)
    header_font = load_font(max(20, int(w / 45)))
    banner_h = font_size * 2 + 10

    def draw_elements(img, elems, title, default_clr=None):
        out = img.copy()
        draw = ImageDraw.Draw(out)
        if title:
            draw.rectangle([0, 0, w, banner_h], fill=(24, 28, 36))
            draw.text((15, 8), title, fill=(255, 255, 255), font=header_font)

        for el in elems:
            b = el.get("bbox")
            if not b or len(b) < 4:
                continue
            lbl = el.get("type") or el.get("label", "unknown")
            order = el.get("reading_order") if el.get("reading_order") is not None else el.get("order")
            color = default_clr or get_color(lbl)

            draw.rectangle(b, outline=color, width=3)
            tag = f"#{order} {lbl}" if order is not None else lbl
            tb = draw.textbbox((b[0], b[1]), tag, font=font)
            tb_w = tb[2] - tb[0]
            tb_h = tb[3] - tb[1]
            tag_ymin = max(0, b[1] - tb_h - 4)
            tag_ymax = tag_ymin + tb_h + 4
            tag_xmax = min(w, b[0] + tb_w + 8)

            draw.rectangle([b[0], tag_ymin, tag_xmax, tag_ymax], fill=color)
            draw.text((b[0] + 4, tag_ymin + 1), tag, fill=(255, 255, 255), font=font)

        return out

    # Draw overlay
    overlay_img = draw_elements(base_img, docling_elements, f"{title_overlay} ({len(docling_elements)} blocks)")
    for attempt in range(3):
        try:
            overlay_img.save(overlay_path, "PNG")
            break
        except Exception:
            time.sleep(0.4)

    # Draw comparison
    gt_vis = draw_elements(base_img, gt_elements, f"Ground Truth ({len(gt_elements)} blocks)")
    doc_vis = draw_elements(base_img, docling_elements, f"Docling Predictions ({len(docling_elements)} blocks)")
    sep_w = 8
    comp_img = Image.new("RGB", (w * 2 + sep_w, h), color=(200, 200, 200))
    comp_img.paste(gt_vis, (0, 0))
    comp_img.paste(doc_vis, (w + sep_w, 0))

    for attempt in range(3):
        try:
            comp_img.save(compare_path, "PNG")
            break
        except Exception:
            time.sleep(0.4)


# ----------------------------------------------------------------------
# Standalone Execution: Process all existing files
# ----------------------------------------------------------------------
def main():
    root_dir = Path("C:/Workspace/DocuMind")
    bmark_dir = root_dir / "outputs/01_docling_benchmark"
    diag_dir = root_dir / "outputs/02_diagram_grouping"
    gt_json_path = root_dir / "datasets/OmniDocBench/OmniDocBench.json"
    images_dir = root_dir / "datasets/OmniDocBench/images"

    # Load GT index if exists
    gt_by_filename = {}
    if gt_json_path.exists():
        with open(gt_json_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                p_info = item.get("page_info", {})
                if p_info.get("image_path"):
                    gt_by_filename[p_info["image_path"]] = item

    print("================================================================================")
    print(" EXECUTING UNIFIED IR POSTPROCESSING & MARKDOWN GENERATION")
    print("================================================================================")

    # 1. Process 01_docling_benchmark
    print("\n[1/2] Processing outputs/01_docling_benchmark/...")
    bmark_files = [
        "01_academic_2col.json",
        "02_ppt_slide.json",
        "03_merged_table.json",
        "04_financial_report.json",
    ]

    for fn in bmark_files:
        json_path = bmark_dir / fn
        if not json_path.exists():
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            doc_ir = json.load(f)

        src_fn = doc_ir.get("source_file", "")
        img_path = images_dir / src_fn

        # Postprocess IR
        doc_ir, md_content = postprocess_ir_structure(doc_ir)

        # Save updated JSON IR
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(doc_ir, f, indent=2, ensure_ascii=False)
        print(f"  [+] Updated JSON IR: {json_path}")

        # Save Markdown file (.md)
        md_path = bmark_dir / f"{json_path.stem}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"  [+] Generated Markdown: {md_path}")

        # Redraw visualizations with updated reading order
        if img_path.exists():
            gt_item = gt_by_filename.get(src_fn, {})
            gt_elements = []
            for d_idx, d in enumerate(gt_item.get("layout_dets", [])):
                poly = d.get("poly", [])
                bbox = None
                if len(poly) >= 8:
                    xs = poly[0::2]
                    ys = poly[1::2]
                    bbox = [min(xs), min(ys), max(xs), max(ys)]
                gt_elements.append({
                    "order": d.get("order", d_idx),
                    "label": d.get("category_type", "unknown"),
                    "bbox": bbox,
                })

            overlay_p = bmark_dir / f"{json_path.stem}_overlay.png"
            compare_p = bmark_dir / f"{json_path.stem}_compare.png"
            redraw_overlay_and_comparison(
                str(img_path),
                gt_elements,
                doc_ir["elements"],
                str(overlay_p),
                str(compare_p),
                title_overlay=f"Docling Sorted Predictions: {json_path.stem}",
            )
            print(f"  [+] Re-rendered overlays: {overlay_p.name}, {compare_p.name}")

    # 2. Process 02_diagram_grouping
    print("\n[2/2] Processing outputs/02_diagram_grouping/...")
    diag_files = [
        "01_ppt_cards.json",
        "02_network_topology.json",
        "03_healthcare_infographic.json",
        "04_vector_flowchart.json",
    ]

    for fn in diag_files:
        json_path = diag_dir / fn
        if not json_path.exists():
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            doc_ir = json.load(f)

        # Postprocess IR
        doc_ir, md_content = postprocess_ir_structure(doc_ir)

        # Save updated JSON IR
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(doc_ir, f, indent=2, ensure_ascii=False)
        print(f"  [+] Updated JSON IR: {json_path}")

        # Save Markdown file (.md)
        md_path = diag_dir / f"{json_path.stem}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"  [+] Generated Markdown: {md_path}")

    print("\n================================================================================")
    print(" ALL POSTPROCESSING AND MARKDOWN GENERATION COMPLETED SUCCESSFULLY!")
    print("================================================================================")


if __name__ == "__main__":
    main()
