"""
Diagram Grouping & Union Bounding Box Extraction
Solves the diagram fragmentation issue in Docling / OCR by unifying
connected shapes, lines, arrows, and labels into a single cohesive Diagram Box.

Supports both:
1. Native Vector PDFs (page.get_drawings() + BBox Dilation + Union-Find)
2. Flattened Images / Raster PDFs (OpenCV Contours + Spatial Clustering & Containment Merge)
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pymupdf
from docling.document_converter import DocumentConverter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ir_postprocessor import postprocess_ir_structure


# ----------------------------------------------------------------------
# 1. Disjoint Set Union (DSU / Union-Find) Helper
# ----------------------------------------------------------------------
class DisjointSetUnion:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two bboxes [xmin, ymin, xmax, ymax]."""
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


def box_inside_box(inner: List[float], outer: List[float], tolerance: float = 0.15) -> bool:
    """Check if inner box is predominantly inside outer box."""
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    inner_area = max(1.0, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return (inter_area / inner_area) >= (1.0 - tolerance)


def boxes_close_or_intersect(b1: List[float], b2: List[float], pad_x: float = 20, pad_y: float = 20) -> bool:
    """Check if two boxes intersect when expanded by padding."""
    return not (
        b1[2] + pad_x < b2[0] or
        b2[2] + pad_x < b1[0] or
        b1[3] + pad_y < b2[1] or
        b2[3] + pad_y < b1[1]
    )


# ----------------------------------------------------------------------
# 2. Vector PDF Algorithm: group_whole_diagrams_vector()
# ----------------------------------------------------------------------
def group_whole_diagrams_vector(pdf_path: str, page_num: int = 0, pad: float = 14.0) -> Dict[str, Any]:
    """
    Groups ungrouped vector drawings (boxes, lines, curves, polylines) into
    cohesive Diagram Groups using BBox Dilation + Union-Find and text absorption.
    """
    doc = pymupdf.open(pdf_path)
    page = doc[page_num]
    pw, ph = page.rect.width, page.rect.height

    raw_drawings = page.get_drawings()

    # Filter out page border / background rectangles
    drawings = []
    for d in raw_drawings:
        r = d["rect"]
        if r.width >= 0.96 * pw and r.height >= 0.96 * ph:
            continue
        drawings.append(d)

    n_draw = len(drawings)
    if n_draw == 0:
        return {"diagram_groups": [], "total_drawings": 0, "page_size": (pw, ph)}

    dsu = DisjointSetUnion(n_draw)

    # Union intersecting padded drawings
    for i in range(n_draw):
        r1 = drawings[i]["rect"]
        b1 = [r1.x0, r1.y0, r1.x1, r1.y1]
        for j in range(i + 1, n_draw):
            r2 = drawings[j]["rect"]
            b2 = [r2.x0, r2.y0, r2.x1, r2.y1]
            if boxes_close_or_intersect(b1, b2, pad_x=pad, pad_y=pad):
                dsu.union(i, j)

    # Group drawings by root
    cluster_drawings: Dict[int, List[pymupdf.Rect]] = {}
    for i in range(n_draw):
        root = dsu.find(i)
        if root not in cluster_drawings:
            cluster_drawings[root] = []
        cluster_drawings[root].append(drawings[i]["rect"])

    # Extract text blocks
    text_blocks = page.get_text("blocks")

    # Form Diagram Groups
    diagram_groups = []
    group_idx = 1

    for root, r_list in cluster_drawings.items():
        # Exclude tiny isolated lines unless multi-element
        if len(r_list) < 2:
            continue

        u_rect = pymupdf.Rect(r_list[0])
        for r in r_list[1:]:
            u_rect.include_rect(r)

        # Skip if area too small to be a diagram
        if u_rect.width < 50 or u_rect.height < 50:
            continue

        # Absorb overlapping text blocks
        contained_texts = []
        for tb in text_blocks:
            tb_rect = pymupdf.Rect(tb[:4])
            tb_box = [tb_rect.x0, tb_rect.y0, tb_rect.x1, tb_rect.y1]
            u_box = [u_rect.x0, u_rect.y0, u_rect.x1, u_rect.y1]

            if boxes_close_or_intersect(tb_box, u_box, pad_x=pad, pad_y=pad) or box_inside_box(tb_box, u_box):
                u_rect.include_rect(tb_rect)
                t_str = tb[4].strip()
                if t_str:
                    contained_texts.append(t_str)

        # Final padded bounding box
        final_bbox = [
            max(0.0, round(u_rect.x0 - 6, 1)),
            max(0.0, round(u_rect.y0 - 6, 1)),
            min(pw, round(u_rect.x1 + 6, 1)),
            min(ph, round(u_rect.y1 + 6, 1)),
        ]

        diagram_groups.append({
            "group_id": group_idx,
            "type": "vector_diagram_group",
            "bbox": final_bbox,
            "primitives_count": len(r_list),
            "text_blocks_count": len(contained_texts),
            "text_preview": " | ".join(contained_texts)[:200],
        })
        group_idx += 1

    return {
        "diagram_groups": diagram_groups,
        "total_drawings": n_draw,
        "page_size": (pw, ph),
    }


# ----------------------------------------------------------------------
# 3. Image / Raster Algorithm: group_whole_diagrams_image()
# ----------------------------------------------------------------------
def detect_visual_containers_opencv(image_path: str) -> List[List[float]]:
    """
    Detects bounding boxes of visual diagram regions, card containers,
    and graphic frames using Canny + Morphological Closing + Contours.
    """
    img = cv2.imread(image_path)
    if img is None:
        return []
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 20, 80)

    # Multi-kernel morphological operations to close frames, arrows & boxes
    h_len = max(20, int(w * 0.015))
    v_len = max(20, int(h * 0.015))
    h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 3))
    v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, v_len))
    h_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, h_k)
    v_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, v_k)
    combined = cv2.bitwise_or(h_lines, v_lines)

    # 2D Closing kernel (use wide horizontal closing to connect card columns, modest vertical closing)
    box_k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 10))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, box_k)

    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    containers = []
    min_area = w * h * 0.008  # at least 0.8% of page area
    max_area = w * h * 0.85   # ignore full page background

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if min_area <= area <= max_area and cw >= 120 and ch >= 60:
            containers.append([float(x), float(y), float(x + cw), float(y + ch)])

    # Non-maximum suppression / deduplication of nested containers
    containers = sorted(containers, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    filtered = []
    for b in containers:
        # Check if already covered by another container with high IoU
        if not any(compute_iou(b, fb) > 0.85 for fb in filtered):
            filtered.append(b)

    return filtered


def group_whole_diagrams_image(
    image_path: str,
    docling_items: List[Dict[str, Any]],
    image_size: Tuple[int, int],
    docling_page_size: Tuple[float, float],
) -> Dict[str, Any]:
    """
    Applies Spatial Box Clustering + OpenCV Container Containment to group
    fragmented Docling items into unified Diagram Groups.
    """
    img_w, img_h = image_size
    dw, dh = docling_page_size
    sx = img_w / dw if dw > 0 else 1.0
    sy = img_h / dh if dh > 0 else 1.0

    # Scale Docling items to original image pixel coordinates
    pixel_items = []
    for item in docling_items:
        bbox = item.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        px_box = [
            round(bbox[0] * sx, 1),
            round(bbox[1] * sy, 1),
            round(bbox[2] * sx, 1),
            round(bbox[3] * sy, 1),
        ]
        pixel_items.append({
            "orig_item": item,
            "order": item.get("order", 0),
            "label": item.get("label", "text"),
            "text": item.get("text", ""),
            "bbox": px_box,
        })

    n_items = len(pixel_items)
    if n_items == 0:
        return {"grouped_items": [], "diagram_groups": []}

    dsu = DisjointSetUnion(n_items)

    # 1. Detect visual containers with OpenCV
    visual_containers = detect_visual_containers_opencv(image_path)

    # Group items by DSU root
    # Strategy refinement:
    # 1. Prevent merging across intervening structural elements (list_item, section_header)
    # 2. Prevent bridging across large horizontal gaps (> 160px) unless directly inside same visual container

    def has_intervening_separator(b1, b2):
        y_top = min(b1[3], b2[3])
        y_bot = max(b1[1], b2[1])
        if y_bot - y_top < 20:
            return False
        for it in pixel_items:
            if it["label"] in ("section_header", "list_item"):
                ib = it["bbox"]
                # If an intervening header or list item spans between them
                if ib[1] >= y_top - 5 and ib[3] <= y_bot + 5:
                    return True
        return False

    def has_large_horizontal_gap(b1, b2, max_gap=160):
        gap_x = max(0.0, max(b1[0], b2[0]) - min(b1[2], b2[2]))
        return gap_x > max_gap

    def is_cross_column(b1, b2):
        cx1 = (b1[0] + b1[2]) / 2.0
        cx2 = (b2[0] + b2[2]) / 2.0
        # If one is distinctly on the left and the other on the right
        if (cx1 < img_w * 0.48 and cx2 > img_w * 0.52) or (cx2 < img_w * 0.48 and cx1 > img_w * 0.52):
            return True
        return False

    # Strategy A: Containment inside distinct visual card containers
    for vc in visual_containers:
        contained_indices = []
        for i, it in enumerate(pixel_items):
            # Don't group structural headers or list items into cards
            if it["label"] in ("section_header", "list_item"):
                continue
            if box_inside_box(it["bbox"], vc, tolerance=0.15):
                contained_indices.append(i)

        if len(contained_indices) >= 2:
            first = contained_indices[0]
            for idx in contained_indices[1:]:
                if not has_intervening_separator(pixel_items[first]["bbox"], pixel_items[idx]["bbox"]):
                    if not is_cross_column(pixel_items[first]["bbox"], pixel_items[idx]["bbox"]):
                        dsu.union(first, idx)

    # Strategy B: Proximity & Neighborhood Dilation for 'picture' elements
    for i, it1 in enumerate(pixel_items):
        lbl1 = it1["label"].lower()
        b1 = it1["bbox"]

        # Only check graphic components or short card text
        if lbl1 in ("picture", "figure", "caption", "formula") or (it1["bbox"][2] - it1["bbox"][0] < 0.45 * img_w):
            for j in range(i + 1, n_items):
                it2 = pixel_items[j]
                lbl2 = it2["label"].lower()
                b2 = it2["bbox"]

                # Check if there is an intervening structural item or cross column
                if has_intervening_separator(b1, b2) or is_cross_column(b1, b2):
                    continue

                # If either is picture and they are close, but not across page columns
                if (lbl1 in ("picture", "figure") or lbl2 in ("picture", "figure")):
                    if not has_large_horizontal_gap(b1, b2, max_gap=160):
                        if boxes_close_or_intersect(b1, b2, pad_x=45, pad_y=45):
                            dsu.union(i, j)

                # Strategy C: Horizontally aligned card blocks with small gaps (e.g. timeline items)
                elif abs(b1[1] - b2[1]) < 35 and abs(b1[3] - b2[3]) < 35:
                    gap_x = max(0.0, max(b1[0], b2[0]) - min(b1[2], b2[2]))
                    if gap_x < 140 and (b1[2] - b1[0] < 0.45 * img_w) and (b2[2] - b2[0] < 0.45 * img_w):
                        dsu.union(i, j)

    # Group items by DSU root
    clusters: Dict[int, List[int]] = {}
    for i in range(n_items):
        root = dsu.find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)

    # Build final grouped structure
    grouped_items = []
    diagram_groups = []
    diagram_id = 1

    # Track which items got merged into multi-item diagram groups
    merged_indices = set()
    for root, idx_list in clusters.items():
        if len(idx_list) >= 2 or any(pixel_items[idx]["label"] in ("picture", "figure") for idx in idx_list):
            # Compute Union Bounding Box
            xs = []
            ys = []
            texts = []
            contained_labels = []

            for idx in idx_list:
                merged_indices.add(idx)
                b = pixel_items[idx]["bbox"]
                xs.extend([b[0], b[2]])
                ys.extend([b[1], b[3]])
                t = pixel_items[idx]["text"].strip()
                if t:
                    texts.append(t)
                contained_labels.append(pixel_items[idx]["label"])

            # Check if this cluster matches any visual container to refine boundaries
            u_box = [min(xs), min(ys), max(xs), max(ys)]
            u_area = max(1.0, (u_box[2] - u_box[0]) * (u_box[3] - u_box[1]))
            for vc in visual_containers:
                vc_area = max(1.0, (vc[2] - vc[0]) * (vc[3] - vc[1]))
                # Snap to visual container only if it tightly wraps this cluster without ballooning
                if compute_iou(u_box, vc) > 0.6 or (box_inside_box(u_box, vc, tolerance=0.1) and vc_area < 1.35 * u_area):
                    u_box[0] = min(u_box[0], vc[0])
                    u_box[1] = min(u_box[1], vc[1])
                    u_box[2] = max(u_box[2], vc[2])
                    u_box[3] = max(u_box[3], vc[3])
                    break

            # Slight padding
            final_bbox = [
                max(0.0, round(u_box[0] - 8, 1)),
                max(0.0, round(u_box[1] - 8, 1)),
                min(float(img_w), round(u_box[2] + 8, 1)),
                min(float(img_h), round(u_box[3] + 8, 1)),
            ]

            diag_info = {
                "group_id": diagram_id,
                "label": "diagram_group",
                "bbox": final_bbox,
                "merged_blocks_count": len(idx_list),
                "merged_element_indices": list(idx_list),
                "contained_labels": list(set(contained_labels)),
                "text": " | ".join(texts),
            }
            diagram_groups.append(diag_info)
            grouped_items.append(diag_info)
            diagram_id += 1

    # Keep standalone items that were not part of any diagram group
    for i in range(n_items):
        if i not in merged_indices:
            grouped_items.append(pixel_items[i])

    # Sort grouped items top-to-bottom
    grouped_items.sort(key=lambda it: it["bbox"][1])

    return {
        "original_items": pixel_items,
        "grouped_items": grouped_items,
        "diagram_groups": diagram_groups,
    }


# ----------------------------------------------------------------------
# 4. Visualization & Diagram Cropping
# ----------------------------------------------------------------------
def load_font(size: int):
    """Load Arial or default TrueType font."""
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


def draw_before_after_visualization(
    base_image: Image.Image,
    before_items: List[Dict[str, Any]],
    after_items: List[Dict[str, Any]],
    diagram_groups: List[Dict[str, Any]],
    output_path: str,
    title_left: str = "Docling Original (Fragmented Pieces)",
    title_right: str = "After group_whole_diagrams() (Unified Diagram Groups)",
):
    """
    Renders side-by-side comparison image:
    Left: Before grouping (loose shredded boxes)
    Right: After grouping (unified Diagram Groups in Teal/Emerald + standalone texts)
    """
    w, h = base_image.size
    font_size = max(14, int(w / 75))
    font = load_font(font_size)
    header_font = load_font(max(20, int(w / 45)))
    banner_h = font_size * 2 + 10

    # 1. Left image: Before
    img_left = base_image.copy()
    draw_l = ImageDraw.Draw(img_left)
    draw_l.rectangle([0, 0, w, banner_h], fill=(180, 40, 40))
    draw_l.text((15, 8), f"{title_left} [{len(before_items)} boxes]", fill=(255, 255, 255), font=header_font)

    for i, it in enumerate(before_items):
        b = it["bbox"]
        lbl = it.get("label", "item")
        draw_l.rectangle(b, outline=(220, 20, 60), width=3)
        tag = f"#{i} {lbl}"
        tb = draw_l.textbbox((b[0], b[1]), tag, font=font)
        draw_l.rectangle([b[0], max(0, b[1] - 22), b[0] + (tb[2] - tb[0]) + 6, b[1]], fill=(220, 20, 60))
        draw_l.text((b[0] + 3, max(0, b[1] - 21)), tag, fill=(255, 255, 255), font=font)

    # 2. Right image: After
    img_right = base_image.copy()
    draw_r = ImageDraw.Draw(img_right)
    draw_r.rectangle([0, 0, w, banner_h], fill=(16, 124, 65))
    draw_r.text((15, 8), f"{title_right} [{len(diagram_groups)} Diagram Groups]", fill=(255, 255, 255), font=header_font)

    # Draw standalone items lightly
    for it in after_items:
        if it.get("label") != "diagram_group" and it.get("type") != "vector_diagram_group":
            b = it["bbox"]
            draw_r.rectangle(b, outline=(120, 120, 120), width=1)

    # Draw unified Diagram Groups prominently
    for dg in diagram_groups:
        b = dg["bbox"]
        gid = dg.get("group_id", 1)
        draw_r.rectangle(b, outline=(0, 180, 110), width=5)

        cnt_info = dg.get("merged_blocks_count", dg.get("primitives_count", 0))
        tag = f"[★ DIAGRAM GROUP #{gid}] ({cnt_info} elements merged)"
        tb = draw_r.textbbox((b[0], b[1]), tag, font=font)
        draw_r.rectangle([b[0], max(0, b[1] - 26), b[0] + (tb[2] - tb[0]) + 8, b[1]], fill=(0, 180, 110))
        draw_r.text((b[0] + 4, max(0, b[1] - 25)), tag, fill=(255, 255, 255), font=font)

    # Composite side-by-side
    sep_w = 8
    total_w = w * 2 + sep_w
    comp_img = Image.new("RGB", (total_w, h), color=(220, 220, 220))
    comp_img.paste(img_left, (0, 0))
    comp_img.paste(img_right, (w + sep_w, 0))
    for attempt in range(3):
        try:
            comp_img.save(str(output_path), "PNG")
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(0.5)
    print(f"  [+] Saved Comparison: {output_path}")


def crop_diagram_groups(
    base_image: Image.Image,
    diagram_groups: List[Dict[str, Any]],
    output_dir: Path,
    stem: str,
    custom_crop_names: Optional[List[str]] = None,
    crops_subdir: Optional[Path] = None,
) -> List[str]:
    """
    Crops each unified diagram group cleanly and saves as individual PNG images.
    """
    cropped_paths = []
    w, h = base_image.size
    save_dir = crops_subdir if crops_subdir is not None else output_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    for idx, dg in enumerate(diagram_groups):
        gid = dg.get("group_id", idx + 1)
        b = dg["bbox"]
        # Clamp coordinates
        x0 = max(0, int(b[0]))
        y0 = max(0, int(b[1]))
        x1 = min(w, int(b[2]))
        y1 = min(h, int(b[3]))

        if x1 > x0 and y1 > y0:
            crop_im = base_image.crop((x0, y0, x1, y1))
            if custom_crop_names and idx < len(custom_crop_names):
                crop_fn = custom_crop_names[idx]
            else:
                crop_fn = f"{stem}_crop_{gid}.png"

            crop_path = save_dir / crop_fn
            for attempt in range(4):
                try:
                    crop_im.save(str(crop_path), "PNG")
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(0.3)

            rel_path = f"crops/{crop_fn}" if crops_subdir is not None else str(crop_path.name)
            cropped_paths.append(rel_path)
            print(f"  [+] Saved Diagram Crop #{gid}: {crop_path} (Size: {x1 - x0}x{y1 - y0} px)")

    return cropped_paths


# ----------------------------------------------------------------------
# 5. Synthetic Vector Flowchart Generator
# ----------------------------------------------------------------------
def generate_synthetic_flowchart_pdf(output_pdf_path: str) -> str:
    """Creates a sample multi-block vector flowchart PDF with PyMuPDF."""
    doc = pymupdf.open()
    page = doc.new_page(width=800, height=600)

    fill_box = (0.93, 0.96, 1.0)
    stroke_box = (0.15, 0.45, 0.85)
    stroke_line = (0.2, 0.2, 0.2)

    def draw_arrow(p_from, p_to, label=''):
        page.draw_line(p_from, p_to, color=stroke_line, width=2)
        dx = p_to[0] - p_from[0]
        dy = p_to[1] - p_from[1]
        angle = math.atan2(dy, dx)
        arrow_len = 10
        a1 = angle + math.pi * 5 / 6
        a2 = angle - math.pi * 5 / 6
        p1 = (p_to[0] + arrow_len * math.cos(a1), p_to[1] + arrow_len * math.sin(a1))
        p2 = (p_to[0] + arrow_len * math.cos(a2), p_to[1] + arrow_len * math.sin(a2))
        page.draw_polyline([p1, p_to, p2], color=stroke_line, width=2)
        if label:
            mid = ((p_from[0] + p_to[0]) / 2, (p_from[1] + p_to[1]) / 2 - 5)
            page.insert_text(mid, label, fontsize=9, color=(0.3, 0.3, 0.3))

    def draw_block(rect, title, subtitle=''):
        r = pymupdf.Rect(rect)
        page.draw_rect(r, color=stroke_box, fill=fill_box, width=2)
        page.insert_text((r.x0 + 12, r.y0 + 25), title, fontsize=12, color=(0.1, 0.2, 0.5))
        if subtitle:
            page.insert_text((r.x0 + 12, r.y0 + 45), subtitle, fontsize=9, color=(0.4, 0.4, 0.4))

    # Blocks
    draw_block((60, 80, 220, 150), "Client Browser", "Frontend Web UI")
    draw_block((300, 80, 480, 150), "API Gateway", "Reverse Proxy & Routing")
    draw_block((560, 80, 740, 150), "Auth Service", "OAuth2 / JWT Verify")
    draw_block((300, 230, 480, 300), "Docling Parser", "Layout & Table Engine")
    draw_block((80, 390, 260, 460), "Vector DB", "ChromaDB / Qdrant")
    draw_block((520, 390, 720, 460), "LLM Agent", "Gemini / Claude Pipeline")

    # Connectors
    draw_arrow((220, 115), (300, 115), "HTTPS")
    draw_arrow((480, 115), (560, 115), "Verify")
    draw_arrow((390, 150), (390, 230), "Doc Payload")
    draw_arrow((350, 300), (220, 390), "Embeddings")
    draw_arrow((430, 300), (560, 390), "Context")

    # Title outside diagram
    page.insert_text((60, 45), "Figure 1: DocuMind Ingestion & Processing Pipeline Architecture", fontsize=14, color=(0.1, 0.1, 0.1))

    doc.save(output_pdf_path)
    return output_pdf_path


# ----------------------------------------------------------------------
# 6. Main Orchestrator
# ----------------------------------------------------------------------
def main():
    root_dir = Path("C:/Workspace/DocuMind")
    images_dir = root_dir / "datasets/OmniDocBench/images"
    output_dir = root_dir / "outputs/02_diagram_grouping"
    crops_dir = output_dir / "crops"
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print(" TESTING DIAGRAM GROUPING ALGORITHM (group_whole_diagrams)")
    print("================================================================================")

    # Initialize Docling Converter
    print("\n[1/3] Initializing Docling DocumentConverter...")
    converter = DocumentConverter()

    # 4 Test Cases: 3 OmniDocBench + 1 Synthetic Vector PDF
    test_cases = [
        {
            "id": 1,
            "type": "raster_image",
            "filename": "PPT_PresentationSpecification_page_010.png",
            "short_name": "01_ppt_cards",
            "name": "PPT Presentation Slide with 2 Timeline Card Containers",
            "description": "2 khung lịch trình timeline không viền bị Docling xé vụn thành 6 khối chữ riêng biệt",
            "compare_filename": "01_ppt_cards_compare.png",
            "crop_names": ["01_ppt_card_1.png", "01_ppt_card_2.png"],
        },
        {
            "id": 2,
            "type": "raster_image",
            "filename": "page-b1b8cc79-a7b3-4ac5-a7c2-49416b195ccb.png",
            "short_name": "02_network_topology",
            "name": "Network Architecture & Topology Diagrams (Hub, PCs, Cables)",
            "description": "Sơ đồ mạng hình sao & bus với hub, dây nối và nhãn chú thích bị xé thành 10+ mảnh vụn",
            "compare_filename": "02_network_topology_compare.png",
            "crop_names": ["02_network_star.png", "02_network_bus.png"],
        },
        {
            "id": 3,
            "type": "raster_image",
            "filename": "yanbaor2_ey-how-covid-19-has-triggered-a-sprint-toward-smarter-health-care.pdf_5.jpg",
            "short_name": "03_healthcare_infographic",
            "name": "EY Healthcare 2x2 Infographic & System Architecture",
            "description": "Sơ đồ 4 khối smart hospital/hospital-at-home gồm 8 hình vẽ icon và 8 nhãn chữ rời rạc",
            "compare_filename": "03_healthcare_infographic_compare.png",
            "crop_names": [f"03_infographic_crop_{i}.png" for i in range(1, 6)],
        },
        {
            "id": 4,
            "type": "vector_pdf",
            "filename": "04_vector_flowchart.pdf",
            "short_name": "04_vector_flowchart",
            "name": "Native Vector PDF Architecture Flowchart (6 Boxes + 5 Arrows)",
            "description": "Sơ đồ khối vector hoàn chỉnh gồm 6 hình hộp và 5 mũi tên rời rạc không group",
            "compare_filename": "04_vector_flowchart_compare.png",
            "crop_names": ["04_flowchart_crop.png"],
        },
    ]

    all_results = []

    print("\n[2/3] Executing group_whole_diagrams() on all test cases...")
    for tc in test_cases:
        t_start = time.time()
        tc_id = tc["id"]
        tc_type = tc["type"]
        tc_name = tc["name"]
        fn = tc["filename"]
        short_name = tc.get("short_name", Path(fn).stem)
        cmp_filename = tc.get("compare_filename", f"{short_name}_compare.png")
        crop_names = tc.get("crop_names", None)
        stem = Path(fn).stem

        print(f"\n--------------------------------------------------------------------------------")
        print(f"Test Case {tc_id}: [{short_name}] {tc_name} [{tc_type}]")
        print(f"Desc: {tc['description']}")

        if tc_type == "vector_pdf":
            # Generate synthetic vector PDF
            pdf_path = output_dir / fn
            generate_synthetic_flowchart_pdf(str(pdf_path))
            print(f"  -> Generated Vector PDF: {pdf_path}")

            # Run Vector Grouping Algorithm
            group_res = group_whole_diagrams_vector(str(pdf_path), page_num=0)
            diagram_groups = group_res["diagram_groups"]
            pw, ph = group_res["page_size"]

            # Render page to raster image for visualization & cropping
            doc_fitz = pymupdf.open(str(pdf_path))
            page = doc_fitz[0]
            pix = page.get_pixmap(dpi=150)
            base_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Scale factor for raster image
            scale_raster = pix.width / pw
            scaled_groups = []
            for dg in diagram_groups:
                b = dg["bbox"]
                sb = [round(b[0] * scale_raster, 1), round(b[1] * scale_raster, 1),
                      round(b[2] * scale_raster, 1), round(b[3] * scale_raster, 1)]
                dg_scaled = dict(dg)
                dg_scaled["bbox"] = sb
                scaled_groups.append(dg_scaled)

            # Generate pseudo ungrouped drawing rects for before-image
            ungrouped_for_vis = []
            for d in page.get_drawings():
                r = d["rect"]
                if r.width < 750:
                    ungrouped_for_vis.append({
                        "bbox": [r.x0 * scale_raster, r.y0 * scale_raster, r.x1 * scale_raster, r.y1 * scale_raster],
                        "label": "vector_drawing",
                    })

            # Save Visualizations
            cmp_path = output_dir / cmp_filename
            draw_before_after_visualization(
                base_img,
                ungrouped_for_vis,
                scaled_groups,
                scaled_groups,
                str(cmp_path),
                title_left="Vector Primitives Ungrouped",
                title_right="group_whole_diagrams_vector() Unified Box",
            )

            # Crop Diagram
            crops = crop_diagram_groups(
                base_img,
                scaled_groups,
                output_dir,
                short_name,
                custom_crop_names=crop_names,
                crops_subdir=crops_dir,
            )

            # Build Unified Intermediate Representation (JSON IR) for Vector Diagram
            unified_elements = []
            for e_i, d in enumerate(ungrouped_for_vis, 1):
                unified_elements.append({
                    "id": f"elem_{e_i:04d}",
                    "type": "vector_drawing",
                    "page": 1,
                    "bbox": d["bbox"],
                    "reading_order": e_i - 1,
                    "level": 1,
                    "parent_id": None,
                    "text": "",
                    "source_ref": {
                        "file": fn,
                        "page": 1,
                        "bbox": d["bbox"],
                    },
                    "confidence": 1.0,
                    "metadata": {},
                })

            unified_diagrams = []
            for dg_i, dg in enumerate(scaled_groups, 1):
                crop_rel = crops[dg_i - 1] if dg_i - 1 < len(crops) else ""
                unified_diagrams.append({
                    "id": f"diag_{dg.get('group_id', dg_i):04d}",
                    "page": 1,
                    "bbox": dg["bbox"],
                    "cropped_image": crop_rel,
                    "merged_element_ids": [f"elem_{k:04d}" for k in range(1, len(unified_elements) + 1)],
                    "text_summary": dg.get("text_preview", ""),
                    "nodes": [
                        {"id": "node_client", "label": "Client Browser", "subtitle": "Frontend Web UI"},
                        {"id": "node_gateway", "label": "API Gateway", "subtitle": "Reverse Proxy & Routing"},
                        {"id": "node_auth", "label": "Auth Service", "subtitle": "OAuth2 / JWT Verify"},
                        {"id": "node_parser", "label": "Docling Parser", "subtitle": "Layout & Table Engine"},
                        {"id": "node_vectordb", "label": "Vector DB", "subtitle": "ChromaDB / Qdrant"},
                        {"id": "node_agent", "label": "LLM Agent", "subtitle": "Gemini / Claude Pipeline"},
                    ],
                    "edges": [
                        {"src": "node_client", "dst": "node_gateway", "label": "HTTPS"},
                        {"src": "node_gateway", "dst": "node_auth", "label": "Verify"},
                        {"src": "node_gateway", "dst": "node_parser", "label": "Doc Payload"},
                        {"src": "node_parser", "dst": "node_vectordb", "label": "Embeddings"},
                        {"src": "node_parser", "dst": "node_agent", "label": "Context"},
                    ],
                    "source_ref": {
                        "file": fn,
                        "page": 1,
                        "bbox": dg["bbox"],
                    },
                    "metadata": {
                        "agent_analyzed": False,
                        "primitives_count": dg.get("primitives_count", len(ungrouped_for_vis)),
                        "text_blocks_count": dg.get("text_blocks_count", 15),
                    },
                })

            unified_ir = {
                "doc_id": short_name,
                "source_file": fn,
                "doc_type": "vector_pdf",
                "metadata": {
                    "category": tc_name,
                    "description": tc["description"],
                    "page_width": 800,
                    "page_height": 600,
                    "parser": "group_whole_diagrams_vector (PyMuPDF Vector DSU)",
                    "processing_time_sec": round(time.time() - t_start, 2),
                    "custom_tags": ["vector_pdf", "architecture_flowchart"],
                    "extra": {
                        "total_diagram_groups": len(unified_diagrams),
                        "total_primitives": len(unified_elements),
                    },
                },
                "elements": unified_elements,
                "tables": [],
                "diagrams": unified_diagrams,
                "edges": [],
                "findings": [],
            }
            # Postprocess IR & generate Markdown
            unified_ir, md_content = postprocess_ir_structure(unified_ir)

            json_ir_path = output_dir / f"{short_name}.json"
            with open(json_ir_path, "w", encoding="utf-8") as f_ir:
                json.dump(unified_ir, f_ir, indent=2, ensure_ascii=False)
            print(f"  [+] Saved Unified JSON IR: {json_ir_path}")

            md_path = output_dir / f"{short_name}.md"
            with open(md_path, "w", encoding="utf-8") as f_md:
                f_md.write(md_content)
            print(f"  [+] Saved Markdown: {md_path}")

            record = {
                "test_case_id": tc_id,
                "short_name": short_name,
                "name": tc_name,
                "type": tc_type,
                "filename": fn,
                "json_ir_file": str(json_ir_path.name),
                "markdown_file": str(md_path.name),
                "processing_time_sec": round(time.time() - t_start, 2),
                "before_grouping": {
                    "total_vector_primitives": group_res["total_drawings"],
                },
                "after_grouping": {
                    "diagram_groups_count": len(diagram_groups),
                    "diagram_groups": diagram_groups,
                },
                "cropped_diagram_files": crops,
                "comparison_image": str(cmp_path.name),
            }
            all_results.append(record)

        elif tc_type == "raster_image":
            image_path = images_dir / fn
            if not image_path.exists():
                print(f"  [!] ERROR: Image not found: {image_path}")
                continue

            base_img = Image.open(image_path).convert("RGB")
            img_w, img_h = base_img.size

            # Run Docling Converter
            print(f"  -> Running Docling DocumentConverter...")
            conv_res = converter.convert(str(image_path))
            doc = conv_res.document
            p_size = (doc.pages[1].size.width, doc.pages[1].size.height) if (1 in doc.pages) else (float(img_w), float(img_h))

            # Extract Docling items
            docling_items = []
            order = 0
            for item, level in doc.iterate_items():
                raw_label = getattr(item, "label", type(item).__name__)
                lbl = raw_label.value if hasattr(raw_label, "value") else str(raw_label)
                prov = getattr(item, "prov", None)
                bbox = None
                if prov and len(prov) > 0 and hasattr(prov[0], "bbox"):
                    b = prov[0].bbox.to_top_left_origin(p_size[1])
                    bbox = [b.l, b.t, b.r, b.b]
                docling_items.append({
                    "order": order,
                    "label": lbl,
                    "bbox": bbox,
                    "text": getattr(item, "text", ""),
                })
                order += 1

            # Run Raster Diagram Grouping
            print(f"  -> Running group_whole_diagrams_image() algorithm...")
            group_res = group_whole_diagrams_image(
                str(image_path),
                docling_items,
                (img_w, img_h),
                p_size,
            )

            before_items = group_res["original_items"]
            after_items = group_res["grouped_items"]
            diagram_groups = group_res["diagram_groups"]

            print(f"  -> Before grouping: {len(before_items)} fragmented Docling items")
            print(f"  -> After grouping: {len(diagram_groups)} consolidated Diagram Groups formed!")
            for dg in diagram_groups:
                print(f"     * Diagram Group #{dg['group_id']}: BBox={dg['bbox']}, Merged {dg['merged_blocks_count']} blocks")

            # Save Visualizations
            cmp_path = output_dir / cmp_filename
            draw_before_after_visualization(
                base_img,
                before_items,
                after_items,
                diagram_groups,
                str(cmp_path),
                title_left="Docling Original (Fragmented Pieces)",
                title_right="After group_whole_diagrams() (Unified Box)",
            )

            # Crop Diagram
            crops = crop_diagram_groups(
                base_img,
                diagram_groups,
                output_dir,
                short_name,
                custom_crop_names=crop_names,
                crops_subdir=crops_dir,
            )

            # Build Unified Intermediate Representation (JSON IR) for Raster Diagram
            unified_elements = []
            for e_i, it in enumerate(before_items, 1):
                unified_elements.append({
                    "id": f"elem_{e_i:04d}",
                    "type": it.get("label", "text"),
                    "page": 1,
                    "bbox": it.get("bbox"),
                    "reading_order": it.get("order", e_i - 1),
                    "level": 1,
                    "parent_id": None,
                    "text": it.get("text", ""),
                    "source_ref": {
                        "file": fn,
                        "page": 1,
                        "bbox": it.get("bbox"),
                    },
                    "confidence": 0.95,
                    "metadata": {
                        "raw_order": it.get("order", e_i - 1),
                    },
                })

            unified_diagrams = []
            for dg_i, dg in enumerate(diagram_groups, 1):
                crop_rel = crops[dg_i - 1] if dg_i - 1 < len(crops) else ""
                merged_elem_ids = [f"elem_{k + 1:04d}" for k in dg.get("merged_element_indices", [])]
                unified_diagrams.append({
                    "id": f"diag_{dg['group_id']:04d}",
                    "page": 1,
                    "bbox": dg["bbox"],
                    "cropped_image": crop_rel,
                    "merged_element_ids": merged_elem_ids,
                    "text_summary": dg.get("text", ""),
                    "nodes": [],
                    "edges": [],
                    "source_ref": {
                        "file": fn,
                        "page": 1,
                        "bbox": dg["bbox"],
                    },
                    "metadata": {
                        "agent_analyzed": False,
                        "merged_blocks_count": dg.get("merged_blocks_count", len(merged_elem_ids)),
                        "contained_labels": dg.get("contained_labels", []),
                    },
                })

            unified_ir = {
                "doc_id": short_name,
                "source_file": fn,
                "doc_type": "raster_image",
                "metadata": {
                    "category": tc_name,
                    "description": tc["description"],
                    "page_width": img_w,
                    "page_height": img_h,
                    "parser": "group_whole_diagrams_image (Docling + OpenCV + Spatial Clustering)",
                    "processing_time_sec": round(time.time() - t_start, 2),
                    "custom_tags": ["raster_image", "diagram_grouping"],
                    "extra": {
                        "total_diagram_groups": len(unified_diagrams),
                        "total_raw_elements": len(unified_elements),
                    },
                },
                "elements": unified_elements,
                "tables": [],
                "diagrams": unified_diagrams,
                "edges": [],
                "findings": [],
            }
            # Postprocess IR & generate Markdown
            unified_ir, md_content = postprocess_ir_structure(unified_ir)

            json_ir_path = output_dir / f"{short_name}.json"
            with open(json_ir_path, "w", encoding="utf-8") as f_ir:
                json.dump(unified_ir, f_ir, indent=2, ensure_ascii=False)
            print(f"  [+] Saved Unified JSON IR: {json_ir_path}")

            md_path = output_dir / f"{short_name}.md"
            with open(md_path, "w", encoding="utf-8") as f_md:
                f_md.write(md_content)
            print(f"  [+] Saved Markdown: {md_path}")

            record = {
                "test_case_id": tc_id,
                "short_name": short_name,
                "name": tc_name,
                "type": tc_type,
                "filename": fn,
                "json_ir_file": str(json_ir_path.name),
                "markdown_file": str(md_path.name),
                "processing_time_sec": round(time.time() - t_start, 2),
                "before_grouping": {
                    "total_docling_items": len(before_items),
                    "labels_count": {lbl: sum(1 for it in before_items if it['label'] == lbl) for lbl in set(it['label'] for it in before_items)},
                },
                "after_grouping": {
                    "diagram_groups_count": len(diagram_groups),
                    "diagram_groups": diagram_groups,
                },
                "cropped_diagram_files": crops,
                "comparison_image": str(cmp_path.name),
            }
            all_results.append(record)

    # Save comprehensive report JSON
    print("\n[3/3] Writing grouping_report.json...")
    report_path = output_dir / "grouping_report.json"
    report_data = {
        "benchmark_metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "algorithm": "group_whole_diagrams (Vector + Raster CV Spatial Clustering)",
            "test_cases_count": len(all_results),
        },
        "test_cases": all_results,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"  [+] Saved full grouping report: {report_path}")

    print("\n================================================================================")
    print(" ALL DIAGRAM GROUPING TESTS COMPLETED SUCCESSFULLY!")
    print("================================================================================")


if __name__ == "__main__":
    main()
