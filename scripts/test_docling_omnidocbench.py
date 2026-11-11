"""
Test Docling on OmniDocBench Dataset
Evaluates layout detection, block classification, table extraction, and reading order
against OmniDocBench Ground Truth.
"""

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont
from docling.document_converter import DocumentConverter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ir_postprocessor import postprocess_ir_structure


# Selected diverse samples from OmniDocBench
SELECTED_SAMPLES = [
    {
        "filename": "page-83587fc3-fe55-4e51-a3c5-f0664edba743.png",
        "short_name": "01_academic_2col",
        "description": "Trang báo cáo kỹ thuật/học thuật đa cột có 2 bảng toán học và ký hiệu công thức",
        "category": "Academic Literature (Multi-column with Math Tables)",
    },
    {
        "filename": "PPT_PresentationSpecification_page_010.png",
        "short_name": "02_ppt_slide",
        "description": "Trang slide PPT2PDF dạng presentation layout, danh sách mục và bảng lịch trình",
        "category": "Presentation Slide (PPT2PDF)",
    },
    {
        "filename": "page-14a6b411-9097-4eec-86da-92075868d243.png",
        "short_name": "03_merged_table",
        "description": "Trang báo cáo khoa học có bảng phức tạp với nhiều merged cells (colspan, rowspan)",
        "category": "Complex Merged Table (Academic Literature)",
    },
    {
        "filename": "page-73432bc8-cd0f-4e14-b119-0a9de73ab141.png",
        "short_name": "04_financial_report",
        "description": "Trang báo cáo phân tích tài chính/ngành có bảng phân hạng nhiều dòng gộp và thông tin liên hệ",
        "category": "Financial / Industry Research Report",
    },
]

# Color palette for layout visualization (RGB tuples)
CATEGORY_COLORS = {
    # Tables
    "table": (0, 168, 107),              # Emerald Green
    "table_caption": (46, 139, 87),      # Sea Green
    "table_footnote": (60, 179, 113),    # Medium Sea Green
    # Titles & Headers
    "title": (30, 144, 255),             # Dodger Blue
    "section_header": (30, 144, 255),    # Dodger Blue
    "header": (70, 130, 180),            # Steel Blue
    "page_header": (70, 130, 180),
    "footer": (160, 82, 45),             # Sienna
    "page_footer": (160, 82, 45),
    "page_number": (0, 139, 139),        # Dark Cyan
    # Text & Lists
    "text_block": (105, 105, 105),       # Dim Gray
    "text": (105, 105, 105),
    "paragraph": (105, 105, 105),
    "list_group": (255, 140, 0),         # Dark Orange
    "list_item": (255, 140, 0),
    # Figures & Pictures
    "figure": (147, 112, 219),           # Medium Purple
    "picture": (147, 112, 219),
    "figure_caption": (138, 43, 226),    # Blue Violet
    "figure_footnote": (186, 85, 211),
    # Formulas & Code
    "equation_isolated": (220, 20, 60),  # Crimson
    "formula": (220, 20, 60),
    "equation_caption": (199, 21, 133),
    "code_txt": (178, 34, 34),           # Firebrick
    "code": (178, 34, 34),
    # Footnotes & Reference
    "footnote": (188, 143, 143),         # Rosy Brown
    "page_footnote": (188, 143, 143),
    "reference": (106, 90, 205),         # Slate Blue
    # Discarded
    "abandon": (169, 169, 169),          # Gray
}
DEFAULT_COLOR = (120, 120, 120)


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


def get_color(label: str) -> Tuple[int, int, int]:
    """Return an RGB color tuple for a given category/label."""
    normalized = label.lower().replace(" ", "_")
    return CATEGORY_COLORS.get(normalized, DEFAULT_COLOR)


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


def draw_labeled_boxes(
    image: Image.Image,
    elements: List[Dict[str, Any]],
    title: str = "",
    line_width: int = 3,
) -> Image.Image:
    """
    Draw bounding boxes, order indicators, and category labels on an image.
    elements: list of dicts with 'bbox' [xmin, ymin, xmax, ymax], 'label', 'order'
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    font_size = max(14, int(w / 75))
    font = load_font(font_size)
    header_font = load_font(max(20, int(w / 45)))

    # Draw header banner if title is provided
    if title:
        banner_h = font_size * 2 + 10
        draw.rectangle([0, 0, w, banner_h], fill=(24, 28, 36))
        draw.text((15, 8), title, fill=(255, 255, 255), font=header_font)

    for el in elements:
        bbox = el.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        xmin, ymin, xmax, ymax = bbox
        label = el.get("type") or el.get("label", "unknown")
        order = el.get("reading_order") if el.get("reading_order") is not None else el.get("order")
        color = get_color(label)

        # Draw bounding rectangle
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=line_width)

        # Badge text
        if order is not None:
            tag = f"#{order} {label}"
        else:
            tag = f"{label}"

        # Calculate text bounding box for tag background
        text_bbox = draw.textbbox((xmin, ymin), tag, font=font)
        tb_w = text_bbox[2] - text_bbox[0]
        tb_h = text_bbox[3] - text_bbox[1]

        # Draw tag background
        tag_ymin = max(0, ymin - tb_h - 4)
        tag_ymax = tag_ymin + tb_h + 4
        tag_xmax = min(w, xmin + tb_w + 8)
        draw.rectangle([xmin, tag_ymin, tag_xmax, tag_ymax], fill=color)
        draw.text((xmin + 4, tag_ymin + 1), tag, fill=(255, 255, 255), font=font)

    return img


def create_side_by_side_comparison(
    orig_img_path: str,
    gt_elements: List[Dict[str, Any]],
    docling_elements: List[Dict[str, Any]],
    output_path: str,
    sample_desc: str,
):
    """
    Creates a composite side-by-side image:
    Left: Ground Truth annotations
    Right: Docling predictions
    """
    base_img = Image.open(orig_img_path).convert("RGB")
    w, h = base_img.size

    gt_vis = draw_labeled_boxes(base_img, gt_elements, title=f"Ground Truth ({len(gt_elements)} blocks)")
    doc_vis = draw_labeled_boxes(base_img, docling_elements, title=f"Docling Predictions ({len(docling_elements)} blocks)")

    # Composite side by side with separator
    sep_width = 8
    total_w = w * 2 + sep_width
    comp_img = Image.new("RGB", (total_w, h), color=(200, 200, 200))
    comp_img.paste(gt_vis, (0, 0))
    comp_img.paste(doc_vis, (w + sep_width, 0))

    comp_img.save(output_path, "PNG")
    print(f"  [+] Saved side-by-side comparison: {output_path}")


def main():
    root_dir = Path("C:/Workspace/DocuMind")
    dataset_json_path = root_dir / "datasets/OmniDocBench/OmniDocBench.json"
    images_dir = root_dir / "datasets/OmniDocBench/images"
    output_dir = root_dir / "outputs/01_docling_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print(" DOCLING BENCHMARK ON OMNIDOCBENCH")
    print("================================================================================")
    print(f"Dataset GT: {dataset_json_path}")
    print(f"Images Dir: {images_dir}")
    print(f"Output Dir: {output_dir}")

    # Load Ground Truth
    print("\n[1/4] Loading OmniDocBench.json ground truth...")
    with open(dataset_json_path, "r", encoding="utf-8") as f:
        omnidoc_data = json.load(f)

    # Index OmniDocBench items by image_path
    gt_by_filename = {}
    for item in omnidoc_data:
        p_info = item.get("page_info", {})
        img_name = p_info.get("image_path")
        if img_name:
            gt_by_filename[img_name] = item

    print(f"  Indexed {len(gt_by_filename)} ground truth pages.")

    # Initialize DocumentConverter
    print("\n[2/4] Initializing Docling DocumentConverter...")
    t_init = time.time()
    converter = DocumentConverter()
    print(f"  Converter ready in {time.time() - t_init:.2f}s")

    # Process samples
    print("\n[3/4] Processing selected samples...")
    results_summary = []

    for idx, sample in enumerate(SELECTED_SAMPLES, 1):
        filename = sample["filename"]
        short_name = sample.get("short_name", Path(filename).stem)
        desc = sample["description"]
        cat = sample["category"]
        image_path = images_dir / filename
        stem = Path(filename).stem

        print(f"\n--------------------------------------------------------------------------------")
        print(f"Sample {idx}/{len(SELECTED_SAMPLES)}: [{short_name}] {filename}")
        print(f"Type: {cat}")
        print(f"Desc: {desc}")
        print(f"Path: {image_path}")

        if not image_path.exists():
            print(f"  [!] ERROR: Image file not found: {image_path}")
            continue

        gt_item = gt_by_filename.get(filename)
        if not gt_item:
            print(f"  [!] WARNING: Ground truth entry not found for {filename}")
            continue

        # Extract Ground Truth Elements
        gt_page_info = gt_item.get("page_info", {})
        gt_dets = gt_item.get("layout_dets", [])
        gt_elements = []
        gt_categories = Counter()
        gt_tables = []

        for d_idx, d in enumerate(gt_dets):
            c_type = d.get("category_type", "unknown")
            gt_categories[c_type] += 1
            poly = d.get("poly", [])
            bbox = None
            if len(poly) >= 8:
                xs = poly[0::2]
                ys = poly[1::2]
                bbox = [min(xs), min(ys), max(xs), max(ys)]

            el = {
                "order": d.get("order", d_idx),
                "label": c_type,
                "bbox": bbox,
                "text": d.get("text", ""),
                "html": d.get("html", ""),
                "latex": d.get("latex", ""),
            }
            gt_elements.append(el)
            if c_type == "table":
                gt_tables.append(el)

        # Run Docling Conversion
        print(f"  -> Running Docling DocumentConverter...")
        t_start = time.time()
        conv_res = converter.convert(str(image_path))
        proc_time = time.time() - t_start
        print(f"  -> Conversion completed in {proc_time:.2f}s")

        doc = conv_res.document
        page_h = doc.pages[1].size.height if (1 in doc.pages) else gt_page_info.get("height", 1000)
        page_w = doc.pages[1].size.width if (1 in doc.pages) else gt_page_info.get("width", 1000)

        # Extract Docling Items
        docling_elements = []
        docling_labels = Counter()
        order_idx = 0

        for item, level in doc.iterate_items():
            raw_label = getattr(item, "label", type(item).__name__)
            label_str = raw_label.value if hasattr(raw_label, "value") else str(raw_label)
            docling_labels[label_str] += 1

            prov = getattr(item, "prov", None)
            bbox = None
            if prov and len(prov) > 0 and hasattr(prov[0], "bbox"):
                b = prov[0].bbox.to_top_left_origin(page_h)
                bbox = [round(b.l, 2), round(b.t, 2), round(b.r, 2), round(b.b, 2)]

            text_content = getattr(item, "text", "")
            el_info = {
                "order": order_idx,
                "level": level,
                "label": label_str,
                "bbox": bbox,
                "text": text_content,
            }
            docling_elements.append(el_info)
            order_idx += 1

        # Extract Docling Tables specifically
        docling_tables = []
        for t_idx, t in enumerate(doc.tables):
            prov = getattr(t, "prov", None)
            t_bbox = None
            if prov and len(prov) > 0 and hasattr(prov[0], "bbox"):
                tb = prov[0].bbox.to_top_left_origin(page_h)
                t_bbox = [round(tb.l, 2), round(tb.t, 2), round(tb.r, 2), round(tb.b, 2)]

            t_html = t.export_to_html(doc=doc)
            t_md = t.export_to_markdown(doc=doc)
            t_rows = getattr(t.data, "num_rows", None)
            t_cols = getattr(t.data, "num_cols", None)

            # Extract detailed table cells for Unified IR (Section 5.3)
            t_cells = []
            if hasattr(t, "data") and hasattr(t.data, "table_cells") and t.data.table_cells:
                for c in t.data.table_cells:
                    r = getattr(c, "start_row_offset_idx", 0)
                    col = getattr(c, "start_col_offset_idx", 0)
                    r_span = getattr(c, "row_span", 1)
                    c_span = getattr(c, "col_span", 1)
                    is_hdr = bool(getattr(c, "column_header", False) or getattr(c, "row_header", False))
                    is_mrg = bool(r_span > 1 or c_span > 1)
                    c_text = c.text.strip() if hasattr(c, "text") and c.text else ""
                    c_bbox = None
                    if hasattr(c, "bbox") and c.bbox:
                        try:
                            cb = c.bbox.to_top_left_origin(page_h)
                            c_bbox = [round(cb.l, 2), round(cb.t, 2), round(cb.r, 2), round(cb.b, 2)]
                        except Exception:
                            pass

                    t_cells.append({
                        "row": r,
                        "col": col,
                        "rowspan": r_span,
                        "colspan": c_span,
                        "text": c_text,
                        "is_header": is_hdr,
                        "is_merged": is_mrg,
                        "bbox": c_bbox,
                    })

            docling_tables.append({
                "table_index": t_idx,
                "bbox": t_bbox,
                "num_rows": t_rows,
                "num_cols": t_cols,
                "cells": t_cells,
                "html": t_html,
                "markdown": t_md,
            })

        # Match Tables with Ground Truth via IoU
        table_matches = []
        for dt_idx, dt in enumerate(docling_tables):
            best_iou = 0.0
            best_gt_idx = -1
            best_gt_tbl = None
            if dt["bbox"]:
                for gt_i, gt_t in enumerate(gt_tables):
                    if gt_t["bbox"]:
                        iou = compute_iou(dt["bbox"], gt_t["bbox"])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_i
                            best_gt_tbl = gt_t

            match_info = {
                "docling_table_idx": dt_idx,
                "matched_gt_table_idx": best_gt_idx,
                "iou": round(best_iou, 4),
                "docling_shape": f"{dt['num_rows']}x{dt['num_cols']}",
                "docling_html_preview": dt["html"][:180] if dt["html"] else "",
                "gt_html_preview": best_gt_tbl["html"][:180] if best_gt_tbl and best_gt_tbl.get("html") else "",
            }
            table_matches.append(match_info)

        # Build Unified Intermediate Representation (JSON IR) according to Section 5.3
        unified_elements = []
        for e_idx, el in enumerate(docling_elements, 1):
            unified_elements.append({
                "id": f"elem_{e_idx:04d}",
                "type": el["label"],
                "page": 1,
                "bbox": el["bbox"],
                "reading_order": el["order"],
                "level": el.get("level", 1),
                "parent_id": None,
                "text": el.get("text", ""),
                "source_ref": {
                    "file": filename,
                    "page": 1,
                    "bbox": el["bbox"],
                },
                "confidence": 0.95,
                "metadata": {
                    "raw_order": el["order"],
                },
            })

        unified_tables = []
        for tb_idx, dt in enumerate(docling_tables, 1):
            matched_iou = 0.0
            if tb_idx - 1 < len(table_matches):
                matched_iou = table_matches[tb_idx - 1].get("iou", 0.0)

            unified_tables.append({
                "id": f"tbl_{tb_idx:04d}",
                "page": 1,
                "bbox": dt["bbox"],
                "num_rows": dt["num_rows"],
                "num_cols": dt["num_cols"],
                "cells": dt["cells"],
                "html": dt["html"],
                "source_ref": {
                    "file": filename,
                    "page": 1,
                    "bbox": dt["bbox"],
                },
                "metadata": {
                    "gt_matched_iou": matched_iou,
                    "agent_verified": False,
                },
            })

        unified_ir = {
            "doc_id": short_name,
            "source_file": filename,
            "doc_type": "raster_image",
            "metadata": {
                "category": cat,
                "description": desc,
                "page_width": round(page_w, 1),
                "page_height": round(page_h, 1),
                "parser": "docling-2.150.0",
                "processing_time_sec": round(proc_time, 2),
                "custom_tags": ["docling_benchmark", "omnidocbench"],
                "extra": {
                    "total_blocks": len(unified_elements),
                    "total_tables": len(unified_tables),
                    "gt_blocks_count": len(gt_elements),
                },
            },
            "elements": unified_elements,
            "tables": unified_tables,
            "diagrams": [],
            "edges": [],
            "findings": [],
        }

        # Postprocess IR (Column-aware reading order sorting, hierarchy tree, table-caption linking)
        unified_ir, md_content = postprocess_ir_structure(unified_ir, page_w, page_h)

        # Export Unified JSON IR
        json_output_path = output_dir / f"{short_name}.json"
        with open(json_output_path, "w", encoding="utf-8") as f_json_out:
            json.dump(unified_ir, f_json_out, indent=2, ensure_ascii=False)
        print(f"  [+] Saved Unified JSON IR: {json_output_path}")

        # Export Markdown (.md)
        md_output_path = output_dir / f"{short_name}.md"
        with open(md_output_path, "w", encoding="utf-8") as f_md_out:
            f_md_out.write(md_content)
        print(f"  [+] Saved Markdown: {md_output_path}")

        # Generate Visualizations with sorted elements
        overlay_output_path = output_dir / f"{short_name}_overlay.png"
        side_by_side_output_path = output_dir / f"{short_name}_compare.png"

        # 1. Overlay of Docling sorted predictions
        orig_img = Image.open(image_path).convert("RGB")
        overlay_img = draw_labeled_boxes(
            orig_img,
            unified_ir["elements"],
            title=f"Docling Sorted Predictions: {short_name} ({len(unified_ir['elements'])} blocks)",
        )
        overlay_img.save(overlay_output_path, "PNG")
        print(f"  [+] Saved Docling overlay: {overlay_output_path}")

        # 2. Side-by-side comparison
        create_side_by_side_comparison(
            str(image_path),
            gt_elements,
            unified_ir["elements"],
            str(side_by_side_output_path),
            desc,
        )

        # Construct evaluation record
        sample_record = {
            "sample_index": idx,
            "short_name": short_name,
            "filename": filename,
            "category": cat,
            "description": desc,
            "page_dimensions": {"width": page_w, "height": page_h},
            "processing_time_sec": round(proc_time, 2),
            "ground_truth": {
                "attributes": gt_page_info.get("page_attribute", {}),
                "total_blocks": len(gt_elements),
                "category_counts": dict(gt_categories),
                "num_tables": len(gt_tables),
            },
            "docling_results": {
                "total_blocks": len(unified_ir["elements"]),
                "label_counts": dict(docling_labels),
                "num_tables": len(docling_tables),
                "table_matches": table_matches,
            },
            "outputs": {
                "json_ir_file": str(json_output_path.name),
                "markdown_file": str(md_output_path.name),
                "overlay_image": str(overlay_output_path.name),
                "comparison_image": str(side_by_side_output_path.name),
            },
            "detailed_docling_blocks": [
                {
                    "order": el["order"],
                    "label": el["label"],
                    "bbox": el["bbox"],
                    "text_preview": (el["text"][:100] + "...") if len(el["text"]) > 100 else el["text"],
                }
                for el in docling_elements
            ],
            "table_extraction_details": [
                {
                    "table_index": dt["table_index"],
                    "bbox": dt["bbox"],
                    "num_rows": dt["num_rows"],
                    "num_cols": dt["num_cols"],
                    "markdown_preview": dt["markdown"][:300] if dt["markdown"] else "",
                    "html_preview": dt["html"][:300] if dt["html"] else "",
                }
                for dt in docling_tables
            ],
        }
        results_summary.append(sample_record)

        # Print comparison recap
        print(f"  -> GT Blocks: {len(gt_elements)} | Docling Blocks: {len(docling_elements)}")
        print(f"  -> GT Tables: {len(gt_tables)} | Docling Tables: {len(docling_tables)}")
        for tm in table_matches:
            print(f"     * Docling Table {tm['docling_table_idx']} matched GT Table {tm['matched_gt_table_idx']} with IoU={tm['iou']} (Shape: {tm['docling_shape']})")

    # Save summary report JSON
    print("\n[4/4] Writing comprehensive summary report JSON...")
    report_path = output_dir / "summary_report.json"
    full_report = {
        "benchmark_metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": "OmniDocBench v1.6",
            "evaluator": "docling 2.150.0",
            "device": "auto",
            "num_samples_evaluated": len(results_summary),
        },
        "samples": results_summary,
    }

    with open(report_path, "w", encoding="utf-8") as f_json:
        json.dump(full_report, f_json, indent=2, ensure_ascii=False)
    print(f"  [+] Saved full benchmark report to: {report_path}")

    print("\n================================================================================")
    print(" BENCHMARK COMPLETED SUCCESSFULLY!")
    print("================================================================================")


if __name__ == "__main__":
    main()
