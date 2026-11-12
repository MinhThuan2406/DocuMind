"""
PDF -> figure crops (no text/IR processing).

Goal: for every page of a PDF, cut each figure (picture or whole diagram) out as ONE image.
What is inside the figure is handled later.

Pipeline (configurations measured on AUTOSAR_CP_EXP_LayeredSoftwareArchitecture, see
docs/04 and outputs/09_refine_vector_compare):
  1. render each page to PNG
  2. Docling layout model on the page image (OCR off) -> `picture` boxes (+ `table` boxes)
  3. optional v1 grouping (group_whole_diagrams_image) to merge pieces of one diagram
  4. optional --refine-vector: snap each Docling box to the PDF's own vector drawings /
     embedded images (exact edges), keep only labels adjacent to the drawing
  5. optional --add-vector-missing: add vector clusters that Docling did not box (tables excluded)
  6. drop boxes smaller than --min-area of the page or in the footer band (repeated logos)
  7. crop from the rendered page, write crops/ and manifest.json

Usage:
  .venv\\Scripts\\python scripts/run_pdf_figure_crops.py path/to/file.pdf
  .venv\\Scripts\\python scripts/run_pdf_figure_crops.py file.pdf --refine-vector --pages 12-20
Output: <out>/<pdf stem>[_<tag>]/{pages/, crops/, manifest.json}
(default out = outputs/08_pdf_figure_crops). Docling items are cached per page and DPI in
<out>/<pdf stem>/_docling_cache_dpi<dpi>/ so that variants can be compared without rerunning Docling.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pymupdf  # noqa: E402
from PIL import Image  # noqa: E402

from test_diagram_grouping import DisjointSetUnion, group_whole_diagrams_image  # noqa: E402

Box = List[float]


# ----------------------------------------------------------------------
# Docling (image branch)
# ----------------------------------------------------------------------
def build_converter(device: str):
    """Docling for PNG pages: layout model only (OCR off); device 'auto' | 'cpu' | 'cuda'."""
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, ImageFormatOption

    opts = ImageFormatOption().pipeline_options
    opts.do_ocr = False
    opts.accelerator_options = AcceleratorOptions(device=device)
    return DocumentConverter(format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts)})


def docling_items(converter, png_path):
    """Docling items (top-left origin) in Docling page units, plus that page size."""
    doc = converter.convert(str(png_path)).document
    page_size = (doc.pages[1].size.width, doc.pages[1].size.height)
    items = []
    for order, (it, _lvl) in enumerate(doc.iterate_items()):
        raw_label = getattr(it, "label", type(it).__name__)
        label = raw_label.value if hasattr(raw_label, "value") else str(raw_label)
        prov = getattr(it, "prov", None)
        bbox = None
        if prov and hasattr(prov[0], "bbox"):
            b = prov[0].bbox.to_top_left_origin(page_size[1])
            bbox = [b.l, b.t, b.r, b.b]
        items.append({"order": order, "label": label, "bbox": bbox, "text": getattr(it, "text", "")})
    return items, page_size


def load_or_run_docling(converter_factory, png_path: Path, cache_file: Path):
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            c = json.load(f)
        return c["items"], tuple(c["page_size"])
    items, page_size = docling_items(converter_factory(), png_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"items": items, "page_size": list(page_size)}, f, ensure_ascii=False)
    return items, page_size


# ----------------------------------------------------------------------
# Geometry helpers (all in PDF points unless stated)
# ----------------------------------------------------------------------
def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _union(boxes: List[Box]) -> Box:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _inter(a: Box, b: Box) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _inside_fraction(r: Box, box: Box) -> float:
    """Fraction of r inside box; hairlines are inflated by 0.5 pt so they have an area."""
    rr = [r[0] - 0.5, r[1] - 0.5, r[2] + 0.5, r[3] + 0.5] if min(r[2] - r[0], r[3] - r[1]) < 1.0 else r
    a = _area(rr)
    return _inter(rr, box) / a if a > 0 else 0.0


def _gap(a: Box, b: Box) -> float:
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return (dx * dx + dy * dy) ** 0.5


# ----------------------------------------------------------------------
# Vector branch primitives (PyMuPDF)
# ----------------------------------------------------------------------
def vector_primitives(page, footer_frac: float) -> Tuple[List[Box], List[Box], List[Box]]:
    """(drawing rects, embedded image rects, word boxes), without page backgrounds or footer-band items."""
    W, H = page.rect.width, page.rect.height
    drawings, images = [], []
    for d in page.get_drawings():
        r = d["rect"]
        if r.width >= 0.96 * W and r.height >= 0.96 * H:
            continue
        if r.y0 >= footer_frac * H:
            continue
        drawings.append([r.x0, r.y0, r.x1, r.y1])
    for ii in page.get_image_info():
        b = list(ii["bbox"])
        if (b[2] - b[0]) >= 0.96 * W and (b[3] - b[1]) >= 0.96 * H:
            continue
        if b[1] >= footer_frac * H:
            continue
        images.append(b)
    words = [[w[0], w[1], w[2], w[3]] for w in page.get_text("words")]
    return drawings, images, words


def refine_box_with_vector(
    box: Box, prims: List[Box], words: List[Box], margin: float = 4.0, inside_thr: float = 0.5,
    max_grow: float = 3.0, label_gap: float = 12.0,
) -> Tuple[Box, bool]:
    """
    Snap a Docling box to the vector primitives that belong to it: primitives at least `inside_thr`
    inside the box (expanded by `margin`) and not much bigger than the box (`max_grow`). The result is
    their union plus words adjacent to it (<= `label_gap` pt) whose centre is inside the Docling box.
    Returns (box, changed). If no vector primitive belongs to the box (raster-only figure) or the result
    would be tiny compared to the Docling box, the Docling box is kept.
    """
    base_area = max(1.0, _area(box))
    cur = box
    members: List[Box] = []
    for _ in range(2):
        ex = [cur[0] - margin, cur[1] - margin, cur[2] + margin, cur[3] + margin]
        members = [p for p in prims if _area(p) <= max_grow * base_area and _inside_fraction(p, ex) >= inside_thr]
        if not members:
            return box, False
        cur = _union(members)
    ex = [box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin]
    adj = [w for w in words
           if ex[0] <= (w[0] + w[2]) / 2 <= ex[2] and ex[1] <= (w[1] + w[3]) / 2 <= ex[3] and _gap(w, cur) <= label_gap]
    if adj:
        cur = _union([cur] + adj)
    if _area(cur) < 0.3 * base_area:
        return box, False
    return cur, True


def _is_table_like(cluster_prims: List[Box]) -> bool:
    """Mostly hairlines (ruled grid) -> table, not a diagram."""
    if not cluster_prims:
        return False
    thin = sum(1 for r in cluster_prims if min(r[2] - r[0], r[3] - r[1]) < 1.5)
    return thin / len(cluster_prims) > 0.6


def vector_cluster_boxes(
    prims: List[Box], pad: float, min_area_pt: float, min_prims: int, W: float, H: float,
) -> List[Box]:
    """Union-Find over primitives dilated by `pad`; no text absorption. Table-like clusters are dropped."""
    n = len(prims)
    if n == 0:
        return []
    dsu = DisjointSetUnion(n)
    for i in range(n):
        a = prims[i]
        for j in range(i + 1, n):
            b = prims[j]
            if not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1]):
                dsu.union(i, j)
    groups: Dict[int, List[Box]] = {}
    for i in range(n):
        groups.setdefault(dsu.find(i), []).append(prims[i])
    out = []
    for g in groups.values():
        if len(g) < min_prims or _is_table_like(g):
            continue
        u = _union(g)
        if _area(u) >= min_area_pt and (u[2] - u[0]) >= 50 and (u[3] - u[1]) >= 30:
            out.append(u)
    return out


def add_missing_vector_figures(
    existing: List[Box], tables: List[Box], drawings: List[Box], images: List[Box], W: float, H: float,
    min_area: float, pad: float = 8.0, min_prims: int = 8,
) -> List[Box]:
    """Vector clusters (drawings + images) that are not already covered by a figure or a Docling table."""
    covered = existing + tables
    free = [p for p in drawings + images if not any(_inside_fraction(p, c) >= 0.5 for c in covered)]
    boxes = vector_cluster_boxes(free, pad, min_area * W * H, min_prims, W, H)
    out = []
    for b in boxes:
        if any(_inter(b, c) >= 0.3 * _area(b) for c in covered):
            continue
        out.append(b)
    return out


# ----------------------------------------------------------------------
# Per-page figure boxes
# ----------------------------------------------------------------------
V1_RULES = ("anchor", "picture-only", "none")


def v1_group_boxes(png_path: Path, items, img_size, page_size, min_area: float, v1_rule: str):
    """
    v1 grouping with a guard against groups that are not figures (done here, group_whole_diagrams_image
    itself is left untouched because the OmniDocBench evaluation scripts and tests depend on it).
      anchor        keep a group only if at least one member picture/figure is, on its own, >= min_area of
                    the page (v1 may grow a real figure but cannot build one out of tables/brackets/icons)
      picture-only  give v1 only picture/figure items (no table/formula/document_index/...)
      none          plain v1 (previous behaviour)
    """
    dw, dh = page_size
    src = [i for i in items if i["bbox"]]
    if v1_rule == "picture-only":
        src = [i for i in src if i["label"] in ("picture", "figure")]
    res = group_whole_diagrams_image(str(png_path), src, img_size, page_size)
    boxes = []
    for g in res["diagram_groups"]:
        if v1_rule == "anchor":
            members = [src[k] for k in g["merged_element_indices"]]
            if not any(
                m["label"] in ("picture", "figure")
                and (m["bbox"][2] - m["bbox"][0]) * (m["bbox"][3] - m["bbox"][1]) >= min_area * dw * dh
                for m in members
            ):
                continue
        boxes.append({"bbox": list(g["bbox"]), "source": "v1_group", "n_merged": g.get("merged_blocks_count", 1)})
    return boxes


def compute_boxes(
    page, png_path: Path, items, page_size, img_size, group: bool, refine: bool, add_missing: bool,
    min_area: float, footer_frac: float, v1_rule: str = "anchor",
) -> List[Dict[str, Any]]:
    """Figure boxes of one page in image pixels: [{'bbox', 'source', 'n_merged', 'refined'}]."""
    iw, ih = img_size
    dw, dh = page_size
    W, H = page.rect.width, page.rect.height
    px_per_pt = iw / W  # image pixels per PDF point

    if group:
        boxes = v1_group_boxes(png_path, items, img_size, page_size, min_area, v1_rule)
    else:
        sx, sy = iw / dw, ih / dh
        boxes = [{"bbox": [it["bbox"][0] * sx, it["bbox"][1] * sy, it["bbox"][2] * sx, it["bbox"][3] * sy],
                  "source": "docling_picture", "n_merged": 1}
                 for it in items if it["label"] in ("picture", "figure") and it["bbox"]]
    for b in boxes:
        b["refined"] = False

    if refine or add_missing:
        drawings, images, words = vector_primitives(page, footer_frac)
        prims = drawings + images
    if refine:
        for b in boxes:
            pt = [v / px_per_pt for v in b["bbox"]]
            new_pt, changed = refine_box_with_vector(pt, prims, words)
            if changed:
                b["bbox"] = [v * px_per_pt for v in new_pt]
                b["refined"] = True
    if add_missing:
        sx, sy = iw / dw, ih / dh
        tables_pt = [[it["bbox"][0] * sx / px_per_pt, it["bbox"][1] * sy / px_per_pt,
                      it["bbox"][2] * sx / px_per_pt, it["bbox"][3] * sy / px_per_pt]
                     for it in items if it["label"] == "table" and it["bbox"]]
        existing_pt = [[v / px_per_pt for v in b["bbox"]] for b in boxes]
        for u in add_missing_vector_figures(existing_pt, tables_pt, drawings, images, W, H, min_area):
            boxes.append({"bbox": [v * px_per_pt for v in u], "source": "vector_only", "n_merged": 1, "refined": False})

    kept = [b for b in boxes if keep_box(b["bbox"], img_size, min_area, footer_frac)]
    kept.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    return kept


def keep_box(b, img_size, min_area, footer_frac):
    iw, ih = img_size
    area = (b[2] - b[0]) * (b[3] - b[1])
    return area >= min_area * iw * ih and b[1] < footer_frac * ih


def parse_pages(spec, n_pages):
    if not spec:
        return list(range(1, n_pages + 1))
    pages = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            pages.update(range(int(a), int(b) + 1))
        else:
            pages.add(int(part))
    return sorted(p for p in pages if 1 <= p <= n_pages)


def main():
    ap = argparse.ArgumentParser(description="Crop every figure of a PDF as one image (no content processing).")
    ap.add_argument("pdf")
    ap.add_argument("--out", default=str(REPO / "outputs" / "08_pdf_figure_crops"))
    ap.add_argument("--tag", default="", help="suffix for the output folder (to keep variants apart)")
    ap.add_argument("--pages", default="", help="e.g. 12-20,29 (default: all pages)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--no-group", action="store_true", help="skip v1 grouping, keep raw Docling picture boxes")
    ap.add_argument("--v1-rule", default="anchor", choices=list(V1_RULES),
                    help="guard for v1 groups: anchor (default), picture-only, none (old behaviour)")
    ap.add_argument("--refine-vector", action="store_true",
                    help="snap each Docling box to the PDF vector drawings/embedded images it contains")
    ap.add_argument("--add-vector-missing", action="store_true",
                    help="also add vector clusters Docling did not box (tables excluded)")
    ap.add_argument("--min-area", type=float, default=0.015, help="drop boxes below this page-area fraction")
    ap.add_argument("--footer-frac", type=float, default=0.88, help="drop boxes whose top is below this page-height fraction")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    base_dir = Path(args.out) / pdf_path.stem
    out_dir = Path(args.out) / (pdf_path.stem + (f"_{args.tag}" if args.tag else ""))
    (out_dir / "pages").mkdir(parents=True, exist_ok=True)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    cache_dir = base_dir / f"_docling_cache_dpi{args.dpi}"

    doc = pymupdf.open(str(pdf_path))
    pages = parse_pages(args.pages, len(doc))
    converter_holder = {}

    def converter_factory():
        if "c" not in converter_holder:
            converter_holder["c"] = build_converter(args.device)
        return converter_holder["c"]

    zoom = args.dpi / 72.0
    manifest = {
        "pdf": str(pdf_path), "dpi": args.dpi, "grouping": "off" if args.no_group else "v1",
        "v1_rule": args.v1_rule, "refine_vector": args.refine_vector, "add_vector_missing": args.add_vector_missing,
        "min_area": args.min_area, "footer_frac": args.footer_frac, "pages": [],
    }
    t0 = time.time()
    total = 0
    for pno in pages:
        png = out_dir / "pages" / f"p{pno:03d}.png"
        doc[pno - 1].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).save(str(png))
        img = Image.open(png).convert("RGB")
        img_size = img.size

        items, page_size = load_or_run_docling(converter_factory, png, cache_dir / f"p{pno:03d}.json")
        boxes = compute_boxes(doc[pno - 1], png, items, page_size, img_size, not args.no_group,
                              args.refine_vector, args.add_vector_missing, args.min_area, args.footer_frac,
                              args.v1_rule)

        figs = []
        for k, b in enumerate(boxes, 1):
            x0, y0, x1, y1 = (int(round(v)) for v in b["bbox"])
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(img_size[0], x1), min(img_size[1], y1)
            if x1 <= x0 or y1 <= y0:
                continue
            crop_rel = f"crops/p{pno:03d}_f{k:02d}.png"
            img.crop((x0, y0, x1, y1)).save(str(out_dir / crop_rel))
            figs.append({
                "id": f"p{pno:03d}_f{k:02d}",
                "bbox_px": [x0, y0, x1, y1],
                "bbox_pt": [round(v / zoom, 1) for v in (x0, y0, x1, y1)],
                "crop": crop_rel,
                "source": b["source"],
                "vector_refined": b["refined"],
                "n_merged": b["n_merged"],
            })
        total += len(figs)
        manifest["pages"].append({"page": pno, "width_px": img_size[0], "height_px": img_size[1], "figures": figs})
        print(f"page {pno}: {len(figs)} figure(s)", flush=True)

    manifest["total_figures"] = total
    manifest["seconds"] = round(time.time() - t0, 1)
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"done: {total} figures on {len(pages)} pages in {manifest['seconds']}s -> {out_dir}")


if __name__ == "__main__":
    main()
