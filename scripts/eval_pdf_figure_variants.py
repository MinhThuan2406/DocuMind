"""
Score the figure-crop variants of scripts/run_pdf_figure_crops.py against hand-labelled boxes.

Ground truth: scripts/ground_truth/autosar_layered_figures.json (30 boxes on 44 sampled pages of
AUTOSAR_CP_EXP_LayeredSoftwareArchitecture.pdf; small and labelled by hand, so differences of one or
two figures are not meaningful).

Two criteria:
  strict   a prediction matches a ground-truth box at IoU >= 0.5
  lenient  a ground-truth box counts as found if one prediction contains >= 95% of it and is at most
           2x its area (extra text around a figure is acceptable, a cut figure is not)

Docling items are cached per page in outputs/08_pdf_figure_crops/<pdf stem>/_docling_cache_dpi150
(created on first use, Docling OCR off).

Usage:
  .venv\\Scripts\\python scripts/eval_pdf_figure_variants.py datasets/AUTOSAR/AUTOSAR_CP_EXP_LayeredSoftwareArchitecture.pdf
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pymupdf  # noqa: E402
from PIL import Image  # noqa: E402

import run_pdf_figure_crops as R  # noqa: E402
from test_diagram_grouping import group_whole_diagrams_vector  # noqa: E402

DPI = 150
ZOOM = DPI / 72.0
MIN_AREA, FOOTER = 0.015, 0.88

# name: (group with v1, v1 rule, refine with vector, add vector clusters)
DOCLING_VARIANTS = {
    "raw (Docling picture boxes)": (False, "none", False, False),
    "v1 (no guard)": (True, "none", False, False),
    "v1 + anchor": (True, "anchor", False, False),
    "v1 + picture-only": (True, "picture-only", False, False),
    "v1 + anchor + refine": (True, "anchor", True, False),
    "v1 + anchor + add": (True, "anchor", False, True),
    "v1 + anchor + refine + add": (True, "anchor", True, True),
    "v1 (no guard) + add": (True, "none", False, True),
}


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a, b):
    i = inter(a, b)
    u = area(a) + area(b) - i
    return i / u if u > 0 else 0.0


def match_count(preds, gts, mode):
    if mode == "strict":
        cand = sorted((-iou(p, g), i, j) for i, p in enumerate(preds) for j, g in enumerate(gts) if iou(p, g) >= 0.5)
    else:
        cand = sorted((area(p) / area(g), i, j) for i, p in enumerate(preds) for j, g in enumerate(gts)
                      if inter(p, g) >= 0.95 * area(g) and area(p) <= 2.0 * area(g))
    used_p, used_g = set(), set()
    for _, i, j in cand:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
    return len(used_p)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("--gt", default=str(REPO / "scripts" / "ground_truth" / "autosar_layered_figures.json"))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    gt = json.load(open(args.gt, encoding="utf-8"))
    figures = {int(k): v for k, v in gt["figures"].items()}
    pages = [p for p in gt["sampled_pages"] if p not in gt["excluded_ambiguous"]]
    pdf_path = Path(args.pdf)
    doc = pymupdf.open(str(pdf_path))
    cache = REPO / "outputs" / "08_pdf_figure_crops" / pdf_path.stem / f"_docling_cache_dpi{DPI}"
    holder = {}

    def factory():
        if "c" not in holder:
            holder["c"] = R.build_converter(args.device)
        return holder["c"]

    preds = {name: {} for name in DOCLING_VARIANTS}
    preds["vector only (group_whole_diagrams_vector)"] = {}
    preds["vector only (PyMuPDF, no text absorption)"] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for p in pages:
            page = doc[p - 1]
            png = Path(tmp) / f"p{p:03d}.png"
            page.get_pixmap(matrix=pymupdf.Matrix(ZOOM, ZOOM)).save(str(png))
            size = Image.open(png).size
            items, psize = R.load_or_run_docling(factory, png, cache / f"p{p:03d}.json")
            for name, (grp, rule, ref, add) in DOCLING_VARIANTS.items():
                boxes = R.compute_boxes(page, png, items, psize, size, grp, ref, add, MIN_AREA, FOOTER, rule)
                preds[name][p] = [[v / ZOOM for v in b["bbox"]] for b in boxes]
            W, H = page.rect.width, page.rect.height
            vo = group_whole_diagrams_vector(str(pdf_path), p - 1)
            preds["vector only (group_whole_diagrams_vector)"][p] = [
                g["bbox"] for g in vo["diagram_groups"] if area(g["bbox"]) >= MIN_AREA * W * H and g["bbox"][1] < FOOTER * H]
            dr, im, _w = R.vector_primitives(page, FOOTER)
            preds["vector only (PyMuPDF, no text absorption)"][p] = R.vector_cluster_boxes(dr + im, 14.0, MIN_AREA * W * H, 2, W, H)

    n_gt = sum(len(figures.get(p, [])) for p in pages)
    print(f"{len(pages)} pages scored, {n_gt} ground-truth figures\n")
    print(f"{'variant':44s} | strict TP/FP/FN   P     R     F1  | lenient TP/FP/FN   P     R     F1")
    for name, pp in preds.items():
        cells = []
        for mode in ("strict", "lenient"):
            tp = fp = fn = 0
            for p in pages:
                P, G = pp.get(p, []), figures.get(p, [])
                m = match_count(P, G, mode)
                tp += m
                fp += len(P) - m
                fn += len(G) - m
            pr, rc, f1 = prf(tp, fp, fn)
            cells.append(f"{tp:3d}/{fp:2d}/{fn:2d}  {pr:5.3f} {rc:5.3f} {f1:5.3f}")
        print(f"{name:44s} | {cells[0]} | {cells[1]}")


if __name__ == "__main__":
    main()
