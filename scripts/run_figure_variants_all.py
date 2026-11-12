"""
Run every figure-crop variant on one or more PDFs so the results can be compared by eye.

Variants (naming: v1 = OpenCV grouping with the default "anchor" guard of run_pdf_figure_crops.py, i.e.
a group needs a member picture that is big enough on its own; refine = snap to PDF vector edges;
add = add vector clusters Docling missed; see scripts/run_pdf_figure_crops.py):
  A_docling_raw        Docling picture boxes only
  B_docling_v1         + v1 grouping
  C_raw_refine         A + refine
  D_v1_refine          B + refine
  E_raw_refine_add     C + add
  F_v1_refine_add      D + add
  I_v1_add_norefine    B + add
  G_vector_original    group_whole_diagrams_vector (PyMuPDF, absorbs nearby text)
  H_vector_notext      vector clusters without text absorption (table-like grids dropped)

Per PDF it writes (default out = outputs/10_autosar_all_variants/<pdf stem>/):
  variants/<name>/manifest.json   boxes (PDF points) for every page
  variants/{B,D,F}/crops/         crop PNGs of the three main Docling-based variants
  compare/pNNN.png                4-panel montage (B | D | F | H) for pages where the variants disagree,
                                  where F added a vector-only box or a box looks like a table
  summary.json                    counts per variant (figures, suspected tables, huge boxes, ...)
and a report.md over all PDFs in the output root.

Usage:
  .venv\\Scripts\\python scripts/run_figure_variants_all.py datasets/AUTOSAR/*.pdf
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pymupdf  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import run_pdf_figure_crops as R  # noqa: E402
from test_diagram_grouping import group_whole_diagrams_vector  # noqa: E402

DPI = 150
ZOOM = DPI / 72.0
MIN_AREA, FOOT = 0.015, 0.88
CROP_VARIANTS = ("B_docling_v1", "D_v1_refine", "F_v1_refine_add")
KEY = ["B_docling_v1", "D_v1_refine", "F_v1_refine_add", "H_vector_notext"]
COLORS = {"B_docling_v1": (0, 0, 255), "D_v1_refine": (255, 140, 0), "F_v1_refine_add": (200, 0, 200),
          "H_vector_notext": (255, 0, 0)}
DOCLING_FLAGS = {  # name: (group, refine, add)
    "A_docling_raw": (False, False, False), "B_docling_v1": (True, False, False),
    "C_raw_refine": (False, True, False), "D_v1_refine": (True, True, False),
    "E_raw_refine_add": (False, True, True), "F_v1_refine_add": (True, True, True),
    "I_v1_add_norefine": (True, False, True),
}
ALL = list(DOCLING_FLAGS) + ["G_vector_original", "H_vector_notext"]


def iou(a, b):
    i = R._inter(a, b)
    u = R._area(a) + R._area(b) - i
    return i / u if u > 0 else 0.0


def same_sets(a, b, thr=0.85):
    """True if both box sets match one-to-one at IoU >= thr."""
    if len(a) != len(b):
        return False
    used = set()
    for x in a:
        best, bj = 0.0, None
        for j, y in enumerate(b):
            if j in used:
                continue
            s = iou(x, y)
            if s > best:
                best, bj = s, j
        if best < thr:
            return False
        used.add(bj)
    return True


def keep_pt(b, W, H):
    return R._area(b) >= MIN_AREA * W * H and b[1] < FOOT * H


def panel(page, boxes, color, title, z=0.9):
    pix = page.get_pixmap(matrix=pymupdf.Matrix(z, z))
    im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    d = ImageDraw.Draw(im)
    for b in boxes:
        d.rectangle([c * z for c in b], outline=color, width=3)
    d.rectangle([0, 0, 420, 14], fill=(255, 255, 255))
    d.text((4, 2), f"p{page.number + 1} {title}", fill=(0, 0, 0))
    return im


def run_pdf(pdf_path: Path, out_root: Path, converter_factory):
    stem = pdf_path.stem
    out = out_root / stem
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = REPO / "outputs" / "08_pdf_figure_crops" / stem / f"_docling_cache_dpi{DPI}"
    doc = pymupdf.open(str(pdf_path))
    n = len(doc)
    per_variant = {v: {} for v in ALL}      # variant -> page -> [{bbox_pt, source}]
    table_boxes = {}                          # page -> Docling table boxes (pt)
    for v in CROP_VARIANTS:
        (out / "variants" / v / "crops").mkdir(parents=True, exist_ok=True)
    (out / "compare").mkdir(exist_ok=True)
    t0 = time.time()

    with tempfile.TemporaryDirectory() as tmp:
        for pno in range(1, n + 1):
            page = doc[pno - 1]
            W, H = page.rect.width, page.rect.height
            png = Path(tmp) / f"p{pno:03d}.png"
            page.get_pixmap(matrix=pymupdf.Matrix(ZOOM, ZOOM)).save(str(png))
            img = Image.open(png).convert("RGB")
            items, psize = R.load_or_run_docling(converter_factory, png, cache_dir / f"p{pno:03d}.json")

            for name, (grp, ref, add) in DOCLING_FLAGS.items():
                boxes = R.compute_boxes(page, png, items, psize, img.size, grp, ref, add, MIN_AREA, FOOT)
                per_variant[name][pno] = [
                    {"bbox_pt": [round(v / ZOOM, 1) for v in b["bbox"]], "source": b["source"],
                     "vector_refined": b["refined"]} for b in boxes]
                if name in CROP_VARIANTS:
                    for k, b in enumerate(boxes, 1):
                        x0, y0, x1, y1 = (int(round(v)) for v in b["bbox"])
                        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(img.size[0], x1), min(img.size[1], y1)
                        if x1 > x0 and y1 > y0:
                            img.crop((x0, y0, x1, y1)).save(str(out / "variants" / name / "crops" / f"p{pno:03d}_f{k:02d}.png"))

            vo = group_whole_diagrams_vector(str(pdf_path), pno - 1)
            per_variant["G_vector_original"][pno] = [
                {"bbox_pt": g["bbox"], "source": "vector_group", "vector_refined": False}
                for g in vo["diagram_groups"] if keep_pt(g["bbox"], W, H)]
            drawings, images, _w = R.vector_primitives(page, FOOT)
            per_variant["H_vector_notext"][pno] = [
                {"bbox_pt": [round(x, 1) for x in u], "source": "vector_notext", "vector_refined": False}
                for u in R.vector_cluster_boxes(drawings + images, 14.0, MIN_AREA * W * H, 2, W, H)]

            sx, sy = img.size[0] / psize[0], img.size[1] / psize[1]
            table_boxes[pno] = [[it["bbox"][0] * sx / ZOOM, it["bbox"][1] * sy / ZOOM, it["bbox"][2] * sx / ZOOM, it["bbox"][3] * sy / ZOOM]
                                for it in items if it["label"] == "table" and it["bbox"]]
            if pno % 25 == 0 or pno == n:
                print(f"  {stem}: {pno}/{n} pages ({round(time.time() - t0)}s)", flush=True)

    # ---- manifests ----
    for v in ALL:
        with open(_mk(out / "variants" / v) / "manifest.json", "w", encoding="utf-8") as f:
            json.dump({"pdf": str(pdf_path), "variant": v,
                       "pages": {str(p): per_variant[v][p] for p in range(1, n + 1)}}, f, ensure_ascii=False)

    # ---- summary ----
    summary = {"pdf": stem, "pages": n, "variants": {}}
    for v in ALL:
        figs = [(p, b) for p in range(1, n + 1) for b in per_variant[v][p]]
        pages_w = len({p for p, _ in figs})
        susp_table = 0
        huge = 0
        for p, b in figs:
            bb = b["bbox_pt"]
            W, H = doc[p - 1].rect.width, doc[p - 1].rect.height
            if any(R._inter(bb, t) >= 0.5 * R._area(bb) for t in table_boxes[p]):
                susp_table += 1
            if R._area(bb) > 0.5 * W * H:
                huge += 1
        summary["variants"][v] = {"figures": len(figs), "pages_with_figures": pages_w,
                                  "figures_overlapping_docling_table": susp_table, "figures_over_50pct_page": huge,
                                  "vector_only_boxes": sum(1 for _, b in figs if b["source"] == "vector_only")}

    # ---- compare montages ----
    montage_pages = []
    for p in range(1, n + 1):
        B, D, F = (per_variant[v][p] for v in CROP_VARIANTS)
        sets = [[b["bbox_pt"] for b in x] for x in (B, D, F)]
        disagree = not (same_sets(sets[0], sets[1]) and same_sets(sets[1], sets[2]))
        has_vec_only = any(b["source"] == "vector_only" for b in F)
        tbl = any(any(R._inter(b["bbox_pt"], t) >= 0.5 * R._area(b["bbox_pt"]) for t in table_boxes[p]) for b in F)
        if disagree or has_vec_only or tbl:
            montage_pages.append((p, "disagree" if disagree else "", "vector_only" if has_vec_only else "", "table?" if tbl else ""))
    for p, *_ in montage_pages:
        ims = [panel(doc[p - 1], [b["bbox_pt"] for b in per_variant[v][p]], COLORS[v], v) for v in KEY]
        w, h = ims[0].size
        sh = Image.new("RGB", (2 * w, 2 * h), "white")
        for i, im in enumerate(ims):
            sh.paste(im, ((i % 2) * w, (i // 2) * h))
        sh.save(out / "compare" / f"p{p:03d}.png")
    summary["compare_pages"] = [{"page": p, "reasons": [r for r in rs if r]} for p, *rs in montage_pages]
    summary["seconds"] = round(time.time() - t0, 1)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    return summary


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_report(out_root: Path):
    lines = ["# So sanh cac thuat toan tren cac file AUTOSAR", "",
             "Khong co ground truth cho cac file nay (chi file LayeredSoftwareArchitecture co, xem outputs/09_refine_vector_compare).",
             "Cot 'trung bang' = so box (bi Docling nhan la table) chiem >=50% dien tich box; 'qua lon' = box > 50% trang.", ""]
    for sp in sorted(out_root.glob("*/summary.json")):
        s = json.load(open(sp, encoding="utf-8"))
        lines += [f"## {s['pdf']} ({s['pages']} trang, {s['seconds']}s)", "",
                  "| Bien the | So hinh | Trang co hinh | Trung bang | Qua lon | Chi tu vector |", "|---|---|---|---|---|---|"]
        for v, m in s["variants"].items():
            lines.append(f"| {v} | {m['figures']} | {m['pages_with_figures']} | {m['figures_overlapping_docling_table']} | "
                         f"{m['figures_over_50pct_page']} | {m['vector_only_boxes']} |")
        lines += ["", f"Trang co anh ghep 4 bien the (B|D|F|H): {len(s['compare_pages'])} trang, thu muc `{s['pdf']}/compare/`.", ""]
    (out_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("--out", default=str(REPO / "outputs" / "10_autosar_all_variants"))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()
    out_root = Path(args.out)
    holder = {}

    def factory():
        if "c" not in holder:
            holder["c"] = R.build_converter(args.device)
        return holder["c"]

    for p in args.pdfs:
        print(f"== {p}", flush=True)
        run_pdf(Path(p), out_root, factory)
        write_report(out_root)
    print("done ->", out_root)


if __name__ == "__main__":
    main()
