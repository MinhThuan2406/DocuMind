"""
Held-out evaluation of the PDF figure-crop variants (scripts/run_pdf_figure_crops.py).

Ground truth: scripts/ground_truth/autosar_figures_v2.json
  - 'test' split: 4 AUTOSAR PDFs, pages sampled with a fixed seed (stratified: pages where some
    Docling variant predicted a box vs. all other pages), labelled blind and never used to tune a
    threshold. Includes pages with no figure (negatives).
  - 'dev' split (Layered only): the earlier 30-box set used to tune thresholds. Reported separately,
    do not use it to choose a configuration.

Differences from scripts/eval_pdf_figure_variants.py:
  - per document and pooled results, strict (IoU >= 0.5) and lenient (box holds >= 95% of the GT figure
    and is <= 2x its area) P/R/F1;
  - ground-truth figures smaller than 1.5% of the page are not scored (the pipeline drops such boxes by
    design) and boxes flagged extent_uncertain are kept but reported in a sensitivity line;
  - false positives on negative pages are counted separately;
  - page-level bootstrap 95% intervals for pooled F1 and for the paired F1 difference against a
    reference variant (resampling pages, so figures on the same page stay together).

Usage:
  .venv\\Scripts\\python scripts/eval_pdf_figures_v2.py --split test
  .venv\\Scripts\\python scripts/eval_pdf_figures_v2.py --split dev
"""
import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pymupdf  # noqa: E402
from PIL import Image  # noqa: E402

import run_pdf_figure_crops as R  # noqa: E402
from eval_pdf_figure_variants import DOCLING_VARIANTS, DPI, FOOTER, MIN_AREA, ZOOM, area, match_count, prf  # noqa: E402
from test_diagram_grouping import group_whole_diagrams_vector  # noqa: E402

REFERENCE = "v1 + anchor + add"
MODES = ("strict", "lenient")


def predict_pdf(pdf_path, pages, device):
    doc = pymupdf.open(str(pdf_path))
    cache = REPO / "outputs" / "08_pdf_figure_crops" / pdf_path.stem / f"_docling_cache_dpi{DPI}"
    holder = {}

    def factory():
        if "c" not in holder:
            holder["c"] = R.build_converter(device)
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
    return preds


def bootstrap(units, fn, n=2000, seed=7):
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        sample = [units[rng.randrange(len(units))] for _ in units]
        vals.append(fn(sample))
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def pooled_f1(sample, name, mode):
    tp = sum(u["res"][name][mode][0] for u in sample)
    fp = sum(u["res"][name][mode][1] for u in sample)
    fn = sum(u["res"][name][mode][2] for u in sample)
    return prf(tp, fp, fn)[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--gt", default=str(REPO / "scripts" / "ground_truth" / "autosar_figures_v2.json"))
    ap.add_argument("--pdf-dir", default=str(REPO / "datasets" / "AUTOSAR"))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", default=str(REPO / "outputs" / "13_eval_set"))
    args = ap.parse_args()

    gt = json.load(open(args.gt, encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    units = []          # one per page: {"doc", "page", "res": {variant: {mode: (tp, fp, fn)}}, "n_gt", "stratum"}
    all_preds = {}
    for stem, d in gt["documents"].items():
        if args.split == "dev":
            if "dev_pages" not in d:
                continue
            pages, figs, strata = d["dev_pages"], d["dev_figures"], {}
        else:
            pages, figs, strata = d["test_pages"], d["figures"], d["strata"]
        pdf = Path(args.pdf_dir) / f"{stem}.pdf"
        preds = predict_pdf(pdf, pages, args.device)
        all_preds[stem] = {name: {str(p): b for p, b in pp.items()} for name, pp in preds.items()}
        for p in pages:
            scored = [f["bbox"] for f in figs.get(str(p), []) if "small" not in f["flags"]]
            res = {}
            for name, pp in preds.items():
                res[name] = {}
                for mode in MODES:
                    m = match_count(pp[p], scored, mode)
                    res[name][mode] = (m, len(pp[p]) - m, len(scored) - m)
            units.append({"doc": stem, "page": p, "res": res, "n_gt": len(scored), "stratum": strata.get(str(p), "dev"),
                          "n_pred": {name: len(pp[p]) for name, pp in preds.items()}})
    json.dump(all_preds, open(out / f"preds_{args.split}.json", "w"))

    names = list(units[0]["res"].keys())
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    n_gt = sum(u["n_gt"] for u in units)
    n_neg = sum(1 for u in units if u["n_gt"] == 0)
    emit(f"# Figure-crop evaluation, split = {args.split}")
    emit(f"{len(units)} pages ({n_neg} with no scored figure), {n_gt} scored ground-truth figures "
         f"(figures < {MIN_AREA:.1%} of the page are not scored)\n")

    for mode in MODES:
        emit(f"## Pooled, {mode}  (F1 with page-bootstrap 95% interval; reference = '{REFERENCE}')")
        emit(f"| variant | TP | FP | FN | P | R | F1 | F1 95% CI | dF1 vs ref (95% CI) | FP on negative pages |")
        emit("|---|---|---|---|---|---|---|---|---|---|")
        for name in names:
            tp = sum(u["res"][name][mode][0] for u in units)
            fp = sum(u["res"][name][mode][1] for u in units)
            fn = sum(u["res"][name][mode][2] for u in units)
            pr, rc, f1 = prf(tp, fp, fn)
            lo, hi = bootstrap(units, lambda s, n=name: pooled_f1(s, n, mode))
            if name == REFERENCE:
                dtxt = "-"
            else:
                dlo, dhi = bootstrap(units, lambda s, n=name: pooled_f1(s, n, mode) - pooled_f1(s, REFERENCE, mode))
                d0 = f1 - prf(*[sum(u["res"][REFERENCE][mode][i] for u in units) for i in range(3)])[2]
                dtxt = f"{d0:+.3f} ({dlo:+.3f}, {dhi:+.3f})"
            neg_fp = sum(u["n_pred"][name] for u in units if u["n_gt"] == 0)
            emit(f"| {name} | {tp} | {fp} | {fn} | {pr:.3f} | {rc:.3f} | {f1:.3f} | ({lo:.3f}, {hi:.3f}) | {dtxt} | {neg_fp} |")
        emit()

    if args.split == "test":
        emit("## Per document, lenient (TP/FP/FN, F1)")
        docs = sorted({u["doc"] for u in units})
        emit("| variant | " + " | ".join(d.replace("AUTOSAR_CP_", "") for d in docs) + " |")
        emit("|---|" + "---|" * len(docs))
        for name in names:
            cells = []
            for dname in docs:
                us = [u for u in units if u["doc"] == dname]
                t = [sum(u["res"][name]["lenient"][i] for u in us) for i in range(3)]
                cells.append(f"{t[0]}/{t[1]}/{t[2]}, {prf(*t)[2]:.3f}")
            emit(f"| {name} | " + " | ".join(cells) + " |")
        emit()
        emit("## Per stratum, lenient (reference variant only)")
        for st in ("candidate", "other"):
            us = [u for u in units if u["stratum"] == st]
            t = [sum(u["res"][REFERENCE]["lenient"][i] for u in us) for i in range(3)]
            emit(f"- {st} pages ({len(us)}): TP/FP/FN = {t[0]}/{t[1]}/{t[2]}, F1 {prf(*t)[2]:.3f}")
        emit()

    emit(f"## Pages where '{REFERENCE}' errs (lenient)")
    for u in units:
        tp, fp, fn = u["res"][REFERENCE]["lenient"]
        if fp or fn:
            emit(f"- {u['doc'].replace('AUTOSAR_CP_', '')} p{u['page']}: TP {tp}, FP {fp}, FN {fn} (GT {u['n_gt']}, predicted {u['n_pred'][REFERENCE]})")
    (out / f"report_{args.split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
