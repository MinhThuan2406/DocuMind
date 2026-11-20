from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pdf.eval.gt import (  # noqa: E402
    PT_PER_PX_150DPI,
    append_run_log,
    build_raster_pdf,
    check_split_guard,
    load_autosar_gt,
    load_autosar_table_gt,
    load_router_gt,
)
from pdf.eval.metrics import (  # noqa: E402
    calibration_bins,
    confidence_separation_summary,
    expected_calibration_error,
    iou,
    matched_pred_indices,
    prf,
)

_AUTOSAR_DEV_TABLE_PAGES: tuple[tuple[str, list[int]], ...] = (
    ("AUTOSAR_CP_EXP_VFB.pdf", [17, 74, 100, 117, 131]),
)
_AUTOSAR_TEST_TABLE_PAGES: tuple[tuple[str, list[int]], ...] = (
    ("AUTOSAR_CP_SWS_OS.pdf", [1, 131, 209, 240, 302]),
)


def _summarize_calibration(
    confs: Sequence[float],
    labels: Sequence[int],
    tp_confs: Sequence[float],
    fp_confs: Sequence[float],
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    sep = confidence_separation_summary(tp_confs, fp_confs)
    return {
        "ece_10": expected_calibration_error(confs, labels, n_bins=n_bins),
        "mean_conf_tp": sep["mean_tp"],
        "mean_conf_fp": sep["mean_fp"],
        "mean_tp_gt_mean_fp": sep["mean_tp_gt_mean_fp"],
        "fp_ge_095": sep["fp_high_conf_count"],
        "fp_ge_095_rate": sep["fp_high_conf_rate"],
        "low_conf_lt_070": sep["low_conf_count"],
        "bins": calibration_bins(confs, labels, n_bins=n_bins),
    }


def _mean_or_none(vals: Sequence[float]) -> float | None:
    return round(sum(vals) / len(vals), 6) if vals else None


def eval_figures_calibration(
    *,
    split: str = "dev",
    layout: str = "heron-onnx",
    grouper: str = "v1_anchor",
    device: str = "auto",
    match_mode: str = "strict",
    gt_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    from pdf.backends import get as get_backend
    from pdf.figures import detect_figures

    docs = load_autosar_gt(split=split, gt_path=gt_path)
    t0 = time.perf_counter()
    all_confs: list[float] = []
    all_labels: list[int] = []
    tp_confs: list[float] = []
    fp_confs: list[float] = []
    total_fn = 0
    total_pages = 0
    total_gt = 0

    backend = get_backend("layout", layout, device=device) if docs else None
    for doc_gt in docs:
        doc_cache = Path(cache_dir) / doc_gt.stem if cache_dir is not None else None
        preds = detect_figures(
            doc_gt.pdf_path,
            doc_gt.pages,
            layout=backend,
            grouper=grouper,
            device=device,
            include_tables=False,
            cache_dir=doc_cache,
        )
        for pno in doc_gt.pages:
            total_pages += 1
            pred_figs = preds.get(pno, ())
            pred_boxes = [list(fig.bbox) for fig in pred_figs]
            gt_boxes = doc_gt.boxes_by_page.get(pno, [])
            total_gt += len(gt_boxes)
            matched = matched_pred_indices(pred_boxes, gt_boxes, mode=match_mode)
            total_fn += len(gt_boxes) - len(matched)
            for idx, fig in enumerate(pred_figs):
                conf = float(fig.confidence)
                all_confs.append(conf)
                if idx in matched:
                    all_labels.append(1)
                    tp_confs.append(conf)
                else:
                    all_labels.append(0)
                    fp_confs.append(conf)

    tp_n, fp_n = len(tp_confs), len(fp_confs)
    p, r, f1 = prf(tp_n, fp_n, total_fn)
    return {
        "task": "figures_calibration",
        "split": split,
        "layout": layout,
        "grouper": grouper,
        "match_mode": match_mode,
        "pages": total_pages,
        "gt_boxes": total_gt,
        "predictions": len(all_confs),
        "tp": tp_n,
        "fp": fp_n,
        "fn": total_fn,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "wall_sec": round(time.perf_counter() - t0, 4),
        **_summarize_calibration(all_confs, all_labels, tp_confs, fp_confs),
    }


def _check_table_source_monotonicity(
    *,
    split: str,
    device: str,
    layout: str,
    table_gt_path: str | Path | None,
    e2e_auto_tatr_confs: Sequence[float],
    max_native_tables: int = 15,
) -> dict[str, Any]:
    import pdfplumber

    from pdf.backends import get as get_backend
    from pdf.figures import detect_figures
    from pdf.tables import extract_tables_on_page
    from pdf.text import extract_words

    custom_gt = load_autosar_table_gt(split=split, gt_path=table_gt_path)
    cases_by_page: dict[tuple[Path, int], list[tuple[float, float, float, float]]] = defaultdict(list)
    if custom_gt:
        for item in custom_gt[:max_native_tables]:
            p = Path(item["pdf_path"])
            if p.is_file():
                cases_by_page[(p, int(item["page"]))].append(tuple(float(v) for v in item["bbox_pt"]))
    else:
        spec = _AUTOSAR_DEV_TABLE_PAGES if split == "dev" else _AUTOSAR_TEST_TABLE_PAGES
        count = 0
        for fname, pages in spec:
            pdf_path = ROOT / "datasets" / "AUTOSAR" / fname
            if not pdf_path.is_file():
                continue
            figs = detect_figures(pdf_path, pages, layout=layout, device=device)
            for pno in pages:
                for box in figs.get(pno, ()):
                    if box.kind != "table":
                        continue
                    cases_by_page[(pdf_path, pno)].append(box.bbox)
                    count += 1
                    if count >= max_native_tables:
                        break
                if count >= max_native_tables:
                    break
            if count >= max_native_tables:
                break

    pl_backend = get_backend("table", "pdfplumber")
    tatr_backend = get_backend("table", "tatr-onnx", device=device)
    auto_backend = get_backend("table", "auto", device=device)

    pl_confs: list[float] = []
    tatr_confs: list[float] = []
    auto_refined_confs: list[float] = []
    auto_tatr_confs: list[float] = list(e2e_auto_tatr_confs)
    paired_count = 0
    paired_violations = 0

    by_pdf: dict[Path, list[tuple[int, list[tuple[float, float, float, float]]]]] = defaultdict(list)
    for (pdf_path, pno), boxes in sorted(cases_by_page.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        by_pdf[pdf_path].append((pno, boxes))

    for pdf_path, page_entries in by_pdf.items():
        with pdfplumber.open(str(pdf_path)) as doc:
            for pno, boxes in page_entries:
                page = doc.pages[pno - 1]
                words = extract_words(page)
                pl_tbls = extract_tables_on_page(page, boxes, words=words, extractor=pl_backend, device=device)
                tatr_tbls = extract_tables_on_page(page, boxes, words=words, extractor=tatr_backend, device=device)
                auto_tbls = extract_tables_on_page(page, boxes, words=words, extractor=auto_backend, device=device)

                for pl_t, tt_t, au_t in zip(pl_tbls, tatr_tbls, auto_tbls):
                    pl_ok = pl_t.num_rows > 0 and pl_t.num_cols > 0
                    tt_ok = tt_t.num_rows > 0 and tt_t.num_cols > 0
                    if pl_ok:
                        pl_confs.append(float(pl_t.confidence))
                    if tt_ok:
                        tatr_confs.append(float(tt_t.confidence))
                    if pl_ok and tt_ok:
                        paired_count += 1
                        if float(pl_t.confidence) < float(tt_t.confidence):
                            paired_violations += 1
                    if au_t.num_rows > 0 and au_t.num_cols > 0:
                        if au_t.source == "auto_refined":
                            auto_refined_confs.append(float(au_t.confidence))
                        elif au_t.source == "auto_tatr":
                            auto_tatr_confs.append(float(au_t.confidence))

    pl_mean = _mean_or_none(pl_confs)
    tt_mean = _mean_or_none(tatr_confs)
    ar_mean = _mean_or_none(auto_refined_confs)
    at_mean = _mean_or_none(auto_tatr_confs)
    paired_ok = paired_count > 0 and paired_violations == 0
    has_pair = (ar_mean is not None and at_mean is not None) or (pl_mean is not None and tt_mean is not None)
    auto_source_ok = True if (ar_mean is None or at_mean is None) else ar_mean >= at_mean
    direct_source_ok = True if (pl_mean is None or tt_mean is None) else pl_mean >= tt_mean
    source_ok = has_pair and auto_source_ok and direct_source_ok
    return {
        "native_tables_checked": sum(len(b) for b in cases_by_page.values()),
        "paired_count": paired_count,
        "paired_violations": paired_violations,
        "paired_monotonicity_ok": paired_ok,
        "pdfplumber_mean_conf": pl_mean,
        "pdfplumber_min_conf": min(pl_confs) if pl_confs else None,
        "tatr_onnx_mean_conf": tt_mean,
        "tatr_onnx_max_conf": max(tatr_confs) if tatr_confs else None,
        "auto_refined_count": len(auto_refined_confs),
        "auto_refined_mean_conf": ar_mean,
        "auto_tatr_count": len(auto_tatr_confs),
        "auto_tatr_mean_conf": at_mean,
        "source_mean_monotonicity_ok": source_ok,
    }


def eval_tables_calibration(
    *,
    split: str = "dev",
    device: str = "auto",
    layout: str = "heron-onnx",
    ocr_backend: str = "rapidocr-onnx",
    limit: int | None = 25,
    table_gt_path: str | Path | None = None,
) -> dict[str, Any]:
    from pdf import extract_ir
    from pdf.eval.tables import load_omnidoc_tables, teds_score

    samples = load_omnidoc_tables(language="english", split=split, limit=limit, whole_pages=True)
    if not samples:
        raise FileNotFoundError(f"OmniDocBench table samples not found for split={split!r}")

    by_img: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_img[sample["image_path"]].append(sample)
    img_paths = list(by_img.keys())

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "omnidoc_tables_cal.pdf"
        build_raster_pdf(img_paths, pdf_path)
        ir = extract_ir(
            pdf_path,
            figures=True,
            ocr=True,
            tables=True,
            layout=layout,
            table_backend="auto",
            ocr_backend=ocr_backend,
            device=device,
            structure=False,
        )

    preds_by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for tbl in ir.get("tables", ()):
        preds_by_page[int(tbl["page"])].append(tbl)

    all_confs: list[float] = []
    all_labels: list[int] = []
    tp_confs: list[float] = []
    fp_confs: list[float] = []
    cell_confs: list[float] = []
    auto_tatr_confs: list[float] = []
    teds_struct_matched: list[float] = []
    teds_full_matched: list[float] = []
    total_fn = 0

    for pno, img_path in enumerate(img_paths, start=1):
        gt_items = by_img[img_path]
        pred_items = preds_by_page.get(pno, [])
        pred_boxes = [list(p["bbox"]) for p in pred_items]
        gt_boxes = [[float(v) * PT_PER_PX_150DPI for v in g["bbox"]] for g in gt_items]

        # Track 1-to-1 greedy IoU >= 0.5 pairs so matched TEDS and TP/FP calibration stay aligned.
        cands = sorted(
            (
                (-matched_score, pi, gi)
                for pi, pb in enumerate(pred_boxes)
                for gi, gb in enumerate(gt_boxes)
                if (matched_score := iou(pb, gb)) >= 0.5
            ),
            key=lambda item: (item[0], item[1], item[2]),
        )
        used_p: set[int] = set()
        used_g: set[int] = set()
        for _, pi, gi in cands:
            if pi in used_p or gi in used_g:
                continue
            used_p.add(pi)
            used_g.add(gi)
            teds_struct_matched.append(teds_score(pred_items[pi]["html"], gt_items[gi]["html"], structure_only=True))
            teds_full_matched.append(teds_score(pred_items[pi]["html"], gt_items[gi]["html"], structure_only=False))

        total_fn += len(gt_items) - len(used_g)
        for idx, pred in enumerate(pred_items):
            conf = float(pred["confidence"])
            all_confs.append(conf)
            if pred.get("source") == "auto_tatr":
                auto_tatr_confs.append(conf)
            if idx in used_p:
                all_labels.append(1)
                tp_confs.append(conf)
            else:
                all_labels.append(0)
                fp_confs.append(conf)
            for cell in pred.get("cells", ()):
                if "confidence" in cell and cell["confidence"] is not None:
                    cell_confs.append(float(cell["confidence"]))

    mono = _check_table_source_monotonicity(
        split=split,
        device=device,
        layout=layout,
        table_gt_path=table_gt_path,
        e2e_auto_tatr_confs=auto_tatr_confs,
    )
    tp_n, fp_n = len(tp_confs), len(fp_confs)
    p, r, f1 = prf(tp_n, fp_n, total_fn)
    return {
        "task": "tables_calibration",
        "split": split,
        "pages": len(img_paths),
        "gt_tables": len(samples),
        "pred_tables": len(all_confs),
        "tp": tp_n,
        "fp": fp_n,
        "fn": total_fn,
        "det_precision": round(p, 4),
        "det_recall": round(r, 4),
        "det_f1": round(f1, 4),
        "teds_struct_matched": _mean_or_none(teds_struct_matched),
        "teds_full_matched": _mean_or_none(teds_full_matched),
        "cell_confidences": {
            "count": len(cell_confs),
            "mean": _mean_or_none(cell_confs),
            "min": min(cell_confs) if cell_confs else None,
            "max": max(cell_confs) if cell_confs else None,
            "low_conf_lt_070": sum(1 for c in cell_confs if c < 0.70),
        },
        "warnings_count": len(ir.get("metadata", {}).get("warnings", ())),
        "monotonicity": mono,
        "wall_sec": round(time.perf_counter() - t0, 4),
        **_summarize_calibration(all_confs, all_labels, tp_confs, fp_confs),
    }


def _build_synthetic_suspect_ocr_pdf(dst: Path) -> None:
    from PIL import Image, ImageDraw

    from pdf._test_pdf import TestPdfBuilder

    builder = TestPdfBuilder()
    p1 = builder.new_page(width=595.0, height=842.0)
    img = Image.new("RGB", (595, 842), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((72, 100), "AUTOSAR calibration suspect OCR layer text.", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    p1.insert_image((0.0, 0.0, 595.0, 842.0), stream=buf.getvalue())
    p1.insert_text((72.0, 72.0), "hidden tiny ocr text words", fontsize=2.0)

    p2 = builder.new_page(width=595.0, height=842.0)
    img2 = Image.new("RGB", (595, 842), (255, 255, 255))
    draw2 = ImageDraw.Draw(img2)
    draw2.text((72, 120), "Second suspect OCR fallback verification line.", fill=(0, 0, 0))
    buf2 = io.BytesIO()
    img2.save(buf2, format="PNG")
    p2.insert_image((0.0, 0.0, 595.0, 842.0), stream=buf2.getvalue())
    p2.insert_text((72.0, 72.0), "\ufffd\ufffd\ufffd bad ocr text layer", fontsize=10.0)
    builder.save(dst)


def eval_monotonicity_invariants(
    *,
    split: str = "dev",
    device: str = "auto",
    ocr_backend: str = "rapidocr-onnx",
    router_gt_path: str | Path | None = None,
    max_native_pages: int = 8,
) -> dict[str, Any]:
    from pdf import extract_ir

    t0 = time.perf_counter()
    native_confs: list[float] = []
    suspect_confs: list[float] = []
    suspect_warnings = 0
    low_conf_elem_count = 0
    low_conf_pages_total: set[tuple[str, int]] = set()
    low_conf_pages_warned: set[tuple[str, int]] = set()

    router_gt = load_router_gt(split=split, gt_path=router_gt_path)
    used_router_gt = bool(router_gt)

    if router_gt:
        pages_by_pdf: dict[Path, list[dict[str, Any]]] = defaultdict(list)
        for entry in router_gt:
            p = Path(entry["pdf_path"])
            if p.is_file():
                pages_by_pdf[p].append(entry)
        for pdf_path, entries in pages_by_pdf.items():
            target_pages = {int(e["page"]): e for e in entries}
            ir = extract_ir(
                pdf_path,
                figures=False,
                ocr=True,
                tables=False,
                diagrams=False,
                ocr_backend=ocr_backend,
                device=device,
                structure=False,
            )
            warn_pages = {
                int(w["page"])
                for w in ir.get("metadata", {}).get("warnings", ())
                if w.get("page") is not None
            }
            suspect_warnings += sum(
                1 for w in ir.get("metadata", {}).get("warnings", ()) if w.get("code") == "suspect_ocr_layer"
            )
            for elem in ir.get("elements", ()):
                pno = int(elem.get("page", 0))
                if pno not in target_pages or elem.get("type") != "text":
                    continue
                conf = float(elem["confidence"])
                if conf < 0.70:
                    low_conf_elem_count += 1
                    low_conf_pages_total.add((pdf_path.name, pno))
                    if pno in warn_pages:
                        low_conf_pages_warned.add((pdf_path.name, pno))
                gt_e = target_pages[pno]
                if gt_e["kind"] == "native" and not gt_e["suspect_ocr_layer"]:
                    native_confs.append(conf)
                elif gt_e["kind"] == "hybrid" and gt_e["suspect_ocr_layer"]:
                    suspect_confs.append(conf)
    else:
        autosar_docs = load_autosar_gt(split=split)
        checked_pages = 0
        for doc_gt in autosar_docs:
            if not doc_gt.pdf_path.is_file() or checked_pages >= max_native_pages:
                continue
            ir = extract_ir(
                doc_gt.pdf_path,
                figures=False,
                ocr=False,
                tables=False,
                diagrams=False,
                device=device,
                structure=False,
            )
            sample_pages = set(doc_gt.pages[: max(1, max_native_pages - checked_pages)])
            checked_pages += len(sample_pages)
            for elem in ir.get("elements", ()):
                if int(elem.get("page", 0)) in sample_pages and elem.get("type") == "text":
                    native_confs.append(float(elem["confidence"]))

        with tempfile.TemporaryDirectory() as tmp:
            syn_pdf = Path(tmp) / "synthetic_suspect_ocr.pdf"
            _build_synthetic_suspect_ocr_pdf(syn_pdf)
            ir_syn = extract_ir(
                syn_pdf,
                figures=False,
                ocr=True,
                tables=False,
                diagrams=False,
                ocr_backend=ocr_backend,
                device=device,
                structure=False,
            )
            warn_list = ir_syn.get("metadata", {}).get("warnings", ())
            warn_pages = {int(w["page"]) for w in warn_list if w.get("page") is not None}
            suspect_warnings = sum(1 for w in warn_list if w.get("code") == "suspect_ocr_layer")
            for elem in ir_syn.get("elements", ()):
                if elem.get("type") != "text":
                    continue
                conf = float(elem["confidence"])
                suspect_confs.append(conf)
                if conf < 0.70:
                    pno = int(elem.get("page", 0))
                    low_conf_elem_count += 1
                    low_conf_pages_total.add((syn_pdf.name, pno))
                    if pno in warn_pages:
                        low_conf_pages_warned.add((syn_pdf.name, pno))

    native_ok = bool(native_confs) and all(c == 1.0 for c in native_confs)
    suspect_ok = bool(suspect_confs) and all(c <= 0.85 for c in suspect_confs)
    strict_order_ok = bool(native_confs and suspect_confs) and min(native_confs) > max(suspect_confs)
    return {
        "task": "monotonicity_invariants",
        "split": split,
        "used_router_gt": used_router_gt,
        "native_text_count": len(native_confs),
        "native_min_conf": min(native_confs) if native_confs else None,
        "native_max_conf": max(native_confs) if native_confs else None,
        "native_all_eq_1_0": native_ok,
        "suspect_ocr_text_count": len(suspect_confs),
        "suspect_ocr_min_conf": min(suspect_confs) if suspect_confs else None,
        "suspect_ocr_max_conf": max(suspect_confs) if suspect_confs else None,
        "suspect_ocr_mean_conf": _mean_or_none(suspect_confs),
        "suspect_ocr_all_le_085": suspect_ok,
        "native_gt_suspect": strict_order_ok,
        "suspect_warnings_emitted": suspect_warnings,
        "low_conf_lt_070_elements": low_conf_elem_count,
        "low_conf_pages_total": len(low_conf_pages_total),
        "low_conf_pages_with_warning": len(low_conf_pages_warned),
        "invariant_ok": native_ok and suspect_ok and strict_order_ok,
        "wall_sec": round(time.perf_counter() - t0, 4),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate Step 4 confidence calibration and monotonicity invariants.")
    ap.add_argument("--task", default="all", choices=["all", "figures", "tables", "monotonicity"])
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--final", action="store_true", help="Required when --split test")
    ap.add_argument("--layout", default="heron-onnx", help="Layout detector backend name")
    ap.add_argument("--grouper", default="v1_anchor", help="Figure grouper name")
    ap.add_argument("--ocr-backend", default="rapidocr-onnx", help="OCR backend name")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--match-mode", default="strict", choices=["strict", "lenient"])
    ap.add_argument("--gt", default=None, help="Optional path to autosar_figures_v2.json")
    ap.add_argument("--router-gt", default=None, help="Optional path to custom router_gt.json")
    ap.add_argument("--table-gt", default=None, help="Optional path to custom autosar_tables_gt.json")
    ap.add_argument("--cache-dir", default=None, help="Optional cache directory for figure layout detection")
    ap.add_argument("--table-limit", type=int, default=25, help="Cap OmniDocBench dev tables (0 = all)")
    ap.add_argument("--log-file", default=str(ROOT / "outputs" / "eval_runs.jsonl"))
    ap.add_argument("--no-log", action="store_true", help="Skip appending run record to eval_runs.jsonl")
    args = ap.parse_args(argv)

    if not check_split_guard(args.split, args.final):
        return 2

    cap = None if args.table_limit == 0 else args.table_limit
    results: list[dict[str, Any]] = []
    if args.task in ("figures", "all"):
        res = eval_figures_calibration(
            split=args.split,
            layout=args.layout,
            grouper=args.grouper,
            device=args.device,
            match_mode=args.match_mode,
            gt_path=args.gt,
            cache_dir=args.cache_dir,
        )
        print(json.dumps(res, ensure_ascii=False))
        results.append(res)
    if args.task in ("tables", "all"):
        res = eval_tables_calibration(
            split=args.split,
            device=args.device,
            layout=args.layout,
            ocr_backend=args.ocr_backend,
            limit=cap,
            table_gt_path=args.table_gt,
        )
        print(json.dumps(res, ensure_ascii=False))
        results.append(res)
    if args.task in ("monotonicity", "all"):
        res = eval_monotonicity_invariants(
            split=args.split,
            device=args.device,
            ocr_backend=args.ocr_backend,
            router_gt_path=args.router_gt,
        )
        print(json.dumps(res, ensure_ascii=False))
        results.append(res)

    if results and not args.no_log:
        append_run_log(
            Path(args.log_file),
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task": f"calibration:{args.task}",
                "split": args.split,
                "final": args.final,
                "results": results,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
