from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pdf.backends import get as get_backend, registered  # noqa: E402
from pdf.backends.rss import peak_rss_mb  # noqa: E402
from pdf.eval.gt import append_run_log, build_raster_pdf, load_autosar_gt, load_omnidoc_gt  # noqa: E402
from pdf.eval.metrics import char_counts, match_count, median, prf, strip_markup  # noqa: E402


def _eval_figures_worker(
    backend_name: str,
    grouper_name: str,
    split: str,
    device: str,
    gt_path: str | None,
    cache_dir: str | None,
    no_cache: bool,
) -> dict[str, Any]:
    from pdf.figures import detect_figures

    docs = load_autosar_gt(split=split, gt_path=gt_path)
    totals = {m: [0, 0, 0] for m in ("strict", "lenient")}
    neg_fp = 0
    total_pages = 0
    total_neg_pages = 0
    total_gt = 0
    per_doc: list[dict[str, Any]] = []

    load_sec = 0.0
    infer_sec = 0.0
    infer_pages = 0
    child_ram_mb = 0.0
    used_stats = False

    with tempfile.TemporaryDirectory() as tmp:
        tmp_cache = Path(tmp) / "cache"
        t_wall0 = time.perf_counter()
        for doc_gt in docs:
            backend = get_backend("layout", backend_name, device=device)
            if no_cache:
                doc_cache: Path | None = tmp_cache / doc_gt.stem
            elif cache_dir is not None:
                doc_cache = Path(cache_dir) / doc_gt.stem
            else:
                doc_cache = None

            preds = detect_figures(
                doc_gt.pdf_path,
                doc_gt.pages,
                layout=backend,
                grouper=grouper_name,
                device=device,
                include_tables=False,
                cache_dir=doc_cache,
            )
            stats = getattr(backend, "last_stats", None)
            if stats is not None:
                used_stats = True
                load_sec = max(load_sec, stats.load_sec)
                infer_sec += stats.infer_sec
                infer_pages += max(stats.pages, len(doc_gt.pages))
                if not stats.in_process:
                    child_ram_mb = max(child_ram_mb, stats.peak_rss_mb)

            doc_strict = [0, 0, 0]
            doc_lenient = [0, 0, 0]
            doc_neg_fp = 0
            for pno in doc_gt.pages:
                total_pages += 1
                pred_boxes = [list(b.bbox) for b in preds.get(pno, ())]
                gt_boxes = doc_gt.boxes_by_page.get(pno, [])
                total_gt += len(gt_boxes)
                if not gt_boxes:
                    total_neg_pages += 1
                    neg_fp += len(pred_boxes)
                    doc_neg_fp += len(pred_boxes)
                for mode, doc_acc in (("strict", doc_strict), ("lenient", doc_lenient)):
                    m = match_count(pred_boxes, gt_boxes, mode)
                    fp = len(pred_boxes) - m
                    fn = len(gt_boxes) - m
                    totals[mode][0] += m
                    totals[mode][1] += fp
                    totals[mode][2] += fn
                    doc_acc[0] += m
                    doc_acc[1] += fp
                    doc_acc[2] += fn
            per_doc.append({
                "stem": doc_gt.stem,
                "pages": len(doc_gt.pages),
                "strict": doc_strict,
                "lenient": doc_lenient,
                "neg_fp": doc_neg_fp,
            })
        wall_sec = round(time.perf_counter() - t_wall0, 4)

    sp = prf(*totals["strict"])
    lp = prf(*totals["lenient"])
    if used_stats and infer_pages > 0:
        steady_s_p = round(infer_sec / infer_pages, 4)
        ram_mb = child_ram_mb if child_ram_mb > 0 else peak_rss_mb()
    else:
        steady_s_p = round(wall_sec / max(1, total_pages), 4)
        ram_mb = peak_rss_mb()

    return {
        "task": "figures",
        "backend": backend_name,
        "grouper": grouper_name,
        "split": split,
        "pages": total_pages,
        "neg_pages": total_neg_pages,
        "gt_boxes": total_gt,
        "strict": {"tp": totals["strict"][0], "fp": totals["strict"][1], "fn": totals["strict"][2], "p": round(sp[0], 4), "r": round(sp[1], 4), "f1": round(sp[2], 4)},
        "lenient": {"tp": totals["lenient"][0], "fp": totals["lenient"][1], "fn": totals["lenient"][2], "p": round(lp[0], 4), "r": round(lp[1], 4), "f1": round(lp[2], 4)},
        "neg_fp": neg_fp,
        "load_sec": round(load_sec, 4),
        "steady_s_p": steady_s_p,
        "wall_sec": wall_sec,
        "ram_mb": ram_mb,
        "per_doc": per_doc,
    }


def _eval_raster_worker(
    reader_name: str,
    layout_name: str,
    ocr_name: str,
    device: str,
    pages_cap: int | None,
) -> dict[str, Any]:
    from pdf import extract_ir

    selected = load_omnidoc_gt(pages=pages_cap)
    if not selected:
        raise FileNotFoundError("OmniDocBench dataset pages not found")

    reader_obj = get_backend("reader", reader_name, layout=layout_name, ocr_backend=ocr_name, device=device)

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "omnidoc_batch.pdf"
        build_raster_pdf([p for _, p, _ in selected], pdf_path)
        t0 = time.perf_counter()
        ir = extract_ir(
            pdf_path,
            figures=False,
            ocr=True,
            reader=reader_obj,
            layout=layout_name,
            ocr_backend=ocr_name,
            device=device,
        )
        wall_sec = round(time.perf_counter() - t0, 4)

    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for elem in ir["elements"]:
        by_page[int(elem["page"])].append(elem)

    recalls: list[float] = []
    precisions: list[float] = []
    total_matched = 0
    total_gt = 0
    total_pred = 0
    for idx, (_, _, gt_chars) in enumerate(selected, start=1):
        ocr_elems = [e for e in by_page[idx] if e["type"] == "text"]
        assert all(e["source"] == "ocr" and isinstance(e["confidence"], float) for e in ocr_elems)
        pred_raw = "".join(str(e["text"]) for e in ocr_elems)
        pred_chars = char_counts(strip_markup(pred_raw))
        matched = sum((gt_chars & pred_chars).values())
        gt_n = sum(gt_chars.values())
        pred_n = sum(pred_chars.values())
        recalls.append(matched / gt_n if gt_n > 0 else 1.0)
        precisions.append(matched / pred_n if pred_n > 0 else 1.0)
        total_matched += matched
        total_gt += gt_n
        total_pred += pred_n

    stats = getattr(reader_obj, "last_stats", None)
    load_sec = stats.load_sec if stats else 0.0
    infer_sec = stats.infer_sec if stats else wall_sec
    n_pages = stats.pages if stats and stats.pages else len(selected)
    ram_mb = stats.peak_rss_mb if stats and not stats.in_process and stats.peak_rss_mb > 0 else peak_rss_mb()

    return {
        "task": "raster",
        "backend": reader_name,
        "pages": len(selected),
        "pooled_recall": round(total_matched / total_gt if total_gt else 0.0, 4),
        "pooled_precision": round(total_matched / total_pred if total_pred else 0.0, 4),
        "mean_recall": round(sum(recalls) / len(recalls), 4),
        "mean_precision": round(sum(precisions) / len(precisions), 4),
        "median_recall": round(median(recalls), 4),
        "min_recall": round(min(recalls), 4),
        "load_sec": round(load_sec, 4),
        "steady_s_p": round(infer_sec / max(1, n_pages), 4),
        "wall_sec": wall_sec,
        "ram_mb": ram_mb,
    }


def _run_isolated(argv: list[str]) -> dict[str, Any]:
    cmd = [sys.executable, "-B", __file__, "--worker", *argv]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _print_figures_table(rows: list[dict[str, Any]]) -> None:
    first = rows[0]
    print(f"=== Task: figures | split={first['split']} | grouper={first['grouper']} | pages={first['pages']} (neg={first['neg_pages']}) | GT={first['gt_boxes']} ===")
    header = f"{'backend':<16} {'strict TP/FP/FN':<17} {'sF1':>6} {'lenient TP/FP/FN':<18} {'lF1':>6} {'negFP':>6} {'load_s':>7} {'s/p':>7} {'ram_mb':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        st, ln = r["strict"], r["lenient"]
        s_str = f"{st['tp']}/{st['fp']}/{st['fn']}"
        l_str = f"{ln['tp']}/{ln['fp']}/{ln['fn']}"
        print(
            f"{r['backend']:<16} {s_str:<17} {st['f1']:>6.3f} {l_str:<18} {ln['f1']:>6.3f} "
            f"{r['neg_fp']:>6d} {r['load_sec']:>7.3f} {r['steady_s_p']:>7.4f} {r['ram_mb']:>8.1f}"
        )


def _print_raster_table(rows: list[dict[str, Any]]) -> None:
    first = rows[0]
    print(f"=== Task: raster | OmniDocBench pages={first['pages']} ===")
    header = f"{'backend':<16} {'pooled_R':>8} {'pooled_P':>8} {'mean_R':>8} {'mean_P':>8} {'load_s':>7} {'s/p':>7} {'ram_mb':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['backend']:<16} {r['pooled_recall']:>8.4f} {r['pooled_precision']:>8.4f} "
            f"{r['mean_recall']:>8.4f} {r['mean_precision']:>8.4f} "
            f"{r['load_sec']:>7.3f} {r['steady_s_p']:>7.4f} {r['ram_mb']:>8.1f}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare swappable pdf/ backends on AUTOSAR figures or OmniDocBench raster.")
    ap.add_argument("--task", required=True, choices=["figures", "raster"])
    ap.add_argument("--backends", default="", help="Comma-separated backend names")
    ap.add_argument("--grouper", default="v1_anchor", choices=registered("grouper"))
    ap.add_argument("--layout", default="heron-onnx", help="Detector name for composed raster reader")
    ap.add_argument("--ocr-backend", default="rapidocr-onnx", help="OCR name for composed raster reader")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--final", action="store_true", help="Required when --split test")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--gt", default=None, help="Optional path to autosar_figures_v2.json")
    ap.add_argument("--cache-dir", default=None, help="Optional cache root for figures")
    ap.add_argument("--no-cache", action="store_true", help="Run figures in an empty temp cache for live timing")
    ap.add_argument("--pages", type=int, default=None, help="Optional cap on OmniDocBench pages")
    ap.add_argument("--log-file", default=str(ROOT / "outputs" / "eval_runs.jsonl"))
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.split == "test":
        if args.task == "raster":
            print(
                "REFUSED: --task raster evaluates the fixed OmniDocBench dev sample set and has no test split.",
                file=sys.stderr,
            )
            return 2
        if not args.final:
            print("REFUSED: --split test requires --final to prevent accidental test-split evaluation.", file=sys.stderr)
            return 2

    names = [b.strip() for b in args.backends.split(",") if b.strip()]
    if not names:
        names = ["heron", "mineru-pipeline"] if args.task == "figures" else ["composed"]

    if args.worker:
        name = names[0]
        if args.task == "figures":
            res = _eval_figures_worker(
                name, args.grouper, args.split, args.device, args.gt, args.cache_dir, args.no_cache
            )
        else:
            res = _eval_raster_worker(name, args.layout, args.ocr_backend, args.device, args.pages)
        print(json.dumps(res, ensure_ascii=False))
        return 0

    rows: list[dict[str, Any]] = []
    for name in names:
        worker_args = ["--task", args.task, "--backends", name, "--device", args.device]
        if args.task == "figures":
            worker_args.extend(["--grouper", args.grouper, "--split", args.split])
            if args.final:
                worker_args.append("--final")
            if args.gt:
                worker_args.extend(["--gt", args.gt])
            if args.cache_dir:
                worker_args.extend(["--cache-dir", args.cache_dir])
            if args.no_cache:
                worker_args.append("--no-cache")
        else:
            worker_args.extend(["--layout", args.layout, "--ocr-backend", args.ocr_backend])
            if args.pages is not None:
                worker_args.extend(["--pages", str(args.pages)])
        rows.append(_run_isolated(worker_args))

    if args.task == "figures":
        _print_figures_table(rows)
    else:
        _print_raster_table(rows)

    append_run_log(
        Path(args.log_file),
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": args.task,
            "split": args.split if args.task == "figures" else "omnidoc",
            "final": args.final,
            "backends": names,
            "results": rows,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
