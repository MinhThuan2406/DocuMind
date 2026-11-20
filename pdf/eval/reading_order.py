from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Hashable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pdf._test_pdf import TestPdfBuilder  # noqa: E402
from pdf.eval.gt import (  # noqa: E402
    OMNIDOC_SAMPLE_PAGES,
    PT_PER_PX_150DPI,
    append_run_log,
    build_raster_pdf,
    omnidoc_image_split,
)
from pdf.eval.metrics import area, inter, iou, median  # noqa: E402

_BLK_TAG_RE = re.compile(r"\bBLK_\d{2}\b")
_EXCLUDED_SAMPLE_PAGES = frozenset(OMNIDOC_SAMPLE_PAGES)


def _dedupe_preserve(items: Sequence[Hashable]) -> list[Hashable]:
    seen: set[Hashable] = set()
    out: list[Hashable] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def kendall_tau_matched(pred_order: Sequence[Hashable], gt_order: Sequence[Hashable]) -> float:
    pred_seq = _dedupe_preserve(pred_order)
    gt_seq = _dedupe_preserve(gt_order)
    if not pred_seq and not gt_seq:
        return 1.0
    gt_pos = {item: idx for idx, item in enumerate(gt_seq)}
    matched = [gt_pos[item] for item in pred_seq if item in gt_pos]
    n = len(matched)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0 if len(gt_seq) == 1 else 0.0
    concordant = 0
    discordant = 0
    for i in range(n):
        pi = matched[i]
        for j in range(i + 1, n):
            if pi < matched[j]:
                concordant += 1
            else:
                discordant += 1
    total_pairs = n * (n - 1) // 2
    return round((concordant - discordant) / total_pairs, 4)


def kendall_tau_penalized(pred_order: Sequence[Hashable], gt_order: Sequence[Hashable]) -> float:
    pred_seq = _dedupe_preserve(pred_order)
    gt_seq = _dedupe_preserve(gt_order)
    union_n = len(set(pred_seq) | set(gt_seq))
    if union_n == 0:
        return 1.0
    gt_pos = {item: idx for idx, item in enumerate(gt_seq)}
    matched = [gt_pos[item] for item in pred_seq if item in gt_pos]
    if union_n == 1:
        return 1.0 if len(matched) == 1 else -1.0
    concordant = 0
    n = len(matched)
    for i in range(n):
        pi = matched[i]
        for j in range(i + 1, n):
            if pi < matched[j]:
                concordant += 1
    union_pairs = union_n * (union_n - 1) // 2
    return round((2.0 * concordant) / union_pairs - 1.0, 4)


def load_omnidoc_reading_order(
    split: str = "dev",
    *,
    language: str = "english",
    limit: int | None = None,
    gt_file: Path | None = None,
    img_dir: Path | None = None,
    require_images: bool = True,
) -> list[dict[str, Any]]:
    if split not in ("dev", "test"):
        raise ValueError(f"unknown split: {split!r}")
    resolved_gt = gt_file if gt_file is not None else (ROOT / "datasets" / "OmniDocBench" / "OmniDocBench.json")
    resolved_imgs = img_dir if img_dir is not None else (ROOT / "datasets" / "OmniDocBench" / "images")
    raw = json.loads(resolved_gt.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for doc in raw:
        info = doc.get("page_info", {})
        attrs = info.get("page_attribute", {})
        if language and attrs.get("language") != language:
            continue
        img_name = str(info.get("image_path", ""))
        if not img_name or img_name in _EXCLUDED_SAMPLE_PAGES:
            continue
        if omnidoc_image_split(img_name) != split:
            continue
        img_path = resolved_imgs / img_name
        if require_images and not img_path.is_file():
            continue

        blocks: list[dict[str, Any]] = []
        used_ids: set[Any] = set()
        for idx, det in enumerate(doc.get("layout_dets", ())):
            if det.get("ignore"):
                continue
            order_val = det.get("order")
            if order_val is None:
                continue
            poly = det.get("poly") or ()
            if len(poly) < 8:
                continue
            xs, ys = poly[0::2], poly[1::2]
            bid = det.get("anno_id", idx)
            if bid in used_ids:
                bid = f"{bid}_{idx}"
            used_ids.add(bid)
            blocks.append({
                "id": bid,
                "order": int(order_val),
                "bbox": (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))),
                "category_type": str(det.get("category_type", "")),
                "text": str(det.get("text", "")),
                "_idx": idx,
            })
        if len(blocks) < 2:
            continue
        blocks.sort(key=lambda b: (b["order"], b["_idx"]))
        for b in blocks:
            b.pop("_idx", None)
        out.append({
            "image_name": img_name,
            "image_path": img_path,
            "width": info.get("width"),
            "height": info.get("height"),
            "layout": attrs.get("layout", ""),
            "blocks": blocks,
        })
        if limit is not None and len(out) >= limit:
            break
    return out


def build_reading_order_fixtures(dst_dir: Path) -> list[dict[str, Any]]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        (
            "two_column",
            [
                ("BLK_01", (45.0, 80.0), "BLK_01 Left column first paragraph."),
                ("BLK_02", (45.0, 150.0), "BLK_02 Left column second paragraph."),
                ("BLK_03", (45.0, 220.0), "BLK_03 Left column third paragraph."),
                ("BLK_04", (320.0, 80.0), "BLK_04 Right column first paragraph."),
                ("BLK_05", (320.0, 150.0), "BLK_05 Right column second paragraph."),
                ("BLK_06", (320.0, 220.0), "BLK_06 Right column third paragraph."),
            ],
        ),
        (
            "banner_two_column",
            [
                ("BLK_01", (45.0, 60.0), "BLK_01 Full Width Document Title Spanning Both Columns Clearly"),
                ("BLK_02", (45.0, 140.0), "BLK_02 Left body upper block."),
                ("BLK_03", (45.0, 210.0), "BLK_03 Left body lower block."),
                ("BLK_04", (320.0, 140.0), "BLK_04 Right body upper block."),
                ("BLK_05", (320.0, 210.0), "BLK_05 Right body lower block."),
                ("BLK_06", (45.0, 300.0), "BLK_06 Full Width Concluding Summary Spanning Across Page"),
            ],
        ),
        (
            "three_column",
            [
                ("BLK_01", (36.0, 80.0), "BLK_01 Col one top."),
                ("BLK_02", (36.0, 160.0), "BLK_02 Col one low."),
                ("BLK_03", (220.0, 80.0), "BLK_03 Col two top."),
                ("BLK_04", (220.0, 160.0), "BLK_04 Col two low."),
                ("BLK_05", (404.0, 80.0), "BLK_05 Col three top."),
                ("BLK_06", (404.0, 160.0), "BLK_06 Col three low."),
            ],
        ),
    ]
    fixtures: list[dict[str, Any]] = []
    for name, items in specs:
        builder = TestPdfBuilder()
        page = builder.new_page(width=595.0, height=842.0)
        gt_order: list[str] = []
        for tag, pos, text in items:
            page.insert_text(pos, text, fontsize=11.0)
            gt_order.append(tag)
        pdf_path = dst_dir / f"{name}.pdf"
        builder.save(pdf_path)
        fixtures.append({"name": name, "pdf_path": pdf_path, "gt_order": gt_order})
    return fixtures


def bench_synthetic_reading_order() -> dict[str, Any]:
    from pdf import extract_ir

    taus_matched: list[float] = []
    taus_penalized: list[float] = []
    per_fixture: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        fixtures = build_reading_order_fixtures(Path(tmp))
        for fix in fixtures:
            ir = extract_ir(fix["pdf_path"], figures=False, ocr=False, tables=False, structure=False)
            pred_order: list[str] = []
            for elem in sorted(ir["elements"], key=lambda e: (int(e["page"]), int(e["reading_order"]))):
                for tag in _BLK_TAG_RE.findall(str(elem.get("text", ""))):
                    pred_order.append(tag)
            gt_order = fix["gt_order"]
            tm = kendall_tau_matched(pred_order, gt_order)
            tp = kendall_tau_penalized(pred_order, gt_order)
            taus_matched.append(tm)
            taus_penalized.append(tp)
            per_fixture.append({"name": fix["name"], "tau_matched": tm, "tau_penalized": tp, "matched_n": len(pred_order), "gt_n": len(gt_order)})

    return {
        "dataset": "synthetic_multicolumn",
        "mode": "e2e_native",
        "split": "synthetic",
        "pages": len(per_fixture),
        "mean_tau_matched": round(sum(taus_matched) / max(1, len(taus_matched)), 4),
        "mean_tau_penalized": round(sum(taus_penalized) / max(1, len(taus_penalized)), 4),
        "median_tau_matched": round(median(taus_matched), 4),
        "wall_sec": round(time.perf_counter() - t0, 4),
        "fixtures": per_fixture,
    }


def _match_elements_to_gt(
    pred_elems: Sequence[dict[str, Any]],
    gt_blocks: Sequence[dict[str, Any]],
    *,
    min_iou: float = 0.1,
    min_containment: float = 0.6,
) -> list[Any]:
    top_elems = [el for el in pred_elems if int(el.get("level", 1)) == 1]
    gt_boxes_pt = [[v * PT_PER_PX_150DPI for v in b["bbox"]] for b in gt_blocks]
    candidates = sorted(
        (
            (-iou(el["bbox"], gb), ei, gi)
            for ei, el in enumerate(top_elems)
            for gi, gb in enumerate(gt_boxes_pt)
            if iou(el["bbox"], gb) >= min_iou
        ),
        key=lambda item: item[0],
    )
    used_e: set[int] = set()
    used_g: set[int] = set()
    elem_to_gts: dict[int, list[int]] = defaultdict(list)
    for _, ei, gi in candidates:
        if ei in used_e or gi in used_g:
            continue
        used_e.add(ei)
        used_g.add(gi)
        elem_to_gts[ei].append(gi)

    # Allow a single coarser element (e.g. multi-line formula or figure) to cover multiple GT blocks.
    for gi, gb in enumerate(gt_boxes_pt):
        if gi in used_g:
            continue
        g_area = area(gb)
        if g_area <= 0.0:
            continue
        best_ei: int | None = None
        best_key: tuple[float, float, float] | None = None
        for ei, el in enumerate(top_elems):
            if el.get("type") not in ("figure", "table") and not str(el.get("text", "")).strip():
                continue
            ioa = inter(el["bbox"], gb) / g_area
            if ioa < min_containment:
                continue
            key = (area(el["bbox"]), -ioa, -iou(el["bbox"], gb))
            if best_key is None or key < best_key:
                best_key = key
                best_ei = ei
        if best_ei is not None:
            used_g.add(gi)
            elem_to_gts[best_ei].append(gi)

    ordered_pred: list[Any] = []
    for ei in range(len(top_elems)):
        if ei in elem_to_gts:
            gis = sorted(
                elem_to_gts[ei],
                key=lambda gi: (round(gt_boxes_pt[gi][1] / 6.0), round(gt_boxes_pt[gi][0], 1), gi),
            )
            ordered_pred.extend(gt_blocks[gi]["id"] for gi in gis)
        else:
            ordered_pred.append(("unmatched", ei))
    return ordered_pred


def bench_omnidoc_reading_order(
    *,
    split: str = "dev",
    mode: str = "oracle",
    limit: int | None = 50,
    device: str = "auto",
    ocr_backend: str = "rapidocr-onnx",
) -> dict[str, Any]:
    if mode not in ("oracle", "e2e"):
        raise ValueError(f"unknown mode: {mode!r}")

    from pdf.backends.rss import peak_rss_mb

    pages = load_omnidoc_reading_order(split=split, limit=limit)
    if not pages:
        raise FileNotFoundError(f"OmniDocBench reading-order pages not found for split={split!r}")

    taus_matched: list[float] = []
    taus_penalized: list[float] = []
    t0 = time.perf_counter()

    if mode == "oracle":
        from pdf.text import xy_cut_order

        for pg in pages:
            blocks = pg["blocks"]
            gt_order = [b["id"] for b in blocks]
            ordered = xy_cut_order(
                blocks,
                lambda b: (
                    b["bbox"][0] * PT_PER_PX_150DPI,
                    b["bbox"][1] * PT_PER_PX_150DPI,
                    b["bbox"][2] * PT_PER_PX_150DPI,
                    b["bbox"][3] * PT_PER_PX_150DPI,
                ),
            )
            pred_order = [b["id"] for b in ordered]
            taus_matched.append(kendall_tau_matched(pred_order, gt_order))
            taus_penalized.append(kendall_tau_penalized(pred_order, gt_order))
    else:
        from pdf import extract_ir

        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "omnidoc_ro.pdf"
            build_raster_pdf([pg["image_path"] for pg in pages], pdf_path)
            ir = extract_ir(
                pdf_path,
                figures=True,
                ocr=True,
                tables=False,
                ocr_backend=ocr_backend,
                device=device,
                structure=False,
            )
        by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for el in ir["elements"]:
            by_page[int(el["page"])].append(el)
        for pno, pg in enumerate(pages, start=1):
            gt_order = [b["id"] for b in pg["blocks"]]
            pred_elems = sorted(by_page.get(pno, ()), key=lambda e: int(e["reading_order"]))
            pred_order = _match_elements_to_gt(pred_elems, pg["blocks"])
            taus_matched.append(kendall_tau_matched(pred_order, gt_order))
            taus_penalized.append(kendall_tau_penalized(pred_order, gt_order))

    wall_sec = round(time.perf_counter() - t0, 4)
    return {
        "dataset": "OmniDocBench_en_reading_order",
        "mode": mode,
        "split": split,
        "pages": len(pages),
        "mean_tau_matched": round(sum(taus_matched) / len(taus_matched), 4),
        "mean_tau_penalized": round(sum(taus_penalized) / len(taus_penalized), 4),
        "median_tau_matched": round(median(taus_matched), 4),
        "median_tau_penalized": round(median(taus_penalized), 4),
        "wall_sec": wall_sec,
        "ram_mb": peak_rss_mb(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark reading order via Kendall's tau on synthetic fixtures and OmniDocBench.")
    ap.add_argument("--task", default="all", choices=["synthetic", "omnidoc", "all"])
    ap.add_argument("--mode", default="oracle", choices=["oracle", "e2e"])
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--final", action="store_true", help="Required when --split test")
    ap.add_argument("--limit", type=int, default=50, help="Cap OmniDocBench pages (None via 0)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--log-file", default=str(ROOT / "outputs" / "eval_runs.jsonl"))
    args = ap.parse_args(argv)

    if args.split == "test" and not args.final:
        print("REFUSED: --split test requires --final to prevent accidental test-split evaluation.", file=sys.stderr)
        return 2

    results: list[dict[str, Any]] = []
    if args.task in ("synthetic", "all"):
        syn = bench_synthetic_reading_order()
        print(json.dumps(syn, ensure_ascii=False))
        results.append(syn)
    if args.task in ("omnidoc", "all"):
        gt_json = ROOT / "datasets" / "OmniDocBench" / "OmniDocBench.json"
        if gt_json.is_file():
            cap = None if args.limit == 0 else args.limit
            omni = bench_omnidoc_reading_order(split=args.split, mode=args.mode, limit=cap, device=args.device)
            print(json.dumps(omni, ensure_ascii=False))
            results.append(omni)
        elif args.task == "omnidoc":
            raise FileNotFoundError(f"OmniDocBench not found at {gt_json}")

    append_run_log(
        Path(args.log_file),
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": f"reading_order:{args.task}",
            "mode": args.mode,
            "split": args.split,
            "final": args.final,
            "results": results,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
