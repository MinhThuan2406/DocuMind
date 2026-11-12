"""
End-to-end reading-order evaluation on OmniDocBench (Docling -> ordering strategies).

Per page:
  1. Docling (DocumentConverter with default options, same call as Tier 2 of
     run_dataset_stress_test.py). Boxes are converted from Docling page units
     (points, bottom-left origin) to image-pixel top-left coordinates and cached in
     <out-dir>/cache/<image_stem>.json. Cached pages are never sent to Docling again.
  2. One-to-one matching between Docling boxes and GT boxes (Hungarian on IoU,
     pairs with IoU >= --iou-thr only). GT filter = Tier 1 filter (gt_elements_from_page).
  3. Four ordering methods, all evaluated on the SAME pages with the SAME matching:
       A  docling_native     Docling's own iterate_items() order
       B  documind_xycut     ir_postprocessor.sort_elements_reading_order on Docling boxes
       C  naive_yx           sort Docling boxes by (y_min, x_min)
       D  oracle_xycut_gt    sort_elements_reading_order on GT boxes (reference only)
  4. Metrics per (page, method):
       tau_matched    Kendall's tau over matched GT/prediction pairs only (old Tier 2 style).
       coverage       matched GT / all GT.
       tau_penalized  Kendall's tau over ALL GT elements. Each unmatched GT is given predicted
                      key N (N = number of predicted elements), i.e. placed after every predicted
                      element. Unmatched GT tie with each other, so their mutual pairs add 0.
       A page with fewer than 2 defined elements gives None (excluded, counted separately);
       it is never scored as 1.0.

Usage:
  .venv\\Scripts\\python scripts\\run_e2e_reading_order_eval.py --stratified 300
  .venv\\Scripts\\python scripts\\run_e2e_reading_order_eval.py --all
  .venv\\Scripts\\python scripts\\run_e2e_reading_order_eval.py --limit 5
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from eval_utils import (  # noqa: E402
    compute_kendall_tau,
    docling_bbox_to_pixel,
    gt_elements_from_page,
    match_one_to_one,
)
from ir_postprocessor import sort_elements_reading_order  # noqa: E402

ROOT = SCRIPT_DIR.parent
DATASET_JSON = ROOT / "datasets/OmniDocBench/OmniDocBench.json"
IMAGES_DIR = ROOT / "datasets/OmniDocBench/images"
DEFAULT_OUT_DIR = ROOT / "outputs/04_e2e_eval"

METHOD_NAMES = {
    "A": "docling_native",
    "B": "documind_xycut",
    "C": "naive_yx",
    "D": "oracle_xycut_gt",
}
METRICS = ("tau_matched", "tau_penalized")
MIN_GROUP_PAGES_SPECIAL = 4
DEFAULT_STRATIFIED_N = 300
BOOTSTRAP_RESAMPLES = 2000


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self._f.write(line + "\n")
        self._f.flush()


# ----------------------------------------------------------------------
# Docling step (cached)
# ----------------------------------------------------------------------
def save_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def load_cache(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None  # corrupt cache file: treat as missing


def _needs_docling(cache_dir: Path, item: Dict[str, Any]) -> bool:
    c = load_cache(cache_dir / f"{Path(item['page_info']['image_path']).stem}.json")
    return c is None or c.get("status") != "ok"


def build_converter(ocr_backend: str, device: str):
    """
    DocumentConverter for PNG pages. Only the OCR engine and the accelerator device differ
    from Docling's defaults; everything else is ImageFormatOption()'s default PdfPipelineOptions.

    ocr_backend: 'auto' (Docling picks RapidOCR/onnxruntime here, same as the legacy cache),
                 'onnxruntime' or 'torch' (RapidOcrOptions with that backend, default lang/model).
    device:      'cpu', 'cuda' or 'auto' (AcceleratorOptions.device for layout/table models and
                 RapidOCR's use_cuda flag).
    """
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import OcrAutoOptions, RapidOcrOptions
    from docling.datamodel.settings import settings
    from docling.document_converter import DocumentConverter, ImageFormatOption

    settings.debug.profile_pipeline_timings = True  # per-stage timings into conv_res.timings
    opts = ImageFormatOption().pipeline_options  # Docling's default for IMAGE input
    if ocr_backend == "off":
        opts.do_ocr = False  # layout/table only; no OCR text cells
    elif ocr_backend == "auto":
        opts.ocr_options = OcrAutoOptions()
    else:
        opts.ocr_options = RapidOcrOptions(backend=ocr_backend)
    opts.accelerator_options = AcceleratorOptions(device=device)
    return DocumentConverter(
        format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts)}
    )


def cache_dir_for(out_dir: Path, ocr_backend: str, device: str, include_furniture: bool = False) -> Path:
    """
    The legacy cache (auto OCR, CPU devices, BODY layer only) stays in <out_dir>/cache. Any other
    configuration gets its own folder so that results from different OCR engines, devices or
    content layers never mix. include_furniture=True adds the suffix '_furniture'.
    """
    if ocr_backend == "auto" and device == "cpu" and not include_furniture:
        return out_dir / "cache"
    suffix = "_furniture" if include_furniture else ""
    return out_dir / f"cache_ocr-{ocr_backend}_dev-{device}{suffix}"


def iter_doc_items(doc, include_furniture: bool):
    """
    Reading-order traversal of a DoclingDocument.
    include_furniture=False: docling_core's default, BODY layer only (FURNITURE items such as
    page_header / page_footer are skipped). This is what the legacy cache contains.
    include_furniture=True: every ContentLayer is included in the same traversal, so the
    FURNITURE items are yielded in the order that iterate_items() gives them.
    """
    if not include_furniture:
        return doc.iterate_items()
    from docling_core.types.doc import ContentLayer
    return doc.iterate_items(included_content_layers=set(ContentLayer))


def run_docling_page(converter, img_path: Path, track_gpu: bool = False,
                     include_furniture: bool = False) -> Dict[str, Any]:
    """
    Runs Docling on one page image and returns pixel-space boxes in reading (iterate) order.
    track_gpu: record torch.cuda peak allocated memory for this page (only meaningful when the
    resolved device is CUDA; avoids initialising a CUDA context in CPU runs).
    include_furniture: see iter_doc_items (default False = legacy BODY-only traversal).
    """
    torch_cuda = _torch_cuda_available()
    if track_gpu and torch_cuda:
        import torch
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    conv_res = converter.convert(str(img_path))
    doc = conv_res.document
    elapsed = time.time() - t0
    stage_times = {k: round(float(sum(v.times)), 4) for k, v in (conv_res.timings or {}).items()}
    gpu_peak_mb = None
    if track_gpu and torch_cuda:
        import torch
        gpu_peak_mb = round(torch.cuda.max_memory_allocated() / 2**20, 1)

    page = doc.pages[1] if 1 in doc.pages else next(iter(doc.pages.values()))
    p_w, p_h = float(page.size.width), float(page.size.height)
    with Image.open(img_path) as im:
        img_w, img_h = im.size
    sx, sy = img_w / p_w, img_h / p_h
    # Docling page size is derived from the image DPI tag; a mismatch in aspect ratio would
    # indicate an orientation change (EXIF / quarter-turn) and would make the scale invalid.
    aspect_mismatch = abs((img_w / img_h) - (p_w / p_h)) / (img_w / img_h) > 0.01

    items: List[Dict[str, Any]] = []
    for order, (it, _level) in enumerate(iter_doc_items(doc, include_furniture)):
        raw = getattr(it, "label", None)
        label = raw.value if hasattr(raw, "value") else str(raw if raw is not None else type(it).__name__)
        bbox = None
        prov = getattr(it, "prov", None)
        if prov and len(prov) > 0 and getattr(prov[0], "bbox", None) is not None:
            b = prov[0].bbox.to_top_left_origin(p_h)
            bbox = docling_bbox_to_pixel([b.l, b.t, b.r, b.b], sx, sy)
        layer = getattr(it, "content_layer", None)
        items.append({
            "order": order,
            "label": label,
            "bbox": bbox,
            "content_layer": layer.value if hasattr(layer, "value") else str(layer),
            "text": (getattr(it, "text", "") or "")[:80],
        })

    return {
        "image_size": [img_w, img_h],
        "docling_page_size_pt": [round(p_w, 3), round(p_h, 3)],
        "scale": [round(sx, 6), round(sy, 6)],
        "aspect_mismatch": aspect_mismatch,
        "docling_time_sec": round(elapsed, 3),
        "stage_times_sec": stage_times,
        "torch_cuda_available": torch_cuda,
        "gpu_peak_allocated_mb": gpu_peak_mb,
        "ocr_engine_evidence": _ocr_engine_evidence(converter),
        "include_furniture": include_furniture,
        "content_layers": "all" if include_furniture else "body",
        "items": items,
    }


def _torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _walk_engine_objects(obj, depth: int = 0, seen: Optional[set] = None, out: Optional[List] = None) -> List:
    """Collects torch modules (with their parameter device) and onnxruntime sessions (with providers)."""
    seen = seen if seen is not None else set()
    out = out if out is not None else []
    if id(obj) in seen or depth > 6:
        return out
    seen.add(id(obj))
    try:
        import torch
        if isinstance(obj, torch.nn.Module):
            try:
                out.append({"kind": "torch", "cls": type(obj).__name__, "device": str(next(obj.parameters()).device)})
            except StopIteration:
                pass
            return out
    except ImportError:
        pass
    try:
        import onnxruntime as ort
        if isinstance(obj, ort.InferenceSession):
            out.append({"kind": "onnx", "cls": type(obj).__name__, "providers": obj.get_providers()})
            return out
    except ImportError:
        pass
    if isinstance(obj, dict):
        children = list(obj.values())
    elif isinstance(obj, (list, tuple)):
        children = list(obj)
    elif hasattr(obj, "__dict__"):
        children = list(vars(obj).values())
    else:
        children = []
    for c in children:
        _walk_engine_objects(c, depth + 1, seen, out)
    return out


def _ocr_engine_evidence(converter) -> Dict[str, Any]:
    """Which OCR engine objects exist and where they run (torch parameter device / onnx providers)."""
    evidence: Dict[str, Any] = {"ocr_model_class": None, "engine_objects": []}
    for pl in getattr(converter, "initialized_pipelines", {}).values():
        ocr = getattr(pl, "ocr_model", None)
        if ocr is None:
            continue
        evidence["ocr_model_class"] = type(ocr).__name__
        evidence["engine_objects"] = _walk_engine_objects(ocr)
    return evidence


# ----------------------------------------------------------------------
# Ordering + metrics
# ----------------------------------------------------------------------
def score_method(
    gt_items: List[Dict[str, Any]],
    gt_to_pred: Dict[int, int],
    pos_pred: Dict[int, int],
) -> Dict[str, Any]:
    """
    gt_to_pred: GT index -> predicted index (matched pairs only).
    pos_pred:   predicted index -> rank in this method's predicted reading order.
    """
    n_gt = len(gt_items)
    n_pred_seq = len(pos_pred)
    matched_g = sorted(gt_to_pred.keys())

    gt_order = [g["order"] for g in gt_items]
    tau_matched = compute_kendall_tau(
        [gt_order[g] for g in matched_g],
        [pos_pred[gt_to_pred[g]] for g in matched_g],
    )

    keys = [pos_pred[gt_to_pred[g]] if g in gt_to_pred else n_pred_seq for g in range(n_gt)]
    tau_penalized = compute_kendall_tau(gt_order, keys) if n_gt >= 2 else None

    return {
        "n_matched": len(matched_g),
        "coverage": (len(matched_g) / n_gt) if n_gt else None,
        "tau_matched": tau_matched,
        "tau_penalized": tau_penalized,
    }


def evaluate_page_ok(
    dataset_index: int,
    item: Dict[str, Any],
    cache: Dict[str, Any],
    gt_items: List[Dict[str, Any]],
    iou_thr: float,
) -> Dict[str, Any]:
    p = item["page_info"]
    attr = p.get("page_attribute", {})
    pw = float(p.get("width", 1000))
    ph = float(p.get("height", 1000))

    pred = [
        {"id": f"p_{k}", "type": it["label"], "bbox": it["bbox"], "order": it["order"]}
        for k, it in enumerate(cache["items"])
        if it.get("bbox") and len(it["bbox"]) == 4
    ]
    n_pred = len(pred)
    n_gt = len(gt_items)
    n_furniture_pred = sum(
        1 for it in cache["items"]
        if it.get("content_layer") == "furniture" and it.get("bbox") and len(it["bbox"]) == 4
    )

    gt_to_pred, _ = match_one_to_one(
        [p_["bbox"] for p_ in pred],
        [g["bbox"] for g in gt_items],
        iou_thr,
    )

    # Method orderings as lists of predicted indices in reading order.
    order_a = list(range(n_pred))  # Docling iterate order
    sorted_b = sort_elements_reading_order(pred, pw, ph)
    id_to_idx = {e["id"]: k for k, e in enumerate(pred)}
    order_b = [id_to_idx[e["id"]] for e in sorted_b]
    order_c = sorted(range(n_pred), key=lambda k: (pred[k]["bbox"][1], pred[k]["bbox"][0]))

    methods: Dict[str, Dict[str, Any]] = {}
    for m, order in (("A", order_a), ("B", order_b), ("C", order_c)):
        pos = {k: r for r, k in enumerate(order)}
        methods[m] = score_method(gt_items, gt_to_pred, pos)

    # D: oracle on GT boxes. Every GT is "matched" to itself.
    sorted_d = sort_elements_reading_order(gt_items, pw, ph)
    gt_id_to_idx = {g["id"]: k for k, g in enumerate(gt_items)}
    pos_d = {gt_id_to_idx[e["id"]]: r for r, e in enumerate(sorted_d)}
    methods["D"] = score_method(gt_items, {g: g for g in range(n_gt)}, pos_d)

    issues = [s for s in attr.get("special_issue", []) if s != "None"]
    # Matching coverage per GT category (same for all methods): explains where misses come from.
    gt_by_type: Dict[str, List[int]] = defaultdict(lambda: [0, 0])  # type -> [n_gt, n_matched]
    for gi, g in enumerate(gt_items):
        gt_by_type[g["type"]][0] += 1
        if gi in gt_to_pred:
            gt_by_type[g["type"]][1] += 1
    return {
        "dataset_index": dataset_index,
        "image_path": p["image_path"],
        "layout": attr.get("layout", "unknown"),
        "data_source": attr.get("data_source", "unknown"),
        "special_issue": issues,
        "status": "ok",
        "n_gt": n_gt,
        "n_pred": n_pred,
        "n_furniture_pred": n_furniture_pred,
        "docling_time_sec": cache.get("docling_time_sec"),
        "stage_times_sec": cache.get("stage_times_sec", {}),
        "gpu_peak_allocated_mb": cache.get("gpu_peak_allocated_mb"),
        "ocr_backend": cache.get("ocr_backend"),
        "device_resolved": cache.get("device_resolved"),
        "aspect_mismatch": bool(cache.get("aspect_mismatch", False)),
        "gt_by_type": dict(gt_by_type),
        "methods": methods,
    }


def coverage_by_gt_type(ok_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    agg: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for r in ok_records:
        for t, (n, m) in r.get("gt_by_type", {}).items():
            agg[t][0] += n
            agg[t][1] += m
    rows = {t: {"n_gt": n, "n_matched": m, "coverage": round(m / n, 4) if n else None}
            for t, (n, m) in agg.items()}
    return dict(sorted(rows.items(), key=lambda kv: -kv[1]["n_gt"]))


# ----------------------------------------------------------------------
# Page selection
# ----------------------------------------------------------------------
def stratified_sample(indexed: List[Tuple[int, Dict[str, Any]]], n: int, seed: int) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Proportional stratified sample over layout x data_source.
    Each non-empty stratum gets at least 1 page; the total is adjusted to exactly n
    (largest-remainder rule). Deterministic for a fixed seed.
    """
    if n >= len(indexed):
        return list(indexed)
    strata: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    by_index = dict(indexed)
    for i, it in indexed:
        a = it["page_info"].get("page_attribute", {})
        strata[(a.get("layout", "unknown"), a.get("data_source", "unknown"))].append(i)
    keys = sorted(strata)
    if n < len(keys):
        raise ValueError(f"--stratified {n} is smaller than the number of strata ({len(keys)})")

    total = len(indexed)
    quota = {k: n * len(strata[k]) / total for k in keys}
    alloc = {k: min(len(strata[k]), max(1, math.floor(quota[k]))) for k in keys}
    diff = n - sum(alloc.values())
    while diff > 0:
        cands = [k for k in keys if alloc[k] < len(strata[k])]
        k = max(cands, key=lambda k: (quota[k] - alloc[k], k))
        alloc[k] += 1
        diff -= 1
    while diff < 0:
        cands = [k for k in keys if alloc[k] > 1]
        k = max(cands, key=lambda k: (alloc[k] - quota[k], k))
        alloc[k] -= 1
        diff += 1

    rng = random.Random(seed)
    chosen: List[int] = []
    for k in keys:
        chosen.extend(rng.sample(sorted(strata[k]), alloc[k]))
    return [(i, by_index[i]) for i in sorted(chosen)]


def select_pages(data: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[List[Tuple[int, Dict[str, Any]]], str]:
    indexed = [(i, it) for i, it in enumerate(data) if (IMAGES_DIR / it["page_info"]["image_path"]).exists()]
    if args.pages_file is not None:
        wanted = {ln.strip() for ln in args.pages_file.read_text(encoding="utf-8").splitlines() if ln.strip()}
        indexed = [(i, it) for i, it in indexed if it["page_info"]["image_path"] in wanted]
        if len(indexed) != len(wanted):
            raise ValueError(f"--pages-file lists {len(wanted)} pages but only {len(indexed)} exist in the dataset")
        sel, mode = indexed, f"pages_file_{args.pages_file.name}"
    elif args.all:
        sel, mode = indexed, "all"
    elif args.stratified is not None:
        sel, mode = stratified_sample(indexed, args.stratified, args.seed), f"stratified_{args.stratified}_seed{args.seed}"
    else:
        sel, mode = indexed, "first_n"
    if args.limit:
        sel = sel[: args.limit]
        mode += f"_limit{args.limit}"
    return sorted(sel, key=lambda x: x[0]), mode


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------
def summarize_values(vals: Sequence[Optional[float]]) -> Dict[str, Any]:
    valid = [v for v in vals if v is not None]
    out: Dict[str, Any] = {
        "n_pages": len(vals),
        "n_defined": len(valid),
        "n_undefined": len(vals) - len(valid),
        "mean": None, "median": None, "min": None, "max": None,
        "pct_ge_0_7": None, "pct_lt_0_5": None,
    }
    if valid:
        out.update({
            "mean": round(sum(valid) / len(valid), 4),
            "median": round(statistics.median(valid), 4),
            "min": round(min(valid), 4),
            "max": round(max(valid), 4),
            "pct_ge_0_7": round(100.0 * sum(v >= 0.7 for v in valid) / len(valid), 1),
            "pct_lt_0_5": round(100.0 * sum(v < 0.5 for v in valid) / len(valid), 1),
        })
    return out


def summarize_methods(ok_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for m in METHOD_NAMES:
        entry: Dict[str, Any] = {"name": METHOD_NAMES[m]}
        for metric in METRICS:
            entry[metric] = summarize_values([r["methods"][m][metric] for r in ok_records])
        cov = [r["methods"][m]["coverage"] for r in ok_records if r["methods"][m]["coverage"] is not None]
        entry["coverage"] = {
            "n_pages": len(cov),
            "mean": round(sum(cov) / len(cov), 4) if cov else None,
            "median": round(statistics.median(cov), 4) if cov else None,
        }
        out[m] = entry
    return out


def group_breakdown(groups: Dict[str, List[Dict[str, Any]]], min_pages: int = 1) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for g, recs in sorted(groups.items()):
        if len(recs) < min_pages:
            continue
        row: Dict[str, Any] = {"n_pages": len(recs), "n_pen_defined": sum(r["methods"]["B"]["tau_penalized"] is not None for r in recs), "methods": {}}
        for m in METHOD_NAMES:
            pen = [r["methods"][m]["tau_penalized"] for r in recs if r["methods"][m]["tau_penalized"] is not None]
            mat = [r["methods"][m]["tau_matched"] for r in recs if r["methods"][m]["tau_matched"] is not None]
            cov = [r["methods"][m]["coverage"] for r in recs if r["methods"][m]["coverage"] is not None]
            row["methods"][m] = {
                "tau_penalized_mean": round(sum(pen) / len(pen), 4) if pen else None,
                "tau_matched_mean": round(sum(mat) / len(mat), 4) if mat else None,
                "coverage_mean": round(sum(cov) / len(cov), 4) if cov else None,
            }
        out[g] = row
    return out


def pairwise(ok_records: List[Dict[str, Any]], m1: str, m2: str, metric: str, seed: int) -> Dict[str, Any]:
    pairs = [(r["methods"][m1][metric], r["methods"][m2][metric]) for r in ok_records]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    deltas = np.array([a - b for a, b in pairs], dtype=float)
    out: Dict[str, Any] = {"n_compared": len(deltas), "wins": 0, "ties": 0, "losses": 0,
                           "mean_delta": None, "median_delta": None, "bootstrap_ci95_mean_delta": None}
    if len(deltas) == 0:
        return out
    eps = 1e-9
    out["wins"] = int(np.sum(deltas > eps))
    out["ties"] = int(np.sum(np.abs(deltas) <= eps))
    out["losses"] = int(np.sum(deltas < -eps))
    out["mean_delta"] = round(float(deltas.mean()), 4)
    out["median_delta"] = round(float(np.median(deltas)), 4)
    if len(deltas) >= 2:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(deltas), size=(BOOTSTRAP_RESAMPLES, len(deltas)))
        boot = deltas[idx].mean(axis=1)
        out["bootstrap_ci95_mean_delta"] = [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)]
    return out


def oracle_full_dataset_check(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Method D (GT boxes only, no Docling) over the whole dataset: must reproduce the old Tier 1 number."""
    vals: List[float] = []
    skipped = 0
    for item in data:
        gt = gt_elements_from_page(item)
        if len(gt) < 2:
            skipped += 1
            continue
        p = item["page_info"]
        sorted_gt = sort_elements_reading_order(gt, float(p.get("width", 1000)), float(p.get("height", 1000)))
        rank = {e["id"]: r for r, e in enumerate(sorted_gt)}
        tau = compute_kendall_tau([g["order"] for g in gt], [rank[g["id"]] for g in gt])
        if tau is not None:
            vals.append(tau)
    summ = summarize_values(vals)
    summ.update({"n_pages_total": len(data), "pages_skipped_lt2": skipped})
    return summ


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
def fmt_tau(v: Optional[float]) -> str:
    return "None" if v is None else f"{v:+.3f}"


def run(args: argparse.Namespace) -> None:
    out_dir: Path = args.out_dir
    cache_dir = cache_dir_for(out_dir, args.ocr_backend, args.device, args.include_furniture)
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_furniture" if args.include_furniture else ""
    log = Logger(out_dir / f"run{suffix}.log")
    wall_t0 = time.time()

    if not (args.all or args.stratified is not None or args.limit or args.pages_file):
        args.stratified = DEFAULT_STRATIFIED_N
    log("=" * 80)
    log(f"E2E reading-order eval start | argv={' '.join(sys.argv[1:])}")

    with open(DATASET_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    selection, mode = select_pages(data, args)
    n_sel = len(selection)
    log(f"Dataset pages: {len(data)} | selected: {n_sel} | mode: {mode} | iou_thr={args.iou_thr}")

    # Count how many selected pages still need a Docling run
    uncached = sum(1 for _, it in selection if _needs_docling(cache_dir, it))
    log(f"Pages already cached (status ok): {n_sel - uncached} | to run with Docling: {uncached}")

    converter = None
    records: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    fresh_times: List[float] = []
    eval_t0 = time.time()

    for pos, (idx, item) in enumerate(selection, 1):
        p = item["page_info"]
        attr = p.get("page_attribute", {})
        img_rel = p["image_path"]
        stem = Path(img_rel).stem
        cache_file = cache_dir / f"{stem}.json"
        cache = load_cache(cache_file)

        source = "cache"
        if cache is None or cache.get("status") != "ok":
            source = "docling"
            if converter is None:
                log(f"Initialising DocumentConverter: ocr_backend={args.ocr_backend} device={args.device} "
                    f"cache_dir={cache_dir.name} torch_cuda={_torch_cuda_available()}")
                converter = build_converter(args.ocr_backend, args.device)
                from docling.utils.accelerator_utils import decide_device
                resolved_device = decide_device(args.device)  # e.g. 'cuda:0' or 'cpu'
            try:
                cache = run_docling_page(converter, IMAGES_DIR / img_rel,
                                         track_gpu=resolved_device.startswith("cuda"),
                                         include_furniture=args.include_furniture)
                cache.update({"status": "ok", "image_path": img_rel, "dataset_index": idx,
                              "ocr_backend": args.ocr_backend, "device_requested": args.device,
                              "device_resolved": resolved_device})
                fresh_times.append(cache["docling_time_sec"])
            except Exception as e:  # record and continue
                cache = {"status": "error", "image_path": img_rel, "dataset_index": idx,
                         "error": f"{type(e).__name__}: {str(e)[:300]}"}
                source = "docling_error"
            save_json_atomic(cache_file, cache)

        gt_items = gt_elements_from_page(item)
        if cache.get("status") == "ok":
            rec = evaluate_page_ok(idx, item, cache, gt_items, args.iou_thr)
            records.append(rec)
            m = rec["methods"]
            log(f"[{pos}/{n_sel}] {stem[:28]:28s} {source:12s} dock={cache.get('docling_time_sec', 0):7.2f}s "
                f"{attr.get('layout', '?'):16s} {attr.get('data_source', '?'):18s} gt={rec['n_gt']:3d} pred={rec['n_pred']:3d} "
                f"pen A={fmt_tau(m['A']['tau_penalized'])} B={fmt_tau(m['B']['tau_penalized'])} "
                f"C={fmt_tau(m['C']['tau_penalized'])} D={fmt_tau(m['D']['tau_penalized'])}")
        else:
            rec = {"dataset_index": idx, "image_path": img_rel, "status": "error",
                   "layout": attr.get("layout", "unknown"), "data_source": attr.get("data_source", "unknown"),
                   "error": cache.get("error", "unknown")}
            errors.append({"image_path": img_rel, "error": rec["error"]})
            log(f"[{pos}/{n_sel}] {stem[:28]:28s} DOCLING ERROR: {rec['error']}")

        if source == "docling" and fresh_times and pos % 10 == 0:
            remaining_fresh = sum(1 for _, it2 in selection[pos:] if _needs_docling(cache_dir, it2))
            avg = sum(fresh_times) / len(fresh_times)
            log(f"  progress {pos}/{n_sel} | elapsed {(time.time() - wall_t0) / 60:.1f} min | "
                f"avg docling {avg:.2f}s/page | ETA for {remaining_fresh} uncached pages ~{avg * remaining_fresh / 60:.1f} min")

    ok_records = [r for r in records if r["status"] == "ok"]
    eval_elapsed = time.time() - eval_t0

    # Timing: use the docling_time_sec stored in every ok cache (true cost of each page)
    cached_times = [r["docling_time_sec"] for r in ok_records if r.get("docling_time_sec") is not None]
    timing = {
        "docling_pages_timed": len(cached_times),
        "docling_mean_sec_per_page": round(sum(cached_times) / len(cached_times), 3) if cached_times else None,
        "docling_median_sec_per_page": round(statistics.median(cached_times), 3) if cached_times else None,
        "docling_total_sec_selected": round(sum(cached_times), 1),
        "fresh_docling_runs_this_invocation": len(fresh_times),
        "fresh_docling_total_sec_this_invocation": round(sum(fresh_times), 1),
        "eval_seconds_this_invocation": round(eval_elapsed, 2),
        "wall_clock_seconds_this_invocation": round(time.time() - wall_t0, 1),
    }
    stage_keys = sorted({k for r in ok_records for k in r.get("stage_times_sec", {})})
    timing["stage_mean_sec_per_page"] = {
        k: round(sum(r.get("stage_times_sec", {}).get(k, 0.0) for r in ok_records) / len(ok_records), 3)
        for k in stage_keys
    } if ok_records else {}
    if cached_times:
        timing["estimated_full_dataset_hours_at_mean"] = round(
            timing["docling_mean_sec_per_page"] * len(data) / 3600.0, 2)

    # Breakdowns
    by_layout: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_issue: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in ok_records:
        by_layout[r["layout"]].append(r)
        by_source[r["data_source"]].append(r)
        if r["special_issue"]:
            for s in r["special_issue"]:
                by_issue[s].append(r)
        else:
            by_issue["standard_page"].append(r)

    results: Dict[str, Any] = {
        "config": {
            "mode": mode,
            "ocr_backend": args.ocr_backend,
            "device": args.device,
            "cache_dir": cache_dir.name,
            "include_furniture": args.include_furniture,
            "content_layers": "all (BODY + FURNITURE + BACKGROUND + INVISIBLE + NOTES)" if args.include_furniture
                              else "body (docling_core default)",
            "device_resolved_observed": sorted({r["device_resolved"] for r in ok_records if r.get("device_resolved")}),
            "dataset_json": str(DATASET_JSON),
            "dataset_pages_total": len(data),
            "selected_pages": n_sel,
            "seed": args.seed,
            "iou_thr": args.iou_thr,
            "matching": "one-to-one Hungarian on IoU (pairs with IoU >= iou_thr)",
            "docling": (f"build_converter(ocr_backend={args.ocr_backend}, device={args.device}): "
                        "ImageFormatOption() defaults except ocr_options and accelerator_options"),
            "methods": METHOD_NAMES,
            "gt_filter": "same as Tier 1: ignore=False, order not None, excluded categories removed",
            "unmatched_gt_convention": "penalized Tau: unmatched GT get predicted key N (after all predictions), tied with each other",
            "special_issue_min_pages": MIN_GROUP_PAGES_SPECIAL,
        },
        "counts": {
            "selected": n_sel,
            "ok": len(ok_records),
            "docling_errors": len(errors),
            "pages_with_lt2_gt": sum(1 for r in ok_records if r["n_gt"] < 2),
            "pages_tau_matched_undefined": sum(1 for r in ok_records if r["methods"]["B"]["tau_matched"] is None),
            "pages_with_aspect_mismatch": sum(1 for r in ok_records if r["aspect_mismatch"]),
            "pages_with_no_gt": sum(1 for r in ok_records if r["n_gt"] == 0),
        },
        "timing": timing,
        "errors": errors,
        "summary": summarize_methods(ok_records),
        "coverage_by_gt_type": coverage_by_gt_type(ok_records),
        "breakdown": {
            "layout": group_breakdown(by_layout),
            "data_source": group_breakdown(by_source),
            "special_issue": group_breakdown(by_issue, min_pages=MIN_GROUP_PAGES_SPECIAL),
        },
        "pairwise": {
            "B_vs_A": {metric: pairwise(ok_records, "B", "A", metric, args.seed) for metric in METRICS},
            "B_vs_C": {metric: pairwise(ok_records, "B", "C", metric, args.seed) for metric in METRICS},
        },
        "full_dataset_oracle_check": oracle_full_dataset_check(data),
        "per_page": records,
    }

    if args.include_furniture:
        base_path = out_dir / "results.json"
        if base_path.exists():
            baseline = json.loads(base_path.read_text(encoding="utf-8"))
            results["comparison_vs_baseline"] = build_comparison(results, baseline)
            log(f"Compared with baseline {base_path.name} (same pages: "
                f"{results['comparison_vs_baseline']['same_page_set']}, same config: "
                f"{results['comparison_vs_baseline']['baseline_config_match']})")
        else:
            log(f"No baseline {base_path.name} found; comparison section skipped")

    results_path = out_dir / f"results{suffix}.json"
    save_json_atomic(results_path, results)
    log(f"Saved {results_path}")

    report_path = out_dir / f"report{suffix}.md"
    write_report(results, report_path)
    log(f"Saved {report_path}")
    log(f"DONE: ok={len(ok_records)} errors={len(errors)} wall={timing['wall_clock_seconds_this_invocation']}s")


# ----------------------------------------------------------------------
# Report (Vietnamese)
# ----------------------------------------------------------------------
OCR_DESC = {
    "auto": "RapidOCR qua onnxruntime (Docling tự chọn, như cache cũ)",
    "onnxruntime": "RapidOCR, backend onnxruntime",
    "torch": "RapidOCR, backend torch (PP-OCRv6)",
    "off": "OCR tắt (do_ocr=False)",
}


def describe_docling(res: Dict[str, Any]) -> str:
    """OCR backend, resolved device and content-layer setting, read from the run's own config."""
    cfg = res["config"]
    devices = cfg.get("device_resolved_observed") or sorted(
        {r["device_resolved"] for r in res["per_page"] if r.get("device_resolved")})
    dev_s = ", ".join(devices) if devices else f"chưa xác định (yêu cầu `{cfg['device']}`)"
    layers = "tất cả lớp (gồm FURNITURE)" if cfg.get("include_furniture") else "chỉ lớp BODY (mặc định docling_core)"
    return (f"OCR: {OCR_DESC.get(cfg['ocr_backend'], cfg['ocr_backend'])}; thiết bị thực tế: {dev_s} "
            f"(yêu cầu `{cfg['device']}`); duyệt item: {layers}")


def build_comparison(res: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    """Before/after numbers against a baseline results.json (expected: same pages, BODY layer only)."""
    new_paths = {r["image_path"] for r in res["per_page"]}
    old_paths = {r["image_path"] for r in base["per_page"]}
    cfg_new, cfg_old = res["config"], base["config"]
    cmp: Dict[str, Any] = {
        "baseline_file": "results.json",
        "baseline_layers": cfg_old.get("content_layers", "body (docling_core default)"),
        "baseline_ocr_device": [cfg_old.get("ocr_backend"), cfg_old.get("device")],
        "new_ocr_device": [cfg_new.get("ocr_backend"), cfg_new.get("device")],
        "baseline_config_match": (cfg_old.get("ocr_backend"), cfg_old.get("device"))
                                 == (cfg_new.get("ocr_backend"), cfg_new.get("device")),
        "same_page_set": new_paths == old_paths,
        "n_pages_new": len(new_paths),
        "n_pages_baseline": len(old_paths),
        "methods": {}, "coverage_by_gt_type": {}, "pairwise_B_vs_A": {}, "layout_B_minus_A": {},
    }
    for m in METHOD_NAMES:
        ob, nw = base["summary"][m], res["summary"][m]
        cmp["methods"][m] = {
            "tau_penalized_mean": {"baseline": ob["tau_penalized"]["mean"], "new": nw["tau_penalized"]["mean"]},
            "tau_matched_mean": {"baseline": ob["tau_matched"]["mean"], "new": nw["tau_matched"]["mean"]},
            "coverage_mean": {"baseline": ob["coverage"]["mean"], "new": nw["coverage"]["mean"]},
        }
    empty = {"n_gt": 0, "n_matched": 0, "coverage": None}
    for t in sorted(set(base["coverage_by_gt_type"]) | set(res["coverage_by_gt_type"])):
        ob = base["coverage_by_gt_type"].get(t, empty)
        nw = res["coverage_by_gt_type"].get(t, empty)
        cmp["coverage_by_gt_type"][t] = {
            "n_gt_baseline": ob["n_gt"], "n_gt_new": nw["n_gt"],
            "matched_baseline": ob["n_matched"], "matched_new": nw["n_matched"],
            "coverage_baseline": ob["coverage"], "coverage_new": nw["coverage"],
        }
    for metric in METRICS:
        cmp["pairwise_B_vs_A"][metric] = {"baseline": base["pairwise"]["B_vs_A"][metric],
                                          "new": res["pairwise"]["B_vs_A"][metric]}

    def b_minus_a(row: Optional[Dict[str, Any]]) -> Optional[float]:
        if not row:
            return None
        a, b = row["methods"]["A"]["tau_penalized_mean"], row["methods"]["B"]["tau_penalized_mean"]
        return None if a is None or b is None else round(b - a, 4)

    lay_b, lay_n = base["breakdown"]["layout"], res["breakdown"]["layout"]
    for g in sorted(set(lay_b) | set(lay_n)):
        cmp["layout_B_minus_A"][g] = {
            "n_pages_baseline": lay_b.get(g, {}).get("n_pages"),
            "n_pages_new": lay_n.get(g, {}).get("n_pages"),
            "delta_baseline": b_minus_a(lay_b.get(g)),
            "delta_new": b_minus_a(lay_n.get(g)),
        }
    return cmp


def _bn(b: Optional[float], n: Optional[float]) -> str:
    return f"{f4(b)} → {f4(n)}"


def _dn(b: Optional[float], n: Optional[float]) -> str:
    return "—" if b is None or n is None else f"{n - b:+.4f}"


def _fd(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:+.4f}"


def _ci(p: Dict[str, Any]) -> str:
    if p["mean_delta"] is None:
        return "—"
    ci = p["bootstrap_ci95_mean_delta"]
    ci_s = "—" if ci is None else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
    return f"{p['mean_delta']:+.4f} {ci_s}"


def comparison_markdown(cmp: Dict[str, Any]) -> List[str]:
    L: List[str] = ["## 7. So sánh với lần chạy trước (chỉ lớp BODY)\n"]
    L.append(f"- Nguồn so sánh: `results.json` ({cmp['baseline_layers']}; OCR/device: "
             f"{cmp['baseline_ocr_device'][0]} / {cmp['baseline_ocr_device'][1]}). "
             f"Lần này: OCR/device {cmp['new_ocr_device'][0]} / {cmp['new_ocr_device'][1]}.")
    L.append(f"- Cùng tập trang: **{'có' if cmp['same_page_set'] else 'KHÔNG'}** "
             f"({cmp['n_pages_new']} trang mới, {cmp['n_pages_baseline']} trang cũ). "
             f"Cùng cấu hình OCR/device: **{'có' if cmp['baseline_config_match'] else 'KHÔNG'}**.\n")
    L.append("### 7.1 Bốn phương pháp, trước → sau\n")
    L.append("| Mã | Tau phạt TB | Δ | Tau khớp TB | Δ | Coverage TB | Δ |")
    L.append("|---|---|---|---|---|---|---|")
    for m in METHOD_NAMES:
        r = cmp["methods"][m]
        tp, tk, cv = r["tau_penalized_mean"], r["tau_matched_mean"], r["coverage_mean"]
        L.append(f"| {m} `{METHOD_NAMES[m]}` | {_bn(tp['baseline'], tp['new'])} | {_dn(tp['baseline'], tp['new'])} | "
                 f"{_bn(tk['baseline'], tk['new'])} | {_dn(tk['baseline'], tk['new'])} | "
                 f"{_bn(cv['baseline'], cv['new'])} | {_dn(cv['baseline'], cv['new'])} |")
    L.append("")
    L.append("### 7.2 Coverage theo loại GT, trước → sau\n")
    L.append("| Loại GT | Số GT (trước → sau) | Số khớp (trước → sau) | Coverage (trước → sau) | Δ coverage (điểm %) |")
    L.append("|---|---|---|---|---|")
    rows = sorted(cmp["coverage_by_gt_type"].items(),
                  key=lambda kv: -max(kv[1]["n_gt_new"], kv[1]["n_gt_baseline"]))
    for t, r in rows:
        cb, cn = r["coverage_baseline"], r["coverage_new"]
        d = "—" if cb is None or cn is None else f"{100 * (cn - cb):+.1f}"
        L.append(f"| `{t}` | {r['n_gt_baseline']} → {r['n_gt_new']} | {r['matched_baseline']} → {r['matched_new']} | "
                 f"{fpct(None if cb is None else 100 * cb)} → {fpct(None if cn is None else 100 * cn)} | {d} |")
    L.append("")
    L.append("### 7.3 B − A (documind_xycut − docling_native), trước → sau\n")
    L.append("| Chỉ số | Trước: số trang | Trước: B thắng / hoà / thua | Trước: Mean Δ [CI 95%] | "
             "Sau: số trang | Sau: B thắng / hoà / thua | Sau: Mean Δ [CI 95%] |")
    L.append("|---|---|---|---|---|---|---|")
    for metric in METRICS:
        b = cmp["pairwise_B_vs_A"][metric]["baseline"]
        n = cmp["pairwise_B_vs_A"][metric]["new"]
        L.append(f"| {metric} | {b['n_compared']} | {b['wins']} / {b['ties']} / {b['losses']} | {_ci(b)} | "
                 f"{n['n_compared']} | {n['wins']} / {n['ties']} / {n['losses']} | {_ci(n)} |")
    L.append("")
    L.append("### 7.4 B − A theo bố cục (tau_penalized TB), trước → sau\n")
    L.append("| Bố cục | Số trang (trước → sau) | Δ trước | Δ sau |")
    L.append("|---|---|---|---|")
    for g, r in cmp["layout_B_minus_A"].items():
        L.append(f"| `{g}` | {r['n_pages_baseline']} → {r['n_pages_new']} | {_fd(r['delta_baseline'])} | {_fd(r['delta_new'])} |")
    L.append("")
    return L


def f4(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:.4f}"


def fpct(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:.1f}%"


def write_report(res: Dict[str, Any], path: Path) -> None:
    cfg, cnt, tm, summ = res["config"], res["counts"], res["timing"], res["summary"]
    pw_b_a = res["pairwise"]["B_vs_A"]
    pw_b_c = res["pairwise"]["B_vs_C"]
    oracle = res["full_dataset_oracle_check"]
    L: List[str] = []

    L.append("# Đánh giá đầu-cuối thứ tự đọc: Docling và DocuMind XY-Cut trên OmniDocBench\n")
    L.append(f"> **Chế độ chọn trang:** `{cfg['mode']}` | **Số trang chọn:** {cfg['selected_pages']} / {cfg['dataset_pages_total']} | "
             f"**Seed:** {cfg['seed']} | **IoU ngưỡng khớp:** {cfg['iou_thr']}")
    L.append(f"> **Docling:** {describe_docling(res)}")
    L.append(f"> **Docling trung bình/trang** (từ cache của mọi trang đã chọn): {tm['docling_mean_sec_per_page']} s "
             f"(trung vị {tm['docling_median_sec_per_page']} s) | "
             f"**Ước tính chạy toàn bộ 1.651 trang:** ~{tm.get('estimated_full_dataset_hours_at_mean', '—')} giờ | "
             f"**Lần chạy này:** {tm['wall_clock_seconds_this_invocation']} s, gọi Docling {tm['fresh_docling_runs_this_invocation']} lần\n")
    L.append("---\n")

    # 1. Summary
    L.append("## 1. Tóm tắt\n")
    L.append(f"- Trang đánh giá thành công: **{cnt['ok']}** / {cnt['selected']} (lỗi Docling: **{cnt['docling_errors']}**).")
    L.append(f"- Trang có < 2 phần tử GT (Tau không xác định, không tính 1.0): **{cnt['pages_with_lt2_gt']}**. "
             f"Trang có < 2 box khớp (Tau khớp không xác định): **{cnt['pages_tau_matched_undefined']}** (theo B).")
    L.append(f"- Trang có cảnh báo lệch tỉ lệ khung (aspect mismatch): **{cnt['pages_with_aspect_mismatch']}**.\n")

    L.append("### 1.1 Bảng toàn cục (4 phương pháp, cùng trang, cùng matching)\n")
    L.append("| Mã | Phương pháp | Tau khớp (mean / median) | Tau phạt (mean / median) | Tau phạt: % ≥ 0.70 | Tau phạt: % < 0.50 | Coverage (mean) |")
    L.append("|---|---|---|---|---|---|---|")
    for m in METHOD_NAMES:
        s = summ[m]
        tm_ = s["tau_matched"]
        tp = s["tau_penalized"]
        L.append(f"| {m} | `{s['name']}` | {f4(tm_['mean'])} / {f4(tm_['median'])} | **{f4(tp['mean'])}** / {f4(tp['median'])} | "
                 f"{fpct(tp['pct_ge_0_7'])} | {fpct(tp['pct_lt_0_5'])} | {f4(s['coverage']['mean'])} |")
    L.append("")
    L.append("Ghi chú: `docling_native` và `documind_xycut` / `naive_yx` cùng dùng tập box Docling. "
             "`oracle_xycut_gt` dùng box GT nên coverage = 1 theo định nghĩa, chỉ là mốc tham chiếu.\n")

    L.append("### 1.2 So sánh cặp: B (documind_xycut) so với A (docling_native)\n")
    L.append("| Chỉ số | Số trang so sánh | B thắng | Hoà | B thua | Mean Δ (B − A) | Median Δ | Khoảng tin cậy 95% của Mean Δ (bootstrap) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for metric in METRICS:
        p = pw_b_a[metric]
        ci = p["bootstrap_ci95_mean_delta"]
        ci_s = "—" if ci is None else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
        mean_s = "—" if p["mean_delta"] is None else f"{p['mean_delta']:+.4f}"
        median_s = "—" if p["median_delta"] is None else f"{p['median_delta']:+.4f}"
        L.append(f"| {metric} | {p['n_compared']} | {p['wins']} | {p['ties']} | {p['losses']} | "
                 f"{mean_s} | {median_s} | {ci_s} |")
    L.append("")
    L.append("Tham chiếu B so với C (`naive_yx`):\n")
    L.append("| Chỉ số | Số trang | B thắng | Hoà | B thua | Mean Δ (B − C) |")
    L.append("|---|---|---|---|---|---|")
    for metric in METRICS:
        p = pw_b_c[metric]
        md = "—" if p["mean_delta"] is None else f"{p['mean_delta']:+.4f}"
        L.append(f"| {metric} | {p['n_compared']} | {p['wins']} | {p['ties']} | {p['losses']} | {md} |")
    L.append("")

    L.append("### 1.3 Kiểm tra chéo với Tầng 1 cũ\n")
    L.append(f"- Phương pháp D (`oracle_xycut_gt`) chạy trên **toàn bộ {oracle['n_pages_total']} trang** (không cần Docling): "
             f"mean = **{f4(oracle['mean'])}**, median = {f4(oracle['median'])}, trang được tính = {oracle['n_defined']}, "
             f"trang bị loại (< 2 GT) = {oracle['pages_skipped_lt2']}.")
    matches = oracle["n_defined"] == 1602 and oracle["mean"] is not None and abs(oracle["mean"] - 0.9369) < 5e-5
    verdict = ("KHỚP: bộ lọc GT và hàm sắp xếp giống Tầng 1."
               if matches else
               "KHÔNG KHỚP: cần kiểm tra lại bộ lọc GT hoặc hàm sắp xếp so với Tầng 1.")
    L.append(f"- Tầng 1 cũ báo 0.9369 trên 1.602 trang. Kết quả trên: **{verdict}**\n")

    L.append("### 1.4 Coverage theo loại GT (cùng matching cho mọi phương pháp)\n")
    L.append("Dùng để giải thích phần tau_penalized: GT không khớp bị xếp cuối nên một lần bỏ sót GT ở đầu trang bị phạt mạnh.\n")
    L.append("| Loại GT (category_type) | Số GT | Số GT khớp | Coverage |")
    L.append("|---|---|---|---|")
    for t, row in list(res["coverage_by_gt_type"].items())[:15]:
        L.append(f"| `{t}` | {row['n_gt']} | {row['n_matched']} | {fpct(None if row['coverage'] is None else 100 * row['coverage'])} |")
    L.append("")

    # 2. Definitions
    L.append("## 2. Quy ước đo\n")
    L.append("- **Matching:** một-một, Hungarian trên IoU (tối đa tổng IoU), chỉ nhận cặp có IoU ≥ ngưỡng. Không ràng buộc nhãn lớp.")
    L.append("- **tau_matched:** Kendall's Tau-a chỉ trên các cặp GT–dự đoán đã khớp (cách tính của Tầng 2 cũ, nhưng khớp một-một).")
    L.append("- **coverage:** số GT khớp / tổng số GT (sau bộ lọc).")
    L.append("- **tau_penalized:** Kendall's Tau-a trên **tất cả** GT. GT không khớp được gán khoá dự đoán = N (N = số phần tử dự đoán), "
             "tức xếp **sau mọi phần tử dự đoán**. Các GT không khớp hoà nhau nên cặp giữa chúng đóng góp 0. "
             "Quy ước này áp dụng giống nhau cho mọi phương pháp.")
    L.append("- **Trang không xác định:** có < 2 phần tử đánh giá được thì Tau = None và được đếm riêng, không tính 1.0.")
    L.append("- **Bộ lọc GT:** giống Tầng 1 (bỏ `ignore`, `order` = None, và các loại `abandon`, `text_mask`, `table_mask`, `chart_mask`, `equation_semantic`, `list_group`).")
    L.append("- **Hệ toạ độ:** box Docling đổi từ đơn vị trang (điểm, gốc dưới-trái) sang pixel ảnh gốc, gốc trên-trái, "
             "bằng hệ số `pixel / pt` lấy từ kích thước ảnh và kích thước trang Docling.\n")

    # 3-5. Breakdowns
    def breakdown_table(title: str, rows: Dict[str, Any], note: str = "") -> None:
        L.append(f"## {title}\n")
        if note:
            L.append(f"{note}\n")
        L.append("| Nhóm | Số trang | A (docling) | B (xycut) | C (naive) | D (oracle) | Δ B−A | Coverage B | Coverage A |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for g, row in rows.items():
            mm = row["methods"]
            a, b = mm["A"]["tau_penalized_mean"], mm["B"]["tau_penalized_mean"]
            delta = None if (a is None or b is None) else b - a
            d_s = "—" if delta is None else f"{delta:+.4f}"
            L.append(f"| `{g}` | {row['n_pages']} | {f4(a)} | **{f4(b)}** | {f4(mm['C']['tau_penalized_mean'])} | "
                     f"{f4(mm['D']['tau_penalized_mean'])} | {d_s} | {f4(mm['B']['coverage_mean'])} | {f4(mm['A']['coverage_mean'])} |")
        L.append("")

    breakdown_table("3. Phân rã theo bố cục (layout)", res["breakdown"]["layout"],
                    "Chỉ số: tau_penalized trung bình. Cột Δ là B − A.")
    breakdown_table("4. Phân rã theo nguồn dữ liệu (data_source)", res["breakdown"]["data_source"],
                    "Chỉ số: tau_penalized trung bình. Cột Δ là B − A.")
    breakdown_table("5. Phân rã theo special_issue (chỉ nhóm có ≥ 4 trang)", res["breakdown"]["special_issue"],
                    "Một trang có thể thuộc nhiều nhóm special_issue. `standard_page` = không có special_issue (gồm `[]` và `['None']`).")

    # 6. Caveats
    L.append("## 6. Giới hạn và lưu ý (Caveat)\n")
    if cfg["mode"].startswith("stratified"):
        L.append(f"- **Mẫu:** phân tầng theo layout × data_source, seed {cfg['seed']}, {cfg['selected_pages']} trang "
                 f"(trên {cfg['dataset_pages_total']}). Đây là **ước lượng từ mẫu**, không phải toàn bộ dataset. "
                 "Khoảng tin cậy bootstrap ở mục 1.2 phản ánh sai số lấy mẫu.")
    elif cfg["mode"] == "all":
        L.append("- **Mẫu:** toàn bộ trang trong dataset.")
    else:
        L.append(f"- **Mẫu:** chế độ `{cfg['mode']}` (chạy thử, không đại diện toàn bộ dataset).")
    L.append(f"- **Thời gian:** Docling trung bình {tm['docling_mean_sec_per_page']} s/trang ({describe_docling(res)}). "
             f"Tổng thời gian Docling của các trang đã chọn: {tm['docling_total_sec_selected']} s. "
             f"Ước tính chạy toàn bộ 1.651 trang: ~{tm.get('estimated_full_dataset_hours_at_mean', '—')} giờ.")
    L.append(f"- **Trang bị loại:** {cnt['pages_with_lt2_gt']} trang có < 2 GT (Tau phạt = None); "
             f"{cnt['pages_tau_matched_undefined']} trang có < 2 box khớp (Tau khớp = None theo B); "
             f"{cnt['docling_errors']} trang lỗi Docling (được ghi trong `results.json` > `errors`, không tính vào thống kê).")
    L.append(f"- **Lệch tỉ lệ khung:** {cnt['pages_with_aspect_mismatch']} trang có tỉ lệ khung ảnh khác tỉ lệ trang Docling (có thể do xoay ảnh). Các trang này vẫn được tính, cần kiểm tra nếu số lượng lớn.")
    L.append("- **Tune-on-test chưa được xử lý:** `sort_elements_reading_order` được phát triển và chỉnh ngưỡng khi đã nhìn vào các trang OmniDocBench "
             "(Tầng 1 và bộ 20 trang Tầng 2). Đánh giá này dùng cùng dataset và không có tập giữ lại (held-out). Số của B vì vậy có thể lạc quan hơn thực tế.")
    L.append(f"- **Docling:** {describe_docling(res)}. Kết quả phản ánh layout và thứ tự của Docling trên ảnh, "
             "không đo chất lượng nhận dạng chữ.")
    if cfg.get("include_furniture"):
        n_furn = sum(r.get("n_furniture_pred", 0) for r in res["per_page"] if r.get("status") == "ok")
        L.append("- **Lớp FURNITURE:** bật. Các item `page_header` / `page_footer` (lớp FURNITURE của docling_core) được "
                 "đưa vào danh sách dự đoán, nên GT header, page_number và footer có thể khớp. Thứ tự A là thứ tự của "
                 "`iterate_items()` với đủ lớp; B và C cũng nhận các item này. "
                 f"Tổng số dự đoán FURNITURE trên các trang đánh giá: **{n_furn}**.")
    else:
        L.append("- **Lớp FURNITURE:** tắt (mặc định). `iterate_items()` chỉ duyệt lớp BODY, nên các item FURNITURE "
                 "(header, footer, số trang) bị loại trước khi khớp; GT header, page_number, footer chỉ khớp được "
                 "khi một item BODY trùng vị trí.")
    L.append("- **Matching:** một-một theo IoU. Một box Docling gộp nhiều GT, hoặc một GT bị chia nhỏ, sẽ làm giảm coverage và bị phạt trong tau_penalized.")
    L.append("- **Quy ước GT không khớp:** xếp cuối và hoà nhau là một lựa chọn theo đặc tả. GT không khớp nằm ở đầu trang (ví dụ `header`, `page_number`, tiêu đề) bị phạt mạnh vì nó đứng sau mọi phần tử dự đoán; GT không khớp nằm ở cuối trang thì được cộng điểm. "
             "Vì vậy tau_penalized tuyệt đối phụ thuộc nhiều vào coverage (mục 1.4). Khi so sánh B với A, hai phương pháp dùng chung matching nên đóng góp của cặp có GT không khớp là giống hệt nhau; "
             "Δ tau_penalized chỉ đến từ các cặp đã khớp và bị co lại theo tỉ lệ cặp đã khớp trên tổng số cặp.")
    L.append("- **Nhãn special_issue:** trong dataset, `['None']` nghĩa là không có special_issue. Bản cũ của Tầng 1 coi đó là một nhãn tên `None`; bản mới loại khỏi nhóm special_issue.")
    L.append("- **group_whole_diagrams:** không được gọi trong đánh giá này. Hàm này chỉ thêm `diagram_groups` và không thay đổi danh sách `elements` hay thứ tự sắp xếp, nên không ảnh hưởng đến các chỉ số thứ tự.")
    L.append("- **D (oracle):** dùng box GT, không phải một phương pháp chạy trên box Docling. Nó là mốc tham chiếu, không phải kết quả end-to-end.")

    if res.get("comparison_vs_baseline"):
        L.extend(comparison_markdown(res["comparison_vs_baseline"]))

    path.write_text("\n".join(L).strip() + "\n", encoding="utf-8")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="E2E reading-order evaluation: Docling vs DocuMind XY-Cut vs baselines.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="run all pages in the dataset")
    g.add_argument("--stratified", type=int, metavar="N", help="stratified sample of N pages (layout x data_source)")
    ap.add_argument("--limit", type=int, default=0, help="keep only the first N selected pages (dataset order)")
    ap.add_argument("--seed", type=int, default=42, help="seed for stratified sampling and bootstrap (default 42)")
    ap.add_argument("--iou-thr", type=float, default=0.3, help="IoU threshold for one-to-one matching (default 0.3)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="output directory (default outputs/04_e2e_eval)")
    ap.add_argument("--ocr-backend", choices=["auto", "onnxruntime", "torch", "off"], default="auto",
                    help="OCR engine: auto (Docling picks RapidOCR/onnxruntime; legacy default), "
                         "onnxruntime, torch (RapidOCR torch backend, PP-OCRv6 default model), "
                         "or off (do_ocr=False; cache folder cache_ocr-off_<device>)")
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu",
                    help="accelerator for layout/table models and RapidOCR use_cuda (default cpu = legacy "
                         "conditions; Docling's own default is auto)")
    ap.add_argument("--pages-file", type=Path, default=None,
                    help="optional text file with one image_path per line; restricts the run to those pages")
    ap.add_argument("--include-furniture", action="store_true",
                    help="traverse all content layers (adds FURNITURE page_header/page_footer items as predictions). "
                         "Default off = BODY only, as in the legacy results. Uses its own cache folder "
                         "(suffix _furniture) and writes results_furniture.json / report_furniture.md / "
                         "run_furniture.log, so the legacy outputs are not overwritten")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
