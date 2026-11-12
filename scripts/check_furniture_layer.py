"""
Kiem chung gia thuyet FURNITURE: Docling co tao item lop FURNITURE (page_header / page_footer)
khong, cac item do co bi loai khi duyet doc.iterate_items() mac dinh (chi lop BODY) khong, va
chung co khop GT header / page_number / footer / equation_caption khong.

Chay Docling that (cung cau hinh torch + cuda nhu cache chinh) tren N trang co GT header,
page_number hoac footer trong 300 trang phan tang da danh gia. Khong ghi vao cache chinh.

Voi moi trang in:
  - so item theo content_layer (body / furniture / khac)
  - iterate_items() mac dinh so voi iterate_items(included_content_layers=all layers)
  - nhan va bbox cua cac item FURNITURE, va co nam trong duyet mac dinh / duyet day du khong
  - matching mot-mot (IoU >= 0.3) cua GT voi (a) item mac dinh, (b) tat ca item
  - voi moi GT header/page_number/footer/equation_caption: IoU tot nhat voi item FURNITURE,
    voi item bat ky, va nhan cua item do

Usage:
  .venv\\Scripts\\python scripts\\check_furniture_layer.py --n 10
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from docling_core.types.doc import ContentLayer  # noqa: E402

import run_e2e_reading_order_eval as E  # noqa: E402
from eval_utils import (  # noqa: E402
    docling_bbox_to_pixel,
    gt_elements_from_page,
    iou_matrix,
    match_one_to_one,
)

IOU_THR = 0.3
PICK_TYPES = ("header", "page_number", "footer")
CHECK_TYPES = ("header", "page_number", "footer", "equation_caption")


def doc_items_all(doc) -> List[Any]:
    items: List[Any] = []
    for attr in ("texts", "tables", "pictures", "key_value_items", "form_items"):
        items.extend(getattr(doc, attr, None) or [])
    return items


def layer_name(it) -> str:
    cl = getattr(it, "content_layer", None)
    return cl.value if hasattr(cl, "value") else str(cl)


def label_name(it) -> str:
    raw = getattr(it, "label", None)
    return raw.value if hasattr(raw, "value") else type(it).__name__


def pixel_bbox(it, p_h: float, sx: float, sy: float) -> Optional[List[float]]:
    prov = getattr(it, "prov", None)
    if not prov:
        return None
    b = prov[0].bbox.to_top_left_origin(p_h)
    return docling_bbox_to_pixel([b.l, b.t, b.r, b.b], sx, sy)


def pick_pages(results_path: Path, n: int, pick_types: Sequence[str]) -> List[Dict[str, Any]]:
    """Deterministic choice from the 300 evaluated pages: prefer new data_source or new GT type first."""
    res = json.loads(results_path.read_text(encoding="utf-8"))
    cands = [r for r in res["per_page"]
             if r.get("status") == "ok" and any(t in r.get("gt_by_type", {}) for t in pick_types)]
    cands.sort(key=lambda r: r["image_path"])
    chosen: List[Dict[str, Any]] = []
    seen_src, seen_types = set(), set()
    for r in cands:
        if len(chosen) >= n:
            break
        types = {t for t in pick_types if t in r["gt_by_type"]}
        if r["data_source"] not in seen_src or not types <= seen_types:
            chosen.append(r)
            seen_src.add(r["data_source"])
            seen_types |= types
    for r in cands:
        if len(chosen) >= n:
            break
        if r not in chosen:
            chosen.append(r)
    return chosen


def check_page(converter, item: Dict[str, Any], img_rel: str, cache: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    img_path = E.IMAGES_DIR / img_rel
    conv = converter.convert(str(img_path))
    doc = conv.document
    page = doc.pages[1] if 1 in doc.pages else next(iter(doc.pages.values()))
    p_w, p_h = float(page.size.width), float(page.size.height)
    with Image.open(img_path) as im:
        img_w, img_h = im.size
    sx, sy = img_w / p_w, img_h / p_h

    recs: List[Dict[str, Any]] = []
    for it in doc_items_all(doc):
        recs.append({
            "self_ref": it.self_ref,
            "label": label_name(it),
            "layer": layer_name(it),
            "bbox": pixel_bbox(it, p_h, sx, sy),
            "text": (getattr(it, "text", "") or "")[:40],
        })

    default_refs = [it.self_ref for it, _ in doc.iterate_items()]
    full_refs = [it.self_ref for it, _ in doc.iterate_items(included_content_layers=set(ContentLayer))]
    default_set, full_set = set(default_refs), set(full_refs)
    for r in recs:
        r["in_default_iter"] = r["self_ref"] in default_set
        r["in_full_iter"] = r["self_ref"] in full_set

    # Consistency with the cached BODY-only run of the same page (reproducibility of the fresh run)
    cache_match = None
    if cache is not None and cache.get("items") is not None:
        ref_to_rec = {r["self_ref"]: r for r in recs}
        fresh_labels = [ref_to_rec[ref]["label"] for ref in default_refs]
        cache_labels = [c["label"] for c in cache["items"]]
        cache_match = {
            "n_default_fresh": len(default_refs),
            "n_default_cache": len(cache_labels),
            "labels_equal": fresh_labels == cache_labels,
        }

    full_order_of_default = [r for r in full_refs if r in default_set]
    default_is_subsequence = full_order_of_default == default_refs

    gt = gt_elements_from_page(item)
    gt_boxes = [g["bbox"] for g in gt]

    default_pred = [r for r in recs if r["in_default_iter"] and r["bbox"] and len(r["bbox"]) == 4]
    full_pred = [r for r in recs if r["in_full_iter"] and r["bbox"] and len(r["bbox"]) == 4]
    g2p_default, _ = match_one_to_one([r["bbox"] for r in default_pred], gt_boxes, IOU_THR)
    g2p_full, _ = match_one_to_one([r["bbox"] for r in full_pred], gt_boxes, IOU_THR)

    all_pred = [r for r in recs if r["bbox"] and len(r["bbox"]) == 4]
    ious = iou_matrix(gt_boxes, [r["bbox"] for r in all_pred]) if gt_boxes and all_pred else None

    gt_rows: List[Dict[str, Any]] = []
    for gi, g in enumerate(gt):
        row: Dict[str, Any] = {
            "gt_type": g["type"],
            "gt_bbox": [round(v, 1) for v in g["bbox"]],
            "gt_text": g["text"],
            "matched_default": gi in g2p_default,
            "matched_full": gi in g2p_full,
            "best_any": None,
            "best_furniture": None,
        }
        if ious is not None:
            row_ious = ious[gi]
            gb = np.array(g["bbox"], dtype=float)
            pb = np.array([r["bbox"] for r in all_pred], dtype=float)
            ix = np.clip(np.minimum(pb[:, 2], gb[2]) - np.maximum(pb[:, 0], gb[0]), 0, None)
            iy = np.clip(np.minimum(pb[:, 3], gb[3]) - np.maximum(pb[:, 1], gb[1]), 0, None)
            cont = (ix * iy) / max((gb[2] - gb[0]) * (gb[3] - gb[1]), 1e-9)
            jc = int(cont.argmax())
            row["best_contain"] = {"contain": round(float(cont[jc]), 3), "iou": round(float(row_ious[jc]), 3),
                                   "label": all_pred[jc]["label"], "layer": all_pred[jc]["layer"],
                                   "text": all_pred[jc]["text"]}
            j = int(row_ious.argmax())
            best = all_pred[j]
            row["best_any"] = {"iou": round(float(row_ious[j]), 3), "label": best["label"],
                               "layer": best["layer"], "text": best["text"]}
            furn_idx = [k for k, r in enumerate(all_pred) if r["layer"] == "furniture"]
            if furn_idx:
                k = max(furn_idx, key=lambda k: row_ious[k])
                fb = all_pred[k]
                row["best_furniture"] = {"iou": round(float(row_ious[k]), 3), "label": fb["label"],
                                         "text": fb["text"]}
        gt_rows.append(row)

    furn = [r for r in recs if r["layer"] == "furniture"]
    full_pos = {ref: i for i, ref in enumerate(full_refs)}
    return {
        "image_path": img_rel,
        "n_items_all": len(recs),
        "n_default": len(default_refs),
        "n_full": len(full_refs),
        "layer_counts": dict(Counter(r["layer"] for r in recs)),
        "furniture_items": [
            {"label": r["label"], "text": r["text"], "bbox": [round(v, 1) for v in r["bbox"]] if r["bbox"] else None,
             "in_default_iter": r["in_default_iter"], "in_full_iter": r["in_full_iter"],
             "order_in_full": full_pos.get(r["self_ref"]), "n_full": len(full_refs)}
            for r in furn
        ],
        "furniture_in_default_iter": sum(r["in_default_iter"] for r in furn),
        "furniture_in_full_iter": sum(r["in_full_iter"] for r in furn),
        "full_minus_default_layers": dict(Counter(r["layer"] for r in recs if r["in_full_iter"] and not r["in_default_iter"])),
        "default_is_subsequence_of_full": default_is_subsequence,
        "cache_consistency": cache_match,
        "n_pred_default": len(default_pred),
        "n_pred_full": len(full_pred),
        "gt_rows": gt_rows,
    }


def summarize(pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_type: Dict[str, Dict[str, int]] = defaultdict(lambda: Counter())
    furn_labels: Counter = Counter()
    furn_total = furn_default = furn_full = 0
    furn_overlap_gt: Counter = Counter()
    unmatched_best_label: Dict[str, Counter] = defaultdict(Counter)
    unmatched_contain: Dict[str, Counter] = defaultdict(Counter)
    for p in pages:
        for fi in p["furniture_items"]:
            furn_total += 1
            furn_labels[fi["label"]] += 1
            furn_default += fi["in_default_iter"]
            furn_full += fi["in_full_iter"]
        for row in p["gt_rows"]:
            t = row["gt_type"]
            if t not in CHECK_TYPES:
                continue
            c = per_type[t]
            c["n_gt"] += 1
            c["matched_default"] += row["matched_default"]
            c["matched_full"] += row["matched_full"]
            if row["best_furniture"] and row["best_furniture"]["iou"] >= IOU_THR:
                c["best_furniture_iou_ge_thr"] += 1
                furn_overlap_gt[(t, row["best_furniture"]["label"])] += 1
            if not row["matched_default"] and row.get("best_contain"):
                bc = row["best_contain"]
                flag = "contain>=0.7" if bc["contain"] >= 0.7 else "contain<0.7"
                unmatched_contain[t][f"{bc['label']}|{bc['layer']}|{flag}"] += 1
            if not row["matched_default"] and row["best_any"]:
                unmatched_best_label[t][f"{row['best_any']['label']}|{row['best_any']['layer']}"
                                        + ("|iou>=thr" if row["best_any"]["iou"] >= IOU_THR else "|iou<thr")] += 1
    return {
        "n_pages": len(pages),
        "furniture_items_total": furn_total,
        "furniture_in_default_iter": furn_default,
        "furniture_in_full_iter": furn_full,
        "furniture_labels": dict(furn_labels),
        "per_gt_type": {t: dict(v) for t, v in per_type.items()},
        "furniture_overlapping_gt": {f"{t}|{lab}": n for (t, lab), n in furn_overlap_gt.items()},
        "unmatched_default_best_any_label": {t: dict(v.most_common(8)) for t, v in unmatched_best_label.items()},
        "unmatched_default_best_contain_label": {t: dict(v.most_common(8)) for t, v in unmatched_contain.items()},
        "default_is_subsequence_all_pages": all(p["default_is_subsequence_of_full"] for p in pages),
        "cache_labels_equal_all_pages": all(
            (p["cache_consistency"] or {}).get("labels_equal", False) for p in pages),
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Check whether Docling FURNITURE items are dropped by default iteration.")
    ap.add_argument("--n", type=int, default=10, help="number of pages to check (default 10)")
    ap.add_argument("--ocr-backend", choices=["auto", "onnxruntime", "torch", "off"], default="torch")
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    ap.add_argument("--pick-types", default=",".join(PICK_TYPES),
                    help="comma-separated GT types that the chosen pages must contain (default header,page_number,footer)")
    ap.add_argument("--out", type=Path, default=E.DEFAULT_OUT_DIR / "furniture_check.json")
    args = ap.parse_args()
    pick_types = [t.strip() for t in args.pick_types.split(",") if t.strip()]

    with open(E.DATASET_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    chosen = pick_pages(E.DEFAULT_OUT_DIR / "results.json", args.n, pick_types)
    cache_dir = E.cache_dir_for(E.DEFAULT_OUT_DIR, args.ocr_backend, args.device)
    print(f"Chosen {len(chosen)} pages from results.json; cache_dir={cache_dir.name}", flush=True)

    converter = E.build_converter(args.ocr_backend, args.device)
    pages: List[Dict[str, Any]] = []
    for k, r in enumerate(chosen, 1):
        item = data[r["dataset_index"]]
        img_rel = item["page_info"]["image_path"]
        cache = E.load_cache(cache_dir / f"{Path(img_rel).stem}.json")
        res = check_page(converter, item, img_rel, cache)
        res.update({"dataset_index": r["dataset_index"], "data_source": r["data_source"],
                    "layout": r["layout"], "gt_by_type_eval": r["gt_by_type"]})
        pages.append(res)
        print(f"[{k}/{len(chosen)}] {Path(img_rel).stem[:30]:30s} {r['data_source']:18s} "
              f"items={res['n_items_all']:3d} default={res['n_default']:3d} full={res['n_full']:3d} "
              f"furniture={len(res['furniture_items'])} "
              f"furn_in_default={res['furniture_in_default_iter']} furn_in_full={res['furniture_in_full_iter']} "
              f"pred_default={res['n_pred_default']} pred_full={res['n_pred_full']}", flush=True)
        for fi in res["furniture_items"]:
            print(f"      FURN {fi['label']:12s} bbox={fi['bbox']} text={fi['text'][:30]!r} "
                  f"in_default={fi['in_default_iter']} in_full={fi['in_full_iter']}", flush=True)

    summary = summarize(pages)
    out = {"config": {"ocr_backend": args.ocr_backend, "device": args.device, "n_pages": len(pages),
                      "iou_thr": IOU_THR, "cache_dir": cache_dir.name,
                      "selection": f"per_page of outputs/04_e2e_eval/results.json with GT {','.join(pick_types)}", "pick_types": pick_types},
           "summary": summary, "pages": pages}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(args.out)
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
