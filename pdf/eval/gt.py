from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .._test_pdf import TestPdfBuilder
from .metrics import char_counts

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PT_PER_PX_150DPI = 72.0 / 150.0

OMNIDOC_SAMPLE_PAGES = (
    "PPT_1001115_eng_page_003.png",
    "PPT_Keuk Chan Narith_page_009.png",
    "PPT_english-guidelines-for-humanities-students.zp199743_page_002.png",
    "book_en_国外数学教材-An Excursion through Elementary Mathematics（初等数学漫游）-An excursion through elementary mathematics, Vol.3 (Caminha Muniz Neto A)_0035.png",
    "book_en_搬书匠-3396-iPhone The Missing Manual 10th Edition-2017-英文版_page_126.png",
    "docstructbench_enbook-zlib-o.O-22206811.pdf_5.jpg",
    "docstructbench_llm-raw-scihub-o.O-j.1444-2906.2002.00466.x.pdf_4.jpg",
    "docstructbench_llm-raw-scihub-o.O-s001280000131.pdf_5.jpg",
    "exam_paper_en-file-putnam-archive_1993_Problems_1993_page_001.png",
    "exam_paper_en-file-putnam-archive_2007_Solutions_2007s_page_003.png",
    "jiaocaineedrop_Evans_PDE_Solution_Chapter_6_Second-Order_Elliptic_Equations.pdf_5.jpg",
    "magazine_TheEconomist.2023.11.18_page_082.png",
    "magazine_TheEconomist.2024.03.09_page_029.png",
    "newspaper_NY Daily News_August 1@magazinesclubnew_page_007.png",
    "newspaper_d7fe1132386670993aae3dabc6247bc9_1.jpg",
    "page-39f38730-0ae6-4cfe-8b99-c65d06d09544.png",
    "page-9390721f-e06c-47ea-8b21-0f27c72df7b7.png",
    "page-d4fe8fcc-37e2-4ac1-9ae2-7d00ac937937.png",
    "scihub_j.conbuildmat.2019.117698.pdf_5.jpg",
    "yanbaopptmerge_abef2a4978ae4d13e931f0392502bd40.pdf_1130.jpg",
)


@dataclass(frozen=True)
class AutosarDocGT:
    stem: str
    pdf_path: Path
    pages: list[int]
    boxes_by_page: dict[int, list[list[float]]]


def resolve_autosar_gt_path(gt_path: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if gt_path is not None:
        candidates.append(Path(gt_path))
    env_gt = os.getenv("DOCUMIND_AUTOSAR_GT")
    if env_gt:
        candidates.append(Path(env_gt))
    candidates.append(Path(__file__).with_name("autosar_figures_v2.json"))
    candidates.append(_REPO_ROOT / "outputs" / "13_eval_set" / "autosar_figures_v2.json")
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError("autosar_figures_v2.json not found; pass --gt or set DOCUMIND_AUTOSAR_GT")


def load_autosar_gt(
    split: str = "dev",
    gt_path: str | Path | None = None,
    pdf_dir: Path | None = None,
) -> list[AutosarDocGT]:
    if split not in ("dev", "test"):
        raise ValueError(f"unknown split: {split!r}")
    resolved_gt = resolve_autosar_gt_path(gt_path)
    raw = json.loads(resolved_gt.read_text(encoding="utf-8-sig"))
    src_dir = pdf_dir if pdf_dir is not None else (_REPO_ROOT / "datasets" / "AUTOSAR")

    docs: list[AutosarDocGT] = []
    for stem, doc_data in raw["documents"].items():
        if split == "dev":
            if "dev_pages" not in doc_data:
                continue
            pages = [int(p) for p in doc_data["dev_pages"]]
            figs_raw = doc_data["dev_figures"]
        else:
            pages = [int(p) for p in doc_data["test_pages"]]
            figs_raw = doc_data["figures"]

        boxes_by_page = {
            p: [
                [float(v) for v in f["bbox"]]
                for f in figs_raw.get(str(p), [])
                if "small" not in f.get("flags", ())
            ]
            for p in pages
        }
        docs.append(
            AutosarDocGT(
                stem=stem,
                pdf_path=src_dir / f"{stem}.pdf",
                pages=pages,
                boxes_by_page=boxes_by_page,
            )
        )
    return docs


def load_omnidoc_gt(
    gt_file: Path | None = None,
    img_dir: Path | None = None,
    pages: int | None = None,
) -> list[tuple[str, Path, Counter[str]]]:
    resolved_gt = gt_file if gt_file is not None else (_REPO_ROOT / "datasets" / "OmniDocBench" / "OmniDocBench.json")
    resolved_imgs = img_dir if img_dir is not None else (_REPO_ROOT / "datasets" / "OmniDocBench" / "images")
    gt_data = json.loads(resolved_gt.read_text(encoding="utf-8"))
    by_name = {d["page_info"]["image_path"]: d for d in gt_data}

    names = OMNIDOC_SAMPLE_PAGES[:pages] if pages is not None else OMNIDOC_SAMPLE_PAGES
    selected: list[tuple[str, Path, Counter[str]]] = []
    for name in names:
        img_path = resolved_imgs / name
        if not img_path.exists() or name not in by_name:
            continue
        gt_raw = "".join(b.get("text", "") for b in by_name[name]["layout_dets"] if b.get("text"))
        selected.append((name, img_path, char_counts(gt_raw)))
    return selected


def build_raster_pdf(image_paths: Sequence[Path], dst: Path) -> None:
    builder = TestPdfBuilder()
    for img_path in image_paths:
        with Image.open(img_path) as im:
            w, h = im.size
        pw, ph = w * 72.0 / 150.0, h * 72.0 / 150.0
        page = builder.new_page(width=pw, height=ph)
        page.insert_image((0.0, 0.0, pw, ph), stream=img_path.read_bytes())
    builder.save(dst)


def omnidoc_image_split(image_name: str) -> str:
    digest = hashlib.sha256(image_name.encode("utf-8")).digest()
    return "dev" if (digest[0] & 1) == 0 else "test"


def append_run_log(log_file: Path, record: dict[str, Any]) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def check_split_guard(split: str, final: bool, *, stream: Any = None) -> bool:
    if split not in ("dev", "test"):
        raise ValueError(f"unknown split: {split!r}")
    if split == "test" and not final:
        out = sys.stderr if stream is None else stream
        print("REFUSED: --split test requires --final to prevent accidental test-split evaluation.", file=out)
        return False
    return True


def _check_nonempty_str(val: Any, ctx: str, errors: list[str]) -> bool:
    if isinstance(val, str) and val.strip():
        return True
    errors.append(f"{ctx}: expected non-empty string")
    return False


def _check_pos_int(val: Any, ctx: str, errors: list[str]) -> bool:
    if isinstance(val, int) and not isinstance(val, bool) and val >= 1:
        return True
    errors.append(f"{ctx}: expected integer >= 1")
    return False


def _check_bool(val: Any, ctx: str, errors: list[str]) -> bool:
    if isinstance(val, bool):
        return True
    errors.append(f"{ctx}: expected boolean")
    return False


def _check_bbox_pt(val: Any, ctx: str, errors: list[str]) -> list[float] | None:
    if not isinstance(val, Sequence) or isinstance(val, (str, bytes)) or len(val) != 4:
        errors.append(f"{ctx}: expected 4-element bbox_pt")
        return None
    coords: list[float] = []
    for coord in val:
        if isinstance(coord, bool) or not isinstance(coord, (int, float)) or not math.isfinite(float(coord)):
            errors.append(f"{ctx}: bbox_pt coordinates must be finite numbers")
            return None
        coords.append(float(coord))
    if coords[2] <= coords[0] or coords[3] <= coords[1]:
        errors.append(f"{ctx}: bbox_pt must have x1 > x0 and y1 > y0")
        return None
    return coords


def _check_optional_split(entry: Mapping[str, Any], ctx: str, errors: list[str]) -> None:
    if "split" in entry and entry["split"] is not None and entry["split"] not in ("dev", "test"):
        errors.append(f"{ctx}.split: expected 'dev' or 'test'")


def validate_router_gt(data: Any) -> list[str]:
    if not isinstance(data, list):
        return ["router GT root must be a list"]
    errors: list[str] = []
    for idx, entry in enumerate(data):
        ctx = f"router[{idx}]"
        if not isinstance(entry, Mapping):
            errors.append(f"{ctx}: expected object")
            continue
        _check_nonempty_str(entry.get("doc"), f"{ctx}.doc", errors)
        _check_pos_int(entry.get("page"), f"{ctx}.page", errors)
        if entry.get("kind") not in ("native", "raster", "hybrid"):
            errors.append(f"{ctx}.kind: expected 'native', 'raster', or 'hybrid'")
        _check_bool(entry.get("suspect_ocr_layer"), f"{ctx}.suspect_ocr_layer", errors)
        _check_bool(entry.get("has_partial_raster"), f"{ctx}.has_partial_raster", errors)
        _check_optional_split(entry, ctx, errors)
    return errors


def validate_autosar_diagram_gt(data: Any) -> list[str]:
    if not isinstance(data, list):
        return ["autosar diagram GT root must be a list"]
    errors: list[str] = []
    for idx, entry in enumerate(data):
        ctx = f"diagram[{idx}]"
        if not isinstance(entry, Mapping):
            errors.append(f"{ctx}: expected object")
            continue
        _check_nonempty_str(entry.get("doc"), f"{ctx}.doc", errors)
        _check_pos_int(entry.get("page"), f"{ctx}.page", errors)
        _check_bbox_pt(entry.get("bbox_pt"), f"{ctx}.bbox_pt", errors)
        _check_optional_split(entry, ctx, errors)

        nodes = entry.get("nodes")
        node_ids: set[str] = set()
        if not isinstance(nodes, list):
            errors.append(f"{ctx}.nodes: expected list")
        else:
            for n_idx, node in enumerate(nodes):
                n_ctx = f"{ctx}.nodes[{n_idx}]"
                if not isinstance(node, Mapping):
                    errors.append(f"{n_ctx}: expected object")
                    continue
                if _check_nonempty_str(node.get("id"), f"{n_ctx}.id", errors):
                    nid = str(node["id"])
                    if nid in node_ids:
                        errors.append(f"{n_ctx}.id: duplicate node id {nid!r}")
                    node_ids.add(nid)
                _check_bbox_pt(node.get("bbox_pt"), f"{n_ctx}.bbox_pt", errors)
                if not isinstance(node.get("label"), str):
                    errors.append(f"{n_ctx}.label: expected string")

        edges = entry.get("edges")
        if not isinstance(edges, list):
            errors.append(f"{ctx}.edges: expected list")
        else:
            for e_idx, edge in enumerate(edges):
                e_ctx = f"{ctx}.edges[{e_idx}]"
                if not isinstance(edge, Mapping):
                    errors.append(f"{e_ctx}: expected object")
                    continue
                for ep in ("source_id", "target_id"):
                    val = edge.get(ep)
                    if val is not None:
                        if not isinstance(val, str) or not val.strip():
                            errors.append(f"{e_ctx}.{ep}: expected string or null")
                        elif isinstance(nodes, list) and val not in node_ids:
                            errors.append(f"{e_ctx}.{ep}: unknown node id {val!r}")
                if edge.get("direction") not in ("directed", "undirected", "bidirectional"):
                    errors.append(f"{e_ctx}.direction: expected 'directed', 'undirected', or 'bidirectional'")
                if not isinstance(edge.get("label", ""), str):
                    errors.append(f"{e_ctx}.label: expected string")

        groups = entry.get("groups")
        group_ids: set[str] = set()
        if not isinstance(groups, list):
            errors.append(f"{ctx}.groups: expected list")
        else:
            for g_idx, group in enumerate(groups):
                g_ctx = f"{ctx}.groups[{g_idx}]"
                if not isinstance(group, Mapping):
                    errors.append(f"{g_ctx}: expected object")
                    continue
                if _check_nonempty_str(group.get("id"), f"{g_ctx}.id", errors):
                    gid = str(group["id"])
                    if gid in group_ids:
                        errors.append(f"{g_ctx}.id: duplicate group id {gid!r}")
                    group_ids.add(gid)
                _check_bbox_pt(group.get("bbox_pt"), f"{g_ctx}.bbox_pt", errors)
                mids = group.get("member_ids")
                if not isinstance(mids, list) or not all(isinstance(m, str) and m.strip() for m in mids):
                    errors.append(f"{g_ctx}.member_ids: expected list of non-empty strings")
                elif isinstance(nodes, list):
                    for mid in mids:
                        if mid not in node_ids:
                            errors.append(f"{g_ctx}.member_ids: unknown node id {mid!r}")
    return errors


def validate_autosar_table_gt(data: Any) -> list[str]:
    if not isinstance(data, list):
        return ["autosar table GT root must be a list"]
    errors: list[str] = []
    for idx, entry in enumerate(data):
        ctx = f"table[{idx}]"
        if not isinstance(entry, Mapping):
            errors.append(f"{ctx}: expected object")
            continue
        _check_nonempty_str(entry.get("doc"), f"{ctx}.doc", errors)
        _check_pos_int(entry.get("page"), f"{ctx}.page", errors)
        _check_bbox_pt(entry.get("bbox_pt"), f"{ctx}.bbox_pt", errors)
        html_str = entry.get("html")
        if not isinstance(html_str, str) or "<table" not in html_str.lower():
            errors.append(f"{ctx}.html: expected HTML string containing '<table'")
        _check_bool(entry.get("has_merged"), f"{ctx}.has_merged", errors)
        _check_optional_split(entry, ctx, errors)
    return errors


def _resolve_optional_gt(
    gt_path: str | Path | None,
    env_var: str,
    default_filenames: Sequence[str],
    *,
    missing_ok: bool,
) -> Path | None:
    if gt_path is not None:
        p = Path(gt_path)
        if p.is_file():
            return p
        if missing_ok:
            return None
        raise FileNotFoundError(f"GT file not found: {p}")
    candidates: list[Path] = []
    env_val = os.getenv(env_var)
    if env_val:
        candidates.append(Path(env_val))
    for name in default_filenames:
        candidates.append(Path(__file__).with_name(name))
        candidates.append(_REPO_ROOT / "outputs" / "13_eval_set" / name)
        candidates.append(_REPO_ROOT / "datasets" / "AUTOSAR" / name)
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _resolve_pdf_for_doc(doc_name: str, pdf_dir: Path) -> Path:
    p = Path(doc_name)
    if p.is_absolute():
        return p
    if p.suffix.lower() != ".pdf":
        return pdf_dir / f"{doc_name}.pdf"
    return pdf_dir / p


def _load_validated_json_gt(
    *,
    split: str,
    gt_path: str | Path | None,
    env_var: str,
    default_filenames: Sequence[str],
    validator: Any,
    schema_name: str,
    pdf_dir: Path | None,
    missing_ok: bool | None,
) -> list[dict[str, Any]]:
    if split not in ("dev", "test"):
        raise ValueError(f"unknown split: {split!r}")
    allow_missing = (gt_path is None) if missing_ok is None else missing_ok
    resolved = _resolve_optional_gt(gt_path, env_var, default_filenames, missing_ok=allow_missing)
    if resolved is None:
        return []
    raw = json.loads(resolved.read_text(encoding="utf-8-sig"))
    errors = validator(raw)
    if errors:
        raise ValueError(f"invalid {schema_name} GT in {resolved}: {'; '.join(errors[:5])}")
    src_dir = pdf_dir if pdf_dir is not None else (_REPO_ROOT / "datasets" / "AUTOSAR")
    out: list[dict[str, Any]] = []
    for entry in raw:
        item_split = entry.get("split")
        if item_split is not None and item_split != split:
            continue
        item = dict(entry)
        item["page"] = int(entry["page"])
        if "bbox_pt" in item:
            item["bbox_pt"] = [float(v) for v in item["bbox_pt"]]
        item.setdefault("pdf_path", _resolve_pdf_for_doc(str(entry["doc"]), src_dir))
        out.append(item)
    return out


def load_router_gt(
    split: str = "dev",
    gt_path: str | Path | None = None,
    *,
    pdf_dir: Path | None = None,
    missing_ok: bool | None = None,
) -> list[dict[str, Any]]:
    return _load_validated_json_gt(
        split=split,
        gt_path=gt_path,
        env_var="DOCUMIND_ROUTER_GT",
        default_filenames=("router_gt.json",),
        validator=validate_router_gt,
        schema_name="router",
        pdf_dir=pdf_dir,
        missing_ok=missing_ok,
    )


def load_autosar_diagram_gt(
    split: str = "dev",
    gt_path: str | Path | None = None,
    *,
    pdf_dir: Path | None = None,
    missing_ok: bool | None = None,
) -> list[dict[str, Any]]:
    return _load_validated_json_gt(
        split=split,
        gt_path=gt_path,
        env_var="DOCUMIND_AUTOSAR_DIAGRAM_GT",
        default_filenames=("autosar_diagrams_gt.json",),
        validator=validate_autosar_diagram_gt,
        schema_name="autosar diagram",
        pdf_dir=pdf_dir,
        missing_ok=missing_ok,
    )


def load_autosar_table_gt(
    split: str = "dev",
    gt_path: str | Path | None = None,
    *,
    pdf_dir: Path | None = None,
    missing_ok: bool | None = None,
) -> list[dict[str, Any]]:
    return _load_validated_json_gt(
        split=split,
        gt_path=gt_path,
        env_var="DOCUMIND_AUTOSAR_TABLE_GT",
        default_filenames=("autosar_tables_gt.json",),
        validator=validate_autosar_table_gt,
        schema_name="autosar table",
        pdf_dir=pdf_dir,
        missing_ok=missing_ok,
    )
