from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PIL import Image

from .metrics import char_counts

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

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
    out = pymupdf.open()
    try:
        for img_path in image_paths:
            with Image.open(img_path) as im:
                w, h = im.size
            page = out.new_page(width=w * 72.0 / 150.0, height=h * 72.0 / 150.0)
            page.insert_image(page.rect, stream=img_path.read_bytes())
        out.save(str(dst))
    finally:
        out.close()
