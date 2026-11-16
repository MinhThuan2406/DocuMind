from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wintypes
import json
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pymupdf  # noqa: E402
from PIL import Image  # noqa: E402

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


def _peak_rss_mb() -> float:
    if sys.platform != "win32":
        import resource

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)

    class _PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
    k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(_PMC)
    k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return round(pmc.PeakWorkingSetSize / (1024.0 * 1024.0), 2)


def _make_subset_pdf(src: Path, dst: Path, pages: list[int]) -> None:
    doc = pymupdf.open(str(src))
    out = pymupdf.open()
    for pno in pages:
        out.insert_pdf(doc, from_page=pno - 1, to_page=pno - 1)
    out.save(str(dst))
    out.close()
    doc.close()


def _make_raster_pdf_from_images(image_paths: list[Path], dst: Path) -> None:
    out = pymupdf.open()
    for img_path in image_paths:
        with Image.open(img_path) as im:
            w, h = im.size
        page = out.new_page(width=w * 72.0 / 150.0, height=h * 72.0 / 150.0)
        page.insert_image(page.rect, stream=img_path.read_bytes())
    out.save(str(dst))
    out.close()


def _run_isolated_bench(mode: str, device: str, pages: int) -> dict[str, object]:
    cmd = [sys.executable, "-B", __file__, "--worker", mode, "--device", device, "--pages", str(pages)]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _worker(mode: str, device: str, n_pages: int) -> None:
    layered = ROOT / "datasets" / "AUTOSAR" / "AUTOSAR_CP_EXP_LayeredSoftwareArchitecture.pdf"
    omni_dir = ROOT / "datasets" / "OmniDocBench" / "images"

    t_import0 = time.perf_counter()
    from pdf import extract_ir

    import_sec = time.perf_counter() - t_import0

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        sample_pdf = tmp_path / "sample.pdf"
        if mode in ("native", "figures_live", "figures_cached"):
            if not layered.exists():
                print(json.dumps({"mode": mode, "skipped": "Layered PDF not found"}))
                return
            page_nums = list(range(10, 10 + n_pages))
            _make_subset_pdf(layered, sample_pdf, page_nums)

            if mode == "native":
                t0 = time.perf_counter()
                extract_ir(sample_pdf, figures=False, ocr=False, device=device)
                total_sec = time.perf_counter() - t0
                per_page = total_sec / len(page_nums)
                print(json.dumps({
                    "mode": mode,
                    "pages": len(page_nums),
                    "import_sec": round(import_sec, 4),
                    "first_call_sec": round(total_sec, 4),
                    "per_page_sec": round(per_page, 4),
                    "peak_rss_mb": _peak_rss_mb(),
                }))
                return

            from pdf.figures import detect_figures

            cache_dir = (
                ROOT / "outputs" / "08_pdf_figure_crops" / layered.stem / "_docling_cache_dpi150"
                if mode == "figures_cached"
                else tmp_path / "uncached_docling"
            )
            src_pages = page_nums if mode == "figures_cached" else list(range(1, len(page_nums) + 1))
            src_pdf = layered if mode == "figures_cached" else sample_pdf

            t_first0 = time.perf_counter()
            detect_figures(src_pdf, [src_pages[0]], device=device, cache_dir=cache_dir)
            first_sec = time.perf_counter() - t_first0

            t_rest0 = time.perf_counter()
            if len(src_pages) > 1:
                detect_figures(src_pdf, src_pages[1:], device=device, cache_dir=cache_dir)
            rest_sec = time.perf_counter() - t_rest0
            per_page = rest_sec / max(1, len(src_pages) - 1)
            print(json.dumps({
                "mode": mode,
                "pages": len(src_pages),
                "import_sec": round(import_sec, 4),
                "first_page_sec": round(first_sec, 4),
                "steady_per_page_sec": round(per_page, 4),
                "peak_rss_mb": _peak_rss_mb(),
            }))
            return

        if mode == "raster":
            if not omni_dir.exists():
                print(json.dumps({"mode": mode, "skipped": "OmniDocBench images not found"}))
                return
            imgs = [omni_dir / name for name in OMNIDOC_SAMPLE_PAGES[:n_pages] if (omni_dir / name).exists()]
            if not imgs:
                print(json.dumps({"mode": mode, "skipped": "No sample images found"}))
                return
            _make_raster_pdf_from_images(imgs, sample_pdf)
            from pdf.raster import read_raster_page

            t_first0 = time.perf_counter()
            read_raster_page(sample_pdf, 1, device=device)
            first_sec = time.perf_counter() - t_first0

            t_rest0 = time.perf_counter()
            for pno in range(2, len(imgs) + 1):
                read_raster_page(sample_pdf, pno, device=device)
            rest_sec = time.perf_counter() - t_rest0
            per_page = rest_sec / max(1, len(imgs) - 1)
            print(json.dumps({
                "mode": mode,
                "pages": len(imgs),
                "import_sec": round(import_sec, 4),
                "first_page_sec": round(first_sec, 4),
                "steady_per_page_sec": round(per_page, 4),
                "peak_rss_mb": _peak_rss_mb(),
            }))
            return


def run_omnidoc_recall(device: str) -> None:
    gt_file = ROOT / "datasets" / "OmniDocBench" / "OmniDocBench.json"
    img_dir = ROOT / "datasets" / "OmniDocBench" / "images"
    if not gt_file.exists() or not img_dir.exists():
        print("SKIP OmniDocBench dataset not present")
        return

    from pdf import extract_ir

    gt_data = json.loads(gt_file.read_text(encoding="utf-8"))
    by_name = {d["page_info"]["image_path"]: d for d in gt_data}

    recalls: list[float] = []
    total_matched = 0
    total_gt = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for idx, name in enumerate(OMNIDOC_SAMPLE_PAGES, start=1):
            img_path = img_dir / name
            if not img_path.exists() or name not in by_name:
                print(f"SKIP missing {name}")
                continue
            gt_raw = "".join(b.get("text", "") for b in by_name[name]["layout_dets"] if b.get("text"))
            gt_chars = Counter("".join(gt_raw.split()))

            pdf_path = tmp_path / f"p{idx:02d}.pdf"
            _make_raster_pdf_from_images([img_path], pdf_path)
            ir = extract_ir(pdf_path, figures=False, ocr=True, device=device)

            ocr_elems = [e for e in ir["elements"] if e["type"] == "text"]
            assert all(e["source"] == "ocr" and isinstance(e["confidence"], float) for e in ocr_elems)
            pred_raw = "".join(e["text"] for e in ocr_elems)
            pred_chars = Counter("".join(pred_raw.split()))

            matched = sum((gt_chars & pred_chars).values())
            gt_n = sum(gt_chars.values())
            recall = matched / gt_n if gt_n > 0 else 1.0
            recalls.append(recall)
            total_matched += matched
            total_gt += gt_n
            print(f"[{idx:02d}/20] {name}: char_recall={recall:.4f} (gt={gt_n}, pred={sum(pred_chars.values())})")

    if recalls:
        mean_r = sum(recalls) / len(recalls)
        pooled_r = total_matched / total_gt if total_gt else 0.0
        sorted_r = sorted(recalls)
        median_r = sorted_r[len(sorted_r) // 2]
        print(
            f"OmniDocBench 20 pages: pooled_char_recall={pooled_r:.4f}, "
            f"mean={mean_r:.4f}, median={median_r:.4f}, min={min(recalls):.4f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark and evaluate the pdf package.")
    ap.add_argument("--mode", default="all", choices=["native", "figures_live", "figures_cached", "raster", "all"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--pages", type=int, default=6)
    ap.add_argument("--omnidoc-recall", action="store_true")
    ap.add_argument("--worker", default="")
    args = ap.parse_args()

    if args.worker:
        _worker(args.worker, args.device, args.pages)
        return

    if args.omnidoc_recall:
        run_omnidoc_recall(args.device)
        return

    modes = ["native", "figures_cached", "figures_live", "raster"] if args.mode == "all" else [args.mode]
    for mode in modes:
        result = _run_isolated_bench(mode, args.device, args.pages)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
