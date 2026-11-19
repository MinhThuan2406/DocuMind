from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pypdfium2 as pdfium  # noqa: E402

from pdf.backends.rss import peak_rss_mb as _peak_rss_mb  # noqa: E402
from pdf.eval.gt import OMNIDOC_SAMPLE_PAGES, build_raster_pdf  # noqa: E402

_VFB_TABLE_PAGES = [17, 35, 50, 74, 100]


def _make_subset_pdf(src: Path, dst: Path, pages: list[int]) -> None:
    doc = pdfium.PdfDocument(str(src))
    out = pdfium.PdfDocument.new()
    try:
        out.import_pages(doc, pages=[pno - 1 for pno in pages])
        out.save(str(dst))
    finally:
        out.close()
        doc.close()


def _make_mixed_pdf(src: Path, native_pages: list[int], raster_imgs: list[Path], dst: Path, tmp_dir: Path) -> None:
    out = pdfium.PdfDocument.new()
    doc = pdfium.PdfDocument(str(src))
    raster_doc = None
    try:
        out.import_pages(doc, pages=[pno - 1 for pno in native_pages])
        if raster_imgs:
            raster_pdf = tmp_dir / "_raster_part.pdf"
            build_raster_pdf(raster_imgs, raster_pdf)
            raster_doc = pdfium.PdfDocument(str(raster_pdf))
            out.import_pages(raster_doc)
        out.save(str(dst))
    finally:
        if raster_doc is not None:
            raster_doc.close()
        doc.close()
        out.close()


def _pick_autosar_pages(src: Path, n_pages: int, *, prefer_tables: bool) -> list[int]:
    if prefer_tables and src.name == "AUTOSAR_CP_EXP_VFB.pdf":
        base = list(_VFB_TABLE_PAGES)
        if n_pages <= len(base):
            return base[:n_pages]
        extra = [p for p in range(10, 10 + n_pages * 2) if p not in base]
        return (base + extra)[:n_pages]
    return list(range(10, 10 + n_pages))


def _run_isolated_bench(
    mode: str,
    device: str,
    pages: int,
    reader: str = "composed",
) -> dict[str, object]:
    if mode == "figures_cached":
        subprocess.run(
            [sys.executable, "-B", __file__, "--worker", "figures_warm", "--device", device, "--pages", str(pages)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
    cmd = [
        sys.executable,
        "-B",
        __file__,
        "--worker",
        mode,
        "--device",
        device,
        "--pages",
        str(pages),
        "--reader",
        reader,
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _worker(mode: str, device: str, n_pages: int, reader: str = "composed") -> None:
    layered = ROOT / "datasets" / "AUTOSAR" / "AUTOSAR_CP_EXP_LayeredSoftwareArchitecture.pdf"
    vfb = ROOT / "datasets" / "AUTOSAR" / "AUTOSAR_CP_EXP_VFB.pdf"
    omni_dir = ROOT / "datasets" / "OmniDocBench" / "images"

    t_import0 = time.perf_counter()
    from pdf import extract_ir

    import_sec = time.perf_counter() - t_import0

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        sample_pdf = tmp_path / "sample.pdf"
        if mode in ("native", "figures_live", "figures_cached", "figures_warm"):
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
                ROOT / "outputs" / "08_pdf_figure_crops" / layered.stem / "_layout_cache"
                if mode in ("figures_cached", "figures_warm")
                else tmp_path / "uncached_layout"
            )
            src_pages = page_nums if mode in ("figures_cached", "figures_warm") else list(range(1, len(page_nums) + 1))
            src_pdf = layered if mode in ("figures_cached", "figures_warm") else sample_pdf

            if mode == "figures_warm":
                detect_figures(src_pdf, src_pages, device=device, cache_dir=cache_dir)
                print(json.dumps({"mode": mode, "warmed": len(src_pages)}))
                return

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

        if mode in ("tables_live", "balanced_e2e"):
            src = vfb if vfb.exists() else layered
            if not src.exists():
                print(json.dumps({"mode": mode, "skipped": "AUTOSAR PDF not found"}))
                return
            if mode == "tables_live":
                page_nums = _pick_autosar_pages(src, n_pages, prefer_tables=True)
                _make_subset_pdf(src, sample_pdf, page_nums)
                t0 = time.perf_counter()
                ir = extract_ir(
                    sample_pdf,
                    figures=True,
                    ocr=True,
                    tables=True,
                    structure=True,
                    reader=reader,
                    device=device,
                )
                total_sec = time.perf_counter() - t0
                print(json.dumps({
                    "mode": mode,
                    "pages": len(page_nums),
                    "tables_extracted": len(ir["tables"]),
                    "elements": len(ir["elements"]),
                    "import_sec": round(import_sec, 4),
                    "total_sec": round(total_sec, 4),
                    "per_page_sec": round(total_sec / max(1, len(page_nums)), 4),
                    "peak_rss_mb": _peak_rss_mb(),
                }))
                return

            raster_imgs = (
                [omni_dir / name for name in OMNIDOC_SAMPLE_PAGES[:1] if (omni_dir / name).exists()]
                if n_pages >= 2 and omni_dir.exists()
                else []
            )
            native_n = max(1, n_pages - len(raster_imgs))
            native_pages = _pick_autosar_pages(src, native_n, prefer_tables=True)
            _make_mixed_pdf(src, native_pages, raster_imgs, sample_pdf, tmp_path)
            total_pages = len(native_pages) + len(raster_imgs)
            t0 = time.perf_counter()
            ir = extract_ir(
                sample_pdf,
                figures=True,
                ocr=True,
                tables=True,
                structure=True,
                reader=reader,
                device=device,
            )
            total_sec = time.perf_counter() - t0
            print(json.dumps({
                "mode": mode,
                "pages": total_pages,
                "native_pages": len(native_pages),
                "raster_pages": len(raster_imgs),
                "tables_extracted": len(ir["tables"]),
                "elements": len(ir["elements"]),
                "outline_len": len(ir["metadata"].get("outline", ())),
                "import_sec": round(import_sec, 4),
                "total_sec": round(total_sec, 4),
                "per_page_sec": round(total_sec / max(1, total_pages), 4),
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
            build_raster_pdf(imgs, sample_pdf)
            from pdf.backends import get as get_backend
            from pdf.raster import read_raster_page, read_raster_pages

            page_reader = get_backend("reader", reader, device=device)
            if getattr(page_reader, "isolated", False):
                t0 = time.perf_counter()
                read_raster_pages(sample_pdf, list(range(1, len(imgs) + 1)), reader=page_reader)
                wall_sec = round(time.perf_counter() - t0, 4)
                stats = page_reader.last_stats
                n_proc = stats.pages if stats and stats.pages else len(imgs)
                infer_sec = stats.infer_sec if stats else 0.0
                print(json.dumps({
                    "mode": mode,
                    "reader": reader,
                    "pages": len(imgs),
                    "import_sec": round(import_sec, 4),
                    "load_sec": stats.load_sec if stats else 0.0,
                    "inference_sec": infer_sec,
                    "steady_per_page_sec": round(infer_sec / max(1, n_proc), 4),
                    "wall_sec": wall_sec,
                    "child_peak_rss_mb": stats.peak_rss_mb if stats else 0.0,
                    "peak_rss_mb": _peak_rss_mb(),
                }))
                return

            t_first0 = time.perf_counter()
            read_raster_page(sample_pdf, 1, reader=page_reader)
            first_sec = time.perf_counter() - t_first0

            t_rest0 = time.perf_counter()
            for pno in range(2, len(imgs) + 1):
                read_raster_page(sample_pdf, pno, reader=page_reader)
            rest_sec = time.perf_counter() - t_rest0
            per_page = rest_sec / max(1, len(imgs) - 1)
            print(json.dumps({
                "mode": mode,
                "reader": reader,
                "pages": len(imgs),
                "import_sec": round(import_sec, 4),
                "first_page_sec": round(first_sec, 4),
                "steady_per_page_sec": round(per_page, 4),
                "peak_rss_mb": _peak_rss_mb(),
            }))
            return


def main() -> None:
    from pdf.backends import registered

    ap = argparse.ArgumentParser(description="Timing benchmark for the pdf package.")
    ap.add_argument(
        "--mode",
        default="all",
        choices=["native", "figures_live", "figures_cached", "raster", "tables_live", "balanced_e2e", "all"],
    )
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--reader", default="composed", choices=registered("reader"))
    ap.add_argument("--pages", type=int, default=6)
    ap.add_argument("--worker", default="")
    args = ap.parse_args()

    if args.worker:
        _worker(args.worker, args.device, args.pages, args.reader)
        return

    modes = (
        ["native", "figures_cached", "figures_live", "raster", "tables_live", "balanced_e2e"]
        if args.mode == "all"
        else [args.mode]
    )
    for mode in modes:
        result = _run_isolated_bench(mode, args.device, args.pages, args.reader)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
