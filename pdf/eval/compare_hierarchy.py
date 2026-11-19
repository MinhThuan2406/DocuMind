from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pdf import extract_ir  # noqa: E402
from pdf.hierarchy import _load_pdf_toc, _norm_title, apply_structure  # noqa: E402

AUTOSAR_PDFS = (
    "AUTOSAR_CP_EXP_VFB.pdf",
    "AUTOSAR_CP_SWS_OS.pdf",
    "AUTOSAR_CP_SWS_ECUStateManager.pdf",
    "AUTOSAR_CP_EXP_LayeredSoftwareArchitecture.pdf",
)

_NUMBERED_PREFIX_RE = re.compile(r"^\d+(?:\.\d+)*\b")


def _score_against_toc(
    outline: list[dict[str, Any]],
    toc: list[tuple[int, str, int]],
) -> dict[str, Any]:
    if not toc:
        return {"toc_len": 0, "matched": 0, "recall": None, "level_acc": None, "extra": len(outline)}

    by_page: dict[int, list[dict[str, Any]]] = {}
    for item in outline:
        by_page.setdefault(int(item.get("page") or 0), []).append(item)

    used_ids: set[str] = set()
    matched = 0
    level_exact = 0
    for toc_lvl, toc_title, toc_page in toc:
        norm_t = _norm_title(toc_title)
        hit: dict[str, Any] | None = None
        for p in (toc_page, toc_page - 1, toc_page + 1):
            for cand in by_page.get(p, ()):
                if cand["id"] in used_ids:
                    continue
                norm_c = _norm_title(cand["title"])
                if norm_c == norm_t or (len(norm_t) >= 8 and (norm_t in norm_c or norm_c in norm_t)):
                    hit = cand
                    break
            if hit is not None:
                break
        if hit is not None:
            used_ids.add(hit["id"])
            matched += 1
            if int(hit["level"]) == toc_lvl:
                level_exact += 1

    extra = [item for item in outline if item["id"] not in used_ids]
    first_toc_page = min(p for _, _, p in toc)
    extra_front = sum(1 for item in extra if int(item.get("page") or 0) < first_toc_page)
    extra_deep_num = sum(
        1
        for item in extra
        if int(item.get("page") or 0) >= first_toc_page
        and _NUMBERED_PREFIX_RE.match(item["title"])
        and int(item["level"]) >= 5
    )
    return {
        "toc_len": len(toc),
        "matched": matched,
        "recall": round(matched / len(toc), 4),
        "level_acc": round(level_exact / matched, 4) if matched else 0.0,
        "extra": len(extra),
        "extra_front": extra_front,
        "extra_deep_num": extra_deep_num,
    }


def _run_heron_sample(pdf_dir: Path, sample_spec: dict[str, list[int]]) -> None:
    import pdfplumber
    import pypdfium2 as pdfium

    from pdf._pdf_io import extract_words_raw
    from pdf.backends import get as get_backend
    from pdf.layout import postprocess_layout

    detector = get_backend("detector", "heron-onnx", device="auto")
    print("\n=== Heron-ONNX visual heading/caption labels (keep_empty_clusters=True, sample pages) ===")
    for name, pages in sample_spec.items():
        path = pdf_dir / name
        if not path.exists():
            continue
        fdoc = pdfium.PdfDocument(str(path))
        with pdfplumber.open(str(path)) as doc:
            try:
                print(f"\n[{name}]")
                for pno in pages:
                    page = doc.pages[pno - 1]
                    zoom = 150.0 / 72.0
                    img = fdoc[pno - 1].render(scale=zoom).to_pil().convert("RGB")
                    iw, ih = img.size
                    raw = detector.detect(img)
                    clusters = postprocess_layout(raw, (float(iw), float(ih)), keep_empty_clusters=True)
                    counts = Counter(c.label for c in clusters)
                    sx, sy = float(page.width) / iw, float(page.height) / ih
                    words = extract_words_raw(page)
                    header_texts = []
                    for c in clusters:
                        if c.label not in ("title", "section_header", "caption"):
                            continue
                        bx0, by0, bx1, by1 = c.bbox[0] * sx, c.bbox[1] * sy, c.bbox[2] * sx, c.bbox[3] * sy
                        txt = " ".join(
                            w["text"]
                            for w in words
                            if bx0 <= (w["x0"] + w["x1"]) * 0.5 <= bx1 and by0 <= (w["top"] + w["bottom"]) * 0.5 <= by1
                        )
                        header_texts.append((c.label, round(c.confidence, 2), txt[:70]))
                    print(f"  p{pno:03d} counts={dict(counts)}")
                    for lbl, conf, txt in header_texts:
                        print(f"    {lbl} ({conf}): {txt!r}")
            finally:
                fdoc.close()
        finally:
            doc.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare current pdf/hierarchy.py vs Docling-style hierarchy on AUTOSAR.")
    ap.add_argument("--pdf-dir", default=str(ROOT / "datasets" / "AUTOSAR"))
    ap.add_argument("--heron-sample", action="store_true", help="Inspect Heron-ONNX section_header/caption on sample pages")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    modes = (
        ("no_toc", lambda base, p: apply_structure(base, use_bookmarks=False)),
        ("with_toc", lambda base, p: apply_structure(base, pdf_path=p, use_bookmarks=True)),
    )

    for name in AUTOSAR_PDFS:
        path = pdf_dir / name
        if not path.exists():
            print(f"SKIP missing {path}")
            continue
        base_ir = extract_ir(path, figures=False, ocr=False, structure=False)
        toc = _load_pdf_toc(path)
        print(f"\n=== {name} (pages={len(base_ir['metadata']['pages'])}, toc={len(toc)}) ===")
        header = f"{'mode':<16} {'out_n':>6} {'toc_rec':>8} {'lvl_acc':>8} {'extra':>6} {'front':>6} {'deep5+':>7} {'ms':>7}  levels"
        print(header)
        print("-" * len(header))
        for mode_name, fn in modes:
            t0 = time.perf_counter()
            res = fn(base_ir, path)
            ms = (time.perf_counter() - t0) * 1000.0
            outline = res["metadata"]["outline"]
            lvls = dict(sorted(Counter(o["level"] for o in outline).items()))
            sc = _score_against_toc(outline, toc)
            rec_s = f"{sc['recall']:.3f}" if sc["recall"] is not None else "  -"
            acc_s = f"{sc['level_acc']:.3f}" if sc["level_acc"] is not None else "  -"
            print(
                f"{mode_name:<16} {len(outline):>6d} {rec_s:>8} {acc_s:>8} "
                f"{sc['extra']:>6d} {sc.get('extra_front', 0):>6d} {sc.get('extra_deep_num', 0):>7d} {ms:>7.1f}  {lvls}"
            )

    if args.heron_sample:
        _run_heron_sample(
            pdf_dir,
            {
                "AUTOSAR_CP_SWS_OS.pdf": [1, 7, 36, 61, 131, 209],
                "AUTOSAR_CP_EXP_LayeredSoftwareArchitecture.pdf": [1, 10, 11, 27, 79],
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
