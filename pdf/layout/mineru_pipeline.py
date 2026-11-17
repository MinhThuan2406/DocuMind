from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..models import FigureBox, RasterPage, TextBlock
from .detector import clamp_and_rescale
from .postprocess import clusters_to_items, postprocess_layout
from .types import LayoutItem

MINERU_PIPELINE_BACKEND = "mineru-pipeline"
MINERU_PIPELINE_REVISION = "4.2.3"
_FIGURE_PATH_LABELS = frozenset({"picture", "table", "formula", "document_index"})
_RASTER_BOX_CONFIDENCE = 0.95
_SKIP_VISUAL_CHILD_TYPES = frozenset({"image_body", "chart_body"})

MINERU_LABEL_MAP: dict[str, str] = {
    "image": "picture",
    "chart": "picture",
    "table": "table",
    "equation": "formula",
    "interline_equation": "formula",
    "isolated_equation": "formula",
    "display_formula": "formula",
    "index": "document_index",
    "text": "text",
    "paragraph_title": "section_header",
    "doc_title": "title",
    "list": "list_item",
    "code": "code",
    "header": "page_header",
    "footer": "page_footer",
    "page_number": "page_footer",
    "page_footnote": "footnote",
    "aside_text": "text",
    "ref_text": "list_item",
    "abstract": "text",
    "algorithm": "code",
}

MINERU_DROPPED_LABELS = frozenset({"inline_equation", "inline_formula"})

_CHILD_SCRIPT = r"""
import ctypes
import ctypes.wintypes as wintypes
import json
import os
import sys
import time
from pathlib import Path


def peak_rss_mb():
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


def main():
    pdf_path = Path(sys.argv[1])
    out_json = Path(sys.argv[2])
    strip_content = sys.argv[3] == "1" if len(sys.argv) > 3 else True
    ocr_mode = sys.argv[4] if len(sys.argv) > 4 else "auto"

    t0 = time.perf_counter()
    from mineru.parser import parse
    t_import = time.perf_counter() - t0

    t1 = time.perf_counter()
    res = parse(pdf_path, tier="basic", ocr_mode=ocr_mode)
    t_parse = time.perf_counter() - t1

    data = res.middle_json.model_dump(mode="json")
    for pg in data.get("pages", []):
        for blk in pg.get("blocks", []):
            blk.pop("image_base64", None)
            if strip_content:
                blk.pop("content", None)
    out_json.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    metrics = {
        "import_sec": round(t_import, 4),
        "parse_sec": round(t_parse, 4),
        "total_sec": round(time.perf_counter() - t0, 4),
        "pages_processed": len(data.get("pages", [])),
        "peak_rss_mb": peak_rss_mb(),
    }
    print("MINERU_METRICS=" + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
"""


def find_mineru_python(python_exe: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if python_exe is not None:
        candidates.append(Path(python_exe))
    env_py = os.getenv("DOCUMIND_MINERU_PYTHON")
    if env_py:
        candidates.append(Path(env_py))
    default_dir = Path.home() / ".venvs" / "mineru-4.2.3"
    candidates.append(default_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        "MinerU Python executable not found; set DOCUMIND_MINERU_PYTHON or install in ~/.venvs/mineru-4.2.3"
    )


def run_mineru_pdf(
    pdf_path: str | Path,
    *,
    device: str = "auto",
    python_exe: str | Path | None = None,
    strip_content: bool = True,
    ocr_mode: str = "auto",
) -> tuple[dict[str, Any], dict[str, Any]]:
    py = find_mineru_python(python_exe)
    env = os.environ.copy()
    default_models = Path.home() / ".cache" / "mineru_models"
    if "MINERU_MODEL_BASE_DIR" not in env and default_models.is_dir():
        env["MINERU_MODEL_BASE_DIR"] = str(default_models)
    if "MINERU_MODEL_SOURCE" not in env and default_models.is_dir():
        env["MINERU_MODEL_SOURCE"] = "local"
    if device == "cpu":
        env["MINERU_MODEL_ONNX_DEVICE"] = "cpu"
    elif device in ("auto", "cuda", "gpu"):
        env["MINERU_MODEL_ONNX_DEVICE"] = "gpu"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        script_path = tmp_dir / "run_mineru.py"
        out_path = tmp_dir / "middle.json"
        script_path.write_text(_CHILD_SCRIPT, encoding="utf-8")

        t0 = time.perf_counter()
        proc = subprocess.run(
            [
                str(py),
                "-B",
                str(script_path),
                str(Path(pdf_path).resolve()),
                str(out_path),
                "1" if strip_content else "0",
                ocr_mode,
            ],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        wall_sec = round(time.perf_counter() - t0, 4)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout)[-2000:]
            raise RuntimeError(f"MinerU subprocess failed (exit {proc.returncode}):\n{tail}")
        middle_json = json.loads(out_path.read_text(encoding="utf-8"))

    metrics: dict[str, Any] = {"wall_sec": wall_sec}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("MINERU_METRICS="):
            metrics.update(json.loads(line.removeprefix("MINERU_METRICS=")))
            break
    combined_logs = f"{proc.stdout}\n{proc.stderr}"
    onnx_init_sec = round(sum(float(m) for m in re.findall(r"\binit_s=([0-9.]+)", combined_logs)), 4)
    metrics["onnx_init_sec"] = onnx_init_sec
    if "import_sec" in metrics and "parse_sec" in metrics:
        load_sec = float(metrics["import_sec"]) + onnx_init_sec
        infer_sec = max(0.0, float(metrics["parse_sec"]) - onnx_init_sec)
        n_pages = max(1, int(metrics.get("pages_processed") or 1))
        metrics["load_sec"] = round(load_sec, 4)
        metrics["inference_sec"] = round(infer_sec, 4)
        metrics["steady_per_page_sec"] = round(infer_sec / n_pages, 4)
    return middle_json, metrics


def _boxes_differ(before: list[LayoutItem], after: list[dict[str, object]]) -> tuple[int, int]:
    dropped = max(0, len(before) - len(after))
    changed = 0
    if len(before) == len(after):
        b_sorted = sorted(before, key=lambda it: (it.bbox[1], it.bbox[0], it.label))
        for b, a in zip(b_sorted, after):
            ab = a["bbox"]
            assert isinstance(ab, list)
            if b.label != a["label"] or max(abs(x - float(y)) for x, y in zip(b.bbox, ab)) > 1e-3:
                changed += 1
    return dropped, changed


def mineru_middle_json_to_cache_pages(
    middle_json: Mapping[str, Any],
    page_sizes: Mapping[int, tuple[float, float]],
) -> tuple[dict[int, tuple[list[dict[str, Any]], tuple[float, float]]], Counter[str], dict[int, tuple[int, int]]]:
    # Convert 0-based normalized [0..1] blocks to PDF points and run Heron's exact postprocess chain.
    pages_out: dict[int, tuple[list[dict[str, Any]], tuple[float, float]]] = {}
    unmapped: Counter[str] = Counter()
    post_stats: dict[int, tuple[int, int]] = {}

    for pg in middle_json.get("pages", ()):
        page_no = int(pg["page_idx"]) + 1
        if page_no not in page_sizes:
            continue
        pw, ph = page_sizes[page_no]
        page_size = (float(pw), float(ph))
        raw_items: list[tuple[str, tuple[float, float, float, float], float]] = []

        for blk in pg.get("blocks", ()):
            raw_type = str(blk.get("type", ""))
            if raw_type in MINERU_DROPPED_LABELS:
                continue
            mapped = MINERU_LABEL_MAP.get(raw_type)
            if mapped is None:
                unmapped[raw_type] += 1
                continue
            bbox = blk.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            pt_bbox = (
                float(bbox[0]) * pw,
                float(bbox[1]) * ph,
                float(bbox[2]) * pw,
                float(bbox[3]) * ph,
            )
            score = float(blk.get("score", 1.0))
            raw_items.append((mapped, pt_bbox, score))

        scaled = clamp_and_rescale(raw_items, page_size, page_size)
        before_fig = [it for it in scaled if it.label in _FIGURE_PATH_LABELS]
        clusters = postprocess_layout(scaled, page_size, keep_empty_clusters=False)
        items = clusters_to_items(clusters)
        dropped, changed = _boxes_differ(before_fig, items)
        if dropped or changed:
            post_stats[page_no] = (dropped, changed)
        pages_out[page_no] = (items, page_size)

    for page_no, size in page_sizes.items():
        if page_no not in pages_out:
            pages_out[page_no] = ([], (float(size[0]), float(size[1])))

    return pages_out, unmapped, post_stats


def _scale_mineru_bbox(bbox: Sequence[Any], pw: float, ph: float) -> tuple[float, float, float, float] | None:
    if len(bbox) != 4:
        return None
    x0 = max(0.0, min(pw, float(bbox[0]) * pw))
    y0 = max(0.0, min(ph, float(bbox[1]) * ph))
    x1 = max(0.0, min(pw, float(bbox[2]) * pw))
    y1 = max(0.0, min(ph, float(bbox[3]) * ph))
    if x1 <= x0 or y1 <= y0:
        return None
    return (round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3))


def _extract_block_text(node: Any) -> str:
    if isinstance(node, str):
        return node.strip()
    if isinstance(node, Mapping):
        if node.get("type") in _SKIP_VISUAL_CHILD_TYPES:
            return ""
        if "content" in node:
            return _extract_block_text(node["content"])
        return ""
    if isinstance(node, list):
        parts: list[str] = []
        for item in node:
            part = _extract_block_text(item)
            if part:
                parts.append(part)
        return " ".join(parts).strip()
    return ""


def _child_label(child_type: str, default_label: str) -> str:
    if child_type.endswith("_caption"):
        return "caption"
    if child_type.endswith("_footnote"):
        return "footnote"
    return default_label


def mineru_middle_json_to_raster_pages(
    middle_json: Mapping[str, Any],
    page_sizes: Mapping[int, tuple[float, float]],
    *,
    target_pages: Sequence[int] | None = None,
) -> dict[int, RasterPage]:
    wanted = set(target_pages) if target_pages is not None else set(page_sizes)
    out: dict[int, RasterPage] = {}

    for pg in middle_json.get("pages", ()):
        page_no = int(pg["page_idx"]) + 1
        if page_no not in wanted or page_no not in page_sizes:
            continue
        pw, ph = float(page_sizes[page_no][0]), float(page_sizes[page_no][1])
        blocks: list[TextBlock] = []
        figures: list[FigureBox] = []

        for blk in pg.get("blocks", ()):
            raw_type = str(blk.get("type", ""))
            bbox_raw = blk.get("bbox")
            if not bbox_raw:
                continue
            pt_bbox = _scale_mineru_bbox(bbox_raw, pw, ph)
            if pt_bbox is None:
                continue
            conf = round(float(blk.get("score", _RASTER_BOX_CONFIDENCE)), 4)

            if raw_type in ("image", "chart", "table"):
                fig_kind = "table" if raw_type == "table" else "figure"
                fig_src = "layout_table" if raw_type == "table" else "layout_picture"
                figures.append(
                    FigureBox(
                        bbox=pt_bbox,
                        kind=fig_kind,
                        source=fig_src,
                        confidence=_RASTER_BOX_CONFIDENCE,
                    )
                )
                content = blk.get("content")
                if isinstance(content, list):
                    for child in content:
                        if not isinstance(child, Mapping):
                            continue
                        text = _extract_block_text(child)
                        if not text:
                            continue
                        lbl = _child_label(str(child.get("type", "")), raw_type)
                        blocks.append(
                            TextBlock(
                                text=text,
                                bbox=pt_bbox,
                                font_size=0.0,
                                bold=False,
                                source="ocr",
                                confidence=conf,
                                words=(),
                                label=lbl,
                            )
                        )
                continue

            text = _extract_block_text(blk.get("content"))
            if not text:
                continue
            mapped_label = MINERU_LABEL_MAP.get(raw_type, raw_type)
            blocks.append(
                TextBlock(
                    text=text,
                    bbox=pt_bbox,
                    font_size=0.0,
                    bold=False,
                    source="ocr",
                    confidence=conf,
                    words=(),
                    label=mapped_label,
                )
            )

        out[page_no] = RasterPage(
            blocks=tuple(blocks),
            figures=tuple(figures),
            reading_order=tuple(range(len(blocks))),
        )

    for page_no in wanted:
        if page_no not in out:
            out[page_no] = RasterPage(blocks=(), figures=(), reading_order=())
    return out
