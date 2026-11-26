from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def main() -> None:
    from rss import peak_rss_mb

    # Drop pdf/backends from sys.path after importing rss so mineru.py does not shadow the mineru package.
    backend_dir = Path(__file__).resolve().parent
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != backend_dir]

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
    print("SUBPROCESS_METRICS=" + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
