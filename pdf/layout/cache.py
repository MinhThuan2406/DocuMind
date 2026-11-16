from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class LayoutCache:
    def __init__(
        self,
        root_dir: Path,
        *,
        backend: str,
        revision: str,
        dpi: int,
        pdf_hash: str,
    ) -> None:
        rev_tag = revision[:12] if revision else "local"
        self.dir = root_dir / f"{backend}_{rev_tag}_dpi{dpi}_{pdf_hash[:16]}"

    def path_for(self, page: int) -> Path:
        return self.dir / f"p{page:03d}.json"

    def load(self, page: int) -> tuple[list[dict[str, Any]], tuple[float, float]] | None:
        p = self.path_for(page)
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        return data["items"], (float(data["page_size"][0]), float(data["page_size"][1]))

    def save(self, page: int, items: list[dict[str, Any]], page_size: tuple[float, float]) -> None:
        p = self.path_for(page)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                json.dumps({"items": items, "page_size": list(page_size)}, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass
