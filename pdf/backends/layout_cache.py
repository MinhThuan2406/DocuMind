from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .base import LayoutBackend, PageLayout, RunStats


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


class CachedLayoutBackend:
    def __init__(self, backend: LayoutBackend, root_dir: Path) -> None:
        self.backend = backend
        self.root_dir = Path(root_dir)
        self.name = backend.name
        self.revision = backend.revision
        self.cache_dpi = getattr(backend, "cache_dpi", 150)

    @property
    def last_stats(self) -> RunStats | None:
        return self.backend.last_stats

    def analyze(self, pdf_path: str | Path, pages: Sequence[int]) -> dict[int, PageLayout]:
        path = Path(pdf_path)
        target_pages = sorted(set(pages))
        cache = LayoutCache(
            self.root_dir,
            backend=self.name,
            revision=self.revision,
            dpi=self.cache_dpi,
            pdf_hash=file_sha256(path),
        )
        out: dict[int, PageLayout] = {}
        missing: list[int] = []
        for pno in target_pages:
            cached = cache.load(pno)
            if cached is not None:
                items, frame = cached
                out[pno] = PageLayout(items=items, frame=frame)
            else:
                missing.append(pno)

        if missing:
            analyzed = self.backend.analyze(path, missing)
            for pno_out, layout in analyzed.items():
                cache.save(pno_out, layout.items, layout.frame)
                if pno_out in missing:
                    out[pno_out] = layout

        return out
