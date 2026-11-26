from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.models.ir import UnifiedDocumentIR


class BaseDocumentAdapter(ABC):
    @property
    @abstractmethod
    def supported_extensions(self) -> tuple[str, ...]:
        pass

    def can_handle(self, file_path: str | Path) -> bool:
        ext = Path(file_path).suffix.lower()
        return ext in self.supported_extensions

    @abstractmethod
    def parse(self, file_path: str | Path) -> UnifiedDocumentIR:
        pass
