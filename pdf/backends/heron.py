from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from ..layout.detector import clamp_and_rescale
from ..layout.types import LayoutItem

HERON_REPO_ID = "docling-project/docling-layout-heron"
HERON_REVISION = "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8"
DEFAULT_SCORE_THRESHOLD = 0.3


def _resolve_device(torch_mod: Any, device: str) -> Any:
    if device == "auto":
        return torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
    return torch_mod.device(device)


def _resolve_model_folder(model_path: str | Path | None, repo_id: str, revision: str) -> Path:
    if model_path is not None:
        return Path(model_path)
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(repo_id=repo_id, revision=revision, local_files_only=True))
    except Exception:
        return Path(snapshot_download(repo_id=repo_id, revision=revision))


def _load_id2label(model_folder: Path, model: Any) -> dict[int, str]:
    cfg_path = model_folder / "config.json"
    if cfg_path.exists():
        raw = json.loads(cfg_path.read_text(encoding="utf-8")).get("id2label", {})
        if raw:
            return {int(k): str(v).lower() for k, v in raw.items()}
    return {int(k): str(v).lower() for k, v in getattr(model.config, "id2label", {}).items()}


class HeronDetector:
    # Ported from docling transformers_engine.py + layout_object_detection_model.py (MIT).
    def __init__(
        self,
        *,
        device: str = "auto",
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        repo_id: str = HERON_REPO_ID,
        revision: str = HERON_REVISION,
        model_path: str | Path | None = None,
    ) -> None:
        self.device_name = device
        self.score_threshold = score_threshold
        self.repo_id = repo_id
        self.revision = revision
        self.model_path = model_path
        self._processor: Any = None
        self._model: Any = None
        self._device: Any = None
        self._id2label: dict[int, str] = {}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        folder = _resolve_model_folder(self.model_path, self.repo_id, self.revision)
        self._device = _resolve_device(torch, self.device_name)
        self._processor = AutoImageProcessor.from_pretrained(str(folder))
        self._model = AutoModelForObjectDetection.from_pretrained(str(folder))
        self._model.to(self._device)
        self._model.eval()
        self._id2label = _load_id2label(folder, self._model)

    def detect(self, image: Image.Image) -> list[LayoutItem]:
        import torch

        self._ensure_loaded()
        rgb = image.convert("RGB")
        inputs = self._processor(images=[rgb], return_tensors="pt").to(self._device, self._model.dtype)
        target_sizes = torch.tensor([[rgb.height, rgb.width]], device=self._device)

        with torch.inference_mode():
            outputs = self._model(**inputs)

        results = self._processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.score_threshold,
        )
        res = results[0]
        raw_items: list[tuple[str, tuple[float, float, float, float], float]] = []
        for label_id, score, box in zip(
            res["labels"].tolist(),
            res["scores"].tolist(),
            res["boxes"].tolist(),
        ):
            label = self._id2label.get(int(label_id))
            if label is None:
                continue
            raw_items.append((label, (float(box[0]), float(box[1]), float(box[2]), float(box[3])), float(score)))

        size = (float(rgb.width), float(rgb.height))
        return clamp_and_rescale(raw_items, size, size)
