from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from ..layout.detector import clamp_and_rescale
from ..layout.types import LayoutItem

HERON_ONNX_REPO_ID = "docling-project/docling-layout-heron-onnx"
HERON_ONNX_REVISION = "40bde044036bb181c130ddf6c51792187268748f"
DEFAULT_SCORE_THRESHOLD = 0.3
_INPUT_SIZE = (640, 640)


def _resolve_model_folder(model_path: str | Path | None, repo_id: str, revision: str) -> Path:
    if model_path is not None:
        return Path(model_path)
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(repo_id=repo_id, revision=revision, local_files_only=True))
    except Exception:
        return Path(snapshot_download(repo_id=repo_id, revision=revision))


def _load_id2label(model_folder: Path) -> dict[int, str]:
    cfg_path = model_folder / "config.json"
    raw = json.loads(cfg_path.read_text(encoding="utf-8")).get("id2label", {})
    return {int(k): str(v).lower() for k, v in raw.items()}


class HeronOnnxDetector:
    def __init__(
        self,
        *,
        device: str = "auto",
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        repo_id: str = HERON_ONNX_REPO_ID,
        revision: str = HERON_ONNX_REVISION,
        model_path: str | Path | None = None,
    ) -> None:
        self.device_name = device
        self.score_threshold = score_threshold
        self.repo_id = repo_id
        self.revision = revision
        self.model_path = model_path
        self._session: Any = None
        self._id2label: dict[int, str] = {}

    def ensure_loaded(self) -> bool:
        if self._session is not None:
            return False
        import onnxruntime as ort

        folder = _resolve_model_folder(self.model_path, self.repo_id, self.revision)
        if self.device_name == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            try:
                ort.preload_dlls()
            except Exception:
                pass
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(folder / "model.onnx"), providers=providers)
        self._id2label = _load_id2label(folder)
        return True

    def detect(self, image: Image.Image) -> list[LayoutItem]:
        import numpy as np

        self.ensure_loaded()
        rgb = image.convert("RGB")
        resized = rgb.resize(_INPUT_SIZE, resample=Image.Resampling.BILINEAR)
        tensor = np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1)[None]
        target_sizes = np.array([[rgb.width, rgb.height]], dtype=np.int64)

        labels, boxes, scores = self._session.run(
            None,
            {"images": tensor, "orig_target_sizes": target_sizes},
        )
        raw_items: list[tuple[str, tuple[float, float, float, float], float]] = []
        for label_id, score, box in zip(labels[0].tolist(), scores[0].tolist(), boxes[0].tolist()):
            if float(score) < self.score_threshold:
                continue
            label = self._id2label.get(int(label_id))
            if label is None:
                continue
            raw_items.append((label, (float(box[0]), float(box[1]), float(box[2]), float(box[3])), float(score)))

        size = (float(rgb.width), float(rgb.height))
        return clamp_and_rescale(raw_items, size, size)
