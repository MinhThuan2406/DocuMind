from __future__ import annotations

import inspect
import io
from pathlib import Path
from typing import Any

from PIL import Image

from .base import RunStats

UNIMERNET_ONNX_REPO_ID = "Cooper114/unimernet-onnx"
UNIMERNET_DEFAULT_TIER = "small"
_UNIMERNET_SIZE = (672, 192)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_DEFAULT_CONFIDENCE = 0.95
_WEIGHT_FILES = {
    "image_resizer_path": "image_resizer.onnx",
    "encoder_path": "encoder.onnx",
    "decoder_path": "decoder.onnx",
    "tokenizer_json": "tokenizer.json",
}
_UNIMERNET_FILES = (
    "encoder_model_quantized.onnx",
    "decoder_model_quantized.onnx",
    "decoder_with_past_model_quantized.onnx",
    "tokenizer.json",
)


def _pil_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _resolve_unimernet_dir(model_path: str | Path | None, repo_id: str, tier: str) -> Path:
    if model_path is not None:
        base = Path(model_path)
        if not base.exists():
            raise FileNotFoundError(f"formula model_path does not exist: {base}")
        nested = base / tier
        return nested if nested.is_dir() else base

    from huggingface_hub import snapshot_download

    patterns = [f"{tier}/{name}" for name in _UNIMERNET_FILES]
    try:
        root = Path(snapshot_download(repo_id=repo_id, allow_patterns=patterns, local_files_only=True))
    except Exception:
        root = Path(snapshot_download(repo_id=repo_id, allow_patterns=patterns))
    return root / tier


def _preprocess_unimernet(crop: Image.Image) -> Any:
    import numpy as np

    target_w, target_h = _UNIMERNET_SIZE
    rgb = crop.convert("RGB")
    w = max(1, rgb.width)
    h = max(1, rgb.height)
    scale = min(target_w / w, target_h / h)
    new_w = max(1, min(target_w, round(w * scale)))
    new_h = max(1, min(target_h, round(h * scale)))
    resized = rgb.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    mean = np.array(_IMAGENET_MEAN, dtype=np.float32)
    std = np.array(_IMAGENET_STD, dtype=np.float32)
    norm = (arr - mean) / std
    return norm.transpose(2, 0, 1)[None, ...].astype(np.float32)


class UnimernetOnnxBackend:
    # Lazy ONNX Runtime wrapper around Cooper114/unimernet-onnx (CPU-first for int8 KV-cache speed).
    name = "unimernet-onnx"
    source = "unimernet_onnx"

    def __init__(
        self,
        *,
        device: str = "cpu",
        repo_id: str = UNIMERNET_ONNX_REPO_ID,
        tier: str = UNIMERNET_DEFAULT_TIER,
        model_path: str | Path | None = None,
        max_tokens: int = 512,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
    ) -> None:
        self.device = device
        self.repo_id = repo_id
        self.tier = tier
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self._encoder: Any = None
        self._decoder: Any = None
        self._decoder_with_past: Any = None
        self._tokenizer: Any = None
        self.last_stats: RunStats | None = None

    def ensure_loaded(self) -> bool:
        if self._encoder is not None and self._decoder is not None and self._decoder_with_past is not None:
            return False
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = _resolve_unimernet_dir(self.model_path, self.repo_id, self.tier)
        if self.device in ("cuda", "gpu"):
            try:
                ort.preload_dlls()
            except Exception:
                pass
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self._encoder = ort.InferenceSession(str(model_dir / "encoder_model_quantized.onnx"), providers=providers)
        self._decoder = ort.InferenceSession(str(model_dir / "decoder_model_quantized.onnx"), providers=providers)
        self._decoder_with_past = ort.InferenceSession(
            str(model_dir / "decoder_with_past_model_quantized.onnx"),
            providers=providers,
        )
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        return True

    def extract_formula(self, crop: Image.Image) -> tuple[str, float]:
        import numpy as np

        self.ensure_loaded()
        pixel_values = _preprocess_unimernet(crop)
        enc_in_name = self._encoder.get_inputs()[0].name
        encoder_hidden_states = self._encoder.run(None, {enc_in_name: pixel_values})[0]

        dec_in_names = {inp.name for inp in self._decoder.get_inputs()}
        dec_out_names = [out.name for out in self._decoder.get_outputs()]
        dec_inputs: dict[str, Any] = {"input_ids": np.array([[self.bos_token_id]], dtype=np.int64)}
        if "encoder_hidden_states" in dec_in_names:
            dec_inputs["encoder_hidden_states"] = encoder_hidden_states
        if "encoder_attention_mask" in dec_in_names:
            dec_inputs["encoder_attention_mask"] = np.ones(encoder_hidden_states.shape[:2], dtype=np.int64)

        dec_outs = self._decoder.run(None, dec_inputs)
        logits_idx = dec_out_names.index("logits") if "logits" in dec_out_names else 0
        next_token = int(np.argmax(dec_outs[logits_idx][0, -1, :]))

        past_kv: dict[str, Any] = {}
        for name, val in zip(dec_out_names, dec_outs):
            if name.startswith("present."):
                past_kv["past_key_values." + name[len("present.") :]] = val

        tokens: list[int] = []
        past_in_names = {inp.name for inp in self._decoder_with_past.get_inputs()}
        past_out_names = [out.name for out in self._decoder_with_past.get_outputs()]
        past_logits_idx = past_out_names.index("logits") if "logits" in past_out_names else 0

        for _ in range(self.max_tokens):
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            step_inputs: dict[str, Any] = {"input_ids": np.array([[next_token]], dtype=np.int64)}
            if "encoder_hidden_states" in past_in_names:
                step_inputs["encoder_hidden_states"] = encoder_hidden_states
            if "encoder_attention_mask" in past_in_names:
                step_inputs["encoder_attention_mask"] = np.ones(encoder_hidden_states.shape[:2], dtype=np.int64)
            for k, v in past_kv.items():
                if k in past_in_names:
                    step_inputs[k] = v

            step_outs = self._decoder_with_past.run(None, step_inputs)
            next_token = int(np.argmax(step_outs[past_logits_idx][0, -1, :]))
            for name, val in zip(past_out_names, step_outs):
                if name.startswith("present.") and ".encoder." not in name:
                    past_kv["past_key_values." + name[len("present.") :]] = val

        latex = str(self._tokenizer.decode(tokens, skip_special_tokens=True)).strip()
        return latex, (_DEFAULT_CONFIDENCE if latex else 0.0)


class RapidLatexOcrBackend:
    # Lazy wrapper around rapid_latex_ocr.LaTeXOCR (pure ONNX Runtime).
    name = "rapid-latex-ocr"
    source = "rapid_latex_ocr"

    def __init__(
        self,
        *,
        device: str = "auto",
        model_path: str | Path | None = None,
        image_resizer_path: str | Path | None = None,
        encoder_path: str | Path | None = None,
        decoder_path: str | Path | None = None,
        tokenizer_json: str | Path | None = None,
        model: Any = None,
    ) -> None:
        self.device = device
        self.model_path = model_path
        self.image_resizer_path = image_resizer_path
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.tokenizer_json = tokenizer_json
        self._model: Any = model
        self.last_stats: RunStats | None = None

    def _build_init_kwargs(self, cls: Any) -> dict[str, Any]:
        weights: dict[str, str] = {}
        if self.model_path is not None:
            folder = Path(self.model_path)
            if not folder.exists():
                raise FileNotFoundError(f"formula model_path does not exist: {folder}")
            if folder.is_dir():
                for arg, fname in _WEIGHT_FILES.items():
                    cand = folder / fname
                    if cand.is_file():
                        weights[arg] = str(cand)
        for arg in _WEIGHT_FILES:
            val = getattr(self, arg)
            if val is not None:
                weights[arg] = str(val)

        try:
            params = inspect.signature(cls).parameters
        except (TypeError, ValueError):
            return weights
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if not has_var_kw:
            weights = {k: v for k, v in weights.items() if k in params}
        if self.device in ("cuda", "gpu") and ("use_cuda" in params or has_var_kw):
            weights["use_cuda"] = True
        return weights

    def ensure_loaded(self) -> bool:
        if self._model is not None:
            return False
        try:
            import rapid_latex_ocr
        except ImportError as exc:
            raise ImportError(
                "rapid_latex_ocr is required for rapid-latex-ocr formula extraction; "
                "install rapid-latex-ocr and onnxruntime"
            ) from exc

        cls = getattr(rapid_latex_ocr, "LaTeXOCR", None) or getattr(rapid_latex_ocr, "LatexOCR", None)
        if cls is None:
            raise RuntimeError("rapid_latex_ocr does not expose LaTeXOCR or LatexOCR")

        kwargs = self._build_init_kwargs(cls)
        self._model = cls(**kwargs)
        return True

    def extract_formula(self, crop: Image.Image) -> tuple[str, float]:
        self.ensure_loaded()
        rgb = crop.convert("RGB")
        mod_name = type(self._model).__module__
        if mod_name.startswith("rapid_latex_ocr"):
            raw = self._model(_pil_to_png_bytes(rgb))
        else:
            try:
                raw = self._model(rgb)
            except Exception as exc:
                if isinstance(exc, (TypeError, ValueError)) or "LoadImage" in type(exc).__name__:
                    raw = self._model(_pil_to_png_bytes(rgb))
                else:
                    raise

        if isinstance(raw, tuple):
            latex = str(raw[0] if raw else "").strip()
        elif isinstance(raw, str):
            latex = raw.strip()
        elif raw is None:
            latex = ""
        else:
            latex = str(raw).strip()

        return latex, (_DEFAULT_CONFIDENCE if latex else 0.0)
