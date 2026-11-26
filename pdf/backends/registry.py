from __future__ import annotations

from collections.abc import Callable
from typing import Any

VALID_KINDS = ("layout", "detector", "ocr", "reader", "grouper", "table", "diagram", "formula")
Factory = Callable[..., Any]

_FACTORIES: dict[tuple[str, str], Factory] = {}
_INSTANCES: dict[tuple[str, str, tuple[tuple[str, Any], ...]], Any] = {}
_BUILTINS_LOADED = False


def _validate_kind(kind: str) -> None:
    if kind not in VALID_KINDS:
        raise ValueError(f"unknown backend kind: {kind!r}")


def _ensure_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    _FACTORIES[("detector", "heron")] = _heron_detector_factory
    _FACTORIES[("detector", "heron-onnx")] = _heron_onnx_detector_factory
    _FACTORIES[("layout", "heron")] = _heron_layout_factory
    _FACTORIES[("layout", "heron-onnx")] = _heron_onnx_layout_factory
    _FACTORIES[("layout", "mineru-pipeline")] = _mineru_layout_factory
    _FACTORIES[("ocr", "rapidocr")] = _rapidocr_factory
    _FACTORIES[("ocr", "rapidocr-onnx")] = _rapidocr_onnx_factory
    _FACTORIES[("ocr", "rapidocr-torch")] = _rapidocr_torch_factory
    _FACTORIES[("reader", "composed")] = _composed_reader_factory
    _FACTORIES[("reader", "heron-onnx+rapidocr-onnx")] = _heron_onnx_rapidocr_onnx_factory
    _FACTORIES[("reader", "heron-onnx+rapidocr-torch")] = _heron_onnx_rapidocr_torch_factory
    _FACTORIES[("reader", "mineru-pipeline")] = _mineru_reader_factory
    _FACTORIES[("grouper", "v1_anchor")] = _v1_anchor_factory
    _FACTORIES[("grouper", "v2_vector_vision")] = _v2_vector_vision_factory
    _FACTORIES[("grouper", "raw_picture")] = _raw_picture_factory
    _FACTORIES[("table", "tatr-onnx")] = _tatr_onnx_factory
    _FACTORIES[("table", "pdfplumber")] = _pdfplumber_table_factory
    _FACTORIES[("table", "auto")] = _auto_table_factory
    _FACTORIES[("diagram", "vector_first")] = _vector_first_diagram_factory
    _FACTORIES[("formula", "unimernet-onnx")] = _unimernet_onnx_factory
    _FACTORIES[("formula", "rapid-latex-ocr")] = _rapid_latex_ocr_factory
    _FACTORIES[("formula", "auto")] = _unimernet_onnx_factory


def _heron_detector_factory(*, device: str = "auto", **_extra: Any) -> Any:
    from .heron import HeronDetector

    return HeronDetector(device=device)


def _heron_onnx_detector_factory(*, device: str = "auto", **_extra: Any) -> Any:
    from .heron_onnx import HeronOnnxDetector

    return HeronOnnxDetector(device=device)


def _heron_layout_factory(*, device: str = "auto", dpi: int = 150, **_extra: Any) -> Any:
    from .heron import HERON_REVISION
    from .layout_image import ImageLayoutBackend

    detector = get("detector", "heron", device=device)
    return ImageLayoutBackend(detector=detector, name="heron", revision=HERON_REVISION, dpi=dpi)


def _heron_onnx_layout_factory(*, device: str = "auto", dpi: int = 150, **_extra: Any) -> Any:
    from .heron_onnx import HERON_ONNX_REVISION
    from .layout_image import ImageLayoutBackend

    detector = get("detector", "heron-onnx", device=device)
    return ImageLayoutBackend(detector=detector, name="heron-onnx", revision=HERON_ONNX_REVISION, dpi=dpi)


def _mineru_layout_factory(*, device: str = "auto", python_exe: str | None = None, **_extra: Any) -> Any:
    from .mineru import MinerULayoutBackend

    return MinerULayoutBackend(device=device, python_exe=python_exe)


def _rapidocr_factory(*, text_score: float = 0.5, lang: str | None = None, **_extra: Any) -> Any:
    from .rapidocr import RapidOcrRecognizer

    return RapidOcrRecognizer(text_score=text_score, engine="onnxruntime", device="cpu", lang=lang)


def _rapidocr_onnx_factory(
    *,
    text_score: float = 0.5,
    device: str = "auto",
    lang: str | None = None,
    **_extra: Any,
) -> Any:
    from .rapidocr import RapidOcrRecognizer

    return RapidOcrRecognizer(text_score=text_score, engine="onnxruntime", device=device, lang=lang)


def _rapidocr_torch_factory(
    *,
    text_score: float = 0.5,
    device: str = "auto",
    lang: str | None = None,
    **_extra: Any,
) -> Any:
    from .rapidocr import RapidOcrRecognizer

    return RapidOcrRecognizer(text_score=text_score, engine="torch", device=device, lang=lang)


def _composed_reader_factory(
    *,
    layout: str = "heron-onnx",
    ocr_backend: str = "rapidocr-onnx",
    device: str = "auto",
    dpi: int = 150,
    ocr_lang: str | None = None,
    **_extra: Any,
) -> Any:
    from .reader_composed import ComposedReader

    detector = get("detector", layout, device=device)
    ocr = get("ocr", ocr_backend, device=device, **({"lang": ocr_lang} if ocr_lang is not None else {}))
    return ComposedReader(detector=detector, ocr=ocr, dpi=dpi)


def _heron_onnx_rapidocr_onnx_factory(
    *,
    device: str = "auto",
    dpi: int = 150,
    ocr_lang: str | None = None,
    **_extra: Any,
) -> Any:
    from .reader_composed import ComposedReader

    detector = get("detector", "heron-onnx", device=device)
    ocr = get("ocr", "rapidocr-onnx", device=device, **({"lang": ocr_lang} if ocr_lang is not None else {}))
    return ComposedReader(detector=detector, ocr=ocr, dpi=dpi)


def _heron_onnx_rapidocr_torch_factory(
    *,
    device: str = "auto",
    dpi: int = 150,
    ocr_lang: str | None = None,
    **_extra: Any,
) -> Any:
    from .reader_composed import ComposedReader

    detector = get("detector", "heron-onnx", device=device)
    ocr = get("ocr", "rapidocr-torch", device=device, **({"lang": ocr_lang} if ocr_lang is not None else {}))
    return ComposedReader(detector=detector, ocr=ocr, dpi=dpi)


def _mineru_reader_factory(*, device: str = "auto", python_exe: str | None = None, **_extra: Any) -> Any:
    from .mineru import MinerUReader

    return MinerUReader(device=device, python_exe=python_exe)


def _v1_anchor_factory(**_extra: Any) -> Any:
    from ..grouping import V1AnchorGrouper

    return V1AnchorGrouper()


def _v2_vector_vision_factory(**_extra: Any) -> Any:
    from ..grouping import V2VectorVisionGrouper

    return V2VectorVisionGrouper()


def _raw_picture_factory(**_extra: Any) -> Any:
    from ..grouping import RawPictureGrouper

    return RawPictureGrouper()


def _tatr_onnx_factory(*, device: str = "auto", **_extra: Any) -> Any:
    from .tatr_onnx import TatrOnnxTableExtractor

    return TatrOnnxTableExtractor(device=device)


def _pdfplumber_table_factory(**_extra: Any) -> Any:
    from .table_pdfplumber import PdfPlumberTableExtractor

    return PdfPlumberTableExtractor()


def _auto_table_factory(*, device: str = "auto", **_extra: Any) -> Any:
    from .table_auto import AutoTableExtractor

    return AutoTableExtractor(device=device)


def _vector_first_diagram_factory(**_extra: Any) -> Any:
    from ..diagrams import VectorFirstDiagramExtractor

    return VectorFirstDiagramExtractor()


def _unimernet_onnx_factory(
    *,
    device: str = "cpu",
    tier: str = "small",
    model_path: str | None = None,
    max_tokens: int = 512,
    **_extra: Any,
) -> Any:
    from .formula_onnx import UnimernetOnnxBackend

    return UnimernetOnnxBackend(
        device=device,
        tier=tier,
        model_path=model_path,
        max_tokens=max_tokens,
    )


def _rapid_latex_ocr_factory(
    *,
    device: str = "auto",
    model_path: str | None = None,
    image_resizer_path: str | None = None,
    encoder_path: str | None = None,
    decoder_path: str | None = None,
    tokenizer_json: str | None = None,
    **_extra: Any,
) -> Any:
    from .formula_onnx import RapidLatexOcrBackend

    return RapidLatexOcrBackend(
        device=device,
        model_path=model_path,
        image_resizer_path=image_resizer_path,
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        tokenizer_json=tokenizer_json,
    )


def register(kind: str, name: str, factory: Factory) -> None:
    _validate_kind(kind)
    _ensure_builtins()
    _FACTORIES[(kind, name)] = factory
    stale = [k for k in _INSTANCES if k[0] == kind and k[1] == name]
    for k in stale:
        del _INSTANCES[k]


def _freeze_opts(opts: dict[str, Any]) -> tuple[tuple[str, Any], ...] | None:
    items = tuple(sorted((k, v) for k, v in opts.items() if v is not None))
    try:
        hash(items)
    except TypeError:
        return None
    return items


def get(kind: str, name: str, **opts: Any) -> Any:
    _validate_kind(kind)
    _ensure_builtins()
    factory = _FACTORIES.get((kind, name))
    if factory is None:
        raise ValueError(f"unknown {kind} backend: {name!r}")
    frozen = _freeze_opts(opts)
    if frozen is not None:
        cache_key = (kind, name, frozen)
        if cache_key not in _INSTANCES:
            _INSTANCES[cache_key] = factory(**opts)
        return _INSTANCES[cache_key]
    return factory(**opts)


def registered(kind: str | None = None) -> dict[str, tuple[str, ...]] | tuple[str, ...]:
    _ensure_builtins()
    if kind is not None:
        _validate_kind(kind)
        return tuple(sorted(n for k, n in _FACTORIES if k == kind))
    return {
        k: tuple(sorted(n for fk, n in _FACTORIES if fk == k))
        for k in VALID_KINDS
    }


def reset_cache() -> None:
    _INSTANCES.clear()
