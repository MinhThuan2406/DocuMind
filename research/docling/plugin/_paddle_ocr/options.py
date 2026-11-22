from typing import ClassVar, Literal

from pydantic import ConfigDict, Field

from docling.datamodel.pipeline_options import OcrOptions


class PaddleOcrOptions(OcrOptions):
    kind: ClassVar[Literal["paddleocr-v6"]] = "paddleocr-v6"
    lang: list[str] = Field(default_factory=lambda: ["en", "ja"])
    scale: float = Field(default=3.0, gt=0)
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    device: str | None = None
    text_detection_model_name: str = "PP-OCRv6_medium_det"
    text_recognition_model_name: str = "PP-OCRv6_medium_rec"
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = True

    model_config = ConfigDict(extra="forbid")
