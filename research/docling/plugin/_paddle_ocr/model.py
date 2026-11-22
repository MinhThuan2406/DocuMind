from collections.abc import Iterable
from pathlib import Path

import numpy as np

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.profiling import TimeRecorder

from .mapping import result_to_cells
from .options import PaddleOcrOptions


class PaddleOcrModel(BaseOcrModel):
    multiple_languages = True

    def __init__(
        self,
        *,
        enabled: bool,
        artifacts_path: Path | None,
        options: PaddleOcrOptions,
        accelerator_options: AcceleratorOptions,
        **_: object,
    ) -> None:
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options = options
        self.scale = options.scale
        self.reader = None
        if enabled:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise ImportError(
                    "PaddleOCR plugin requires `paddleocr` and a PaddlePaddle runtime."
                ) from exc
            kwargs: dict[str, object] = {
                "text_detection_model_name": options.text_detection_model_name,
                "text_recognition_model_name": options.text_recognition_model_name,
                "use_doc_orientation_classify": options.use_doc_orientation_classify,
                "use_doc_unwarping": options.use_doc_unwarping,
                "use_textline_orientation": options.use_textline_orientation,
            }
            if options.device is not None:
                kwargs["device"] = options.device
            self.reader = PaddleOCR(**kwargs)

    @classmethod
    def get_options_type(cls) -> type[PaddleOcrOptions]:
        return PaddleOcrOptions

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return
        assert self.reader is not None
        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
                continue
            with TimeRecorder(conv_res, "ocr"):
                ocr_rects = self.get_ocr_rects(page)
                cells = []
                for rect in ocr_rects:
                    if rect.area() == 0:
                        continue
                    image = page._backend.get_page_image(scale=self.scale, cropbox=rect)
                    results = self.reader.predict(np.asarray(image))
                    for result in results:
                        cells.extend(
                            result_to_cells(
                                result,
                                crop=rect,
                                scale=self.scale,
                                threshold=self.options.confidence_threshold,
                            )
                        )
                self.post_process_cells(cells, page, conv_res)
            yield page
