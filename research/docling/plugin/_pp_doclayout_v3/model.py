from pathlib import Path
from typing import Sequence

import numpy as np

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import LayoutPrediction, Page
from docling.datamodel.document import ConversionResult
from docling.models.base_layout_model import BaseLayoutModel
from docling.utils.profiling import TimeRecorder

from .mapping import v3_result_to_clusters
from .options import PPDocLayoutV3Options


class PPDocLayoutV3Model(BaseLayoutModel):
    def __init__(
        self,
        *,
        artifacts_path: Path | None,
        accelerator_options: AcceleratorOptions,
        options: PPDocLayoutV3Options,
        **_: object,
    ) -> None:
        self.options = options
        try:
            from paddleocr import LayoutDetection
        except ImportError as exc:
            raise ImportError(
                "PP-DocLayoutV3 plugin requires PaddleOCR and PaddlePaddle."
            ) from exc

        model_dir = options.model_dir
        if model_dir is None and artifacts_path is not None:
            candidate = artifacts_path / "PP-DocLayoutV3"
            if candidate.exists():
                model_dir = str(candidate)
        self.model = LayoutDetection(
            model_name="PP-DocLayoutV3",
            model_dir=model_dir,
            img_size=options.img_size,
            threshold=options.threshold,
            layout_nms=options.layout_nms,
            device=options.device,
        )

    @classmethod
    def get_options_type(cls) -> type[PPDocLayoutV3Options]:
        return PPDocLayoutV3Options

    def predict_layout(
        self, conv_res: ConversionResult, pages: Sequence[Page]
    ) -> Sequence[LayoutPrediction]:
        predictions: list[LayoutPrediction] = []
        for page in pages:
            assert page._backend is not None
            if not page._backend.is_valid() or page.size is None:
                predictions.append(page.predictions.layout or LayoutPrediction())
                continue
            image = page.get_image(scale=1.0)
            if image is None:
                predictions.append(LayoutPrediction())
                continue
            with TimeRecorder(conv_res, "layout"):
                results = self.model.predict(np.asarray(image))
            clusters = []
            for result in results:
                clusters.extend(
                    v3_result_to_clusters(
                        result,
                        image_size=image.size,
                        page_size=(page.size.width, page.size.height),
                    )
                )
            for index, cluster in enumerate(clusters):
                cluster.id = index
            predictions.append(LayoutPrediction(clusters=clusters))
        return predictions
