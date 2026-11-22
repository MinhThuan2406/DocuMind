from pathlib import Path
from typing import Sequence

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import LayoutPrediction, Page
from docling.datamodel.document import ConversionResult
from docling.models.base_layout_model import BaseLayoutModel
from docling.utils.profiling import TimeRecorder

from .mapping import detections_to_clusters
from .options import DocLayoutYoloOptions


class DocLayoutYoloModel(BaseLayoutModel):
    def __init__(
        self,
        *,
        artifacts_path: Path | None,
        accelerator_options: AcceleratorOptions,
        options: DocLayoutYoloOptions,
        **_: object,
    ) -> None:
        self.options = options
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "DocLayout-YOLO plugin requires the `doclayout-yolo` package."
            ) from exc

        source = options.model_path
        if source is None and artifacts_path is not None:
            candidate = (
                artifacts_path
                / "doclayout-yolo"
                / Path(options.model_filename).name
            )
            if candidate.exists():
                source = str(candidate)

        if source is None:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise ImportError(
                    "Downloading DocLayout-YOLO weights requires `huggingface-hub`."
                ) from exc
            source = hf_hub_download(
                repo_id=options.repo_id,
                filename=options.model_filename,
            )

        self.model = YOLO(source)

    @classmethod
    def get_options_type(cls) -> type[DocLayoutYoloOptions]:
        return DocLayoutYoloOptions

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
            kwargs: dict[str, object] = {
                "imgsz": self.options.image_size,
                "conf": self.options.confidence,
                "verbose": False,
            }
            if self.options.device is not None:
                kwargs["device"] = self.options.device
            with TimeRecorder(conv_res, "layout"):
                result = self.model.predict(image, **kwargs)[0]
            boxes = result.boxes.xyxy.cpu().tolist()
            class_ids = [int(value) for value in result.boxes.cls.cpu().tolist()]
            scores = [float(value) for value in result.boxes.conf.cpu().tolist()]
            names = result.names
            predictions.append(
                LayoutPrediction(
                    clusters=detections_to_clusters(
                        boxes=boxes,
                        class_ids=class_ids,
                        scores=scores,
                        names=names,
                        image_size=image.size,
                        page_size=(page.size.width, page.size.height),
                    )
                )
            )
        return predictions
