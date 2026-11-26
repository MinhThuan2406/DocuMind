from typing import ClassVar, Literal

from pydantic import ConfigDict, Field

from docling.datamodel.pipeline_options import BaseLayoutOptions


class DocLayoutYoloOptions(BaseLayoutOptions):
    kind: ClassVar[Literal["doclayout-yolo"]] = "doclayout-yolo"
    repo_id: str = "juliozhao/DocLayout-YOLO"
    model_filename: str = "universal/DocLayout-YOLO-universal.pt"
    model_path: str | None = None
    image_size: int = Field(default=1024, ge=320)
    confidence: float = Field(default=0.2, ge=0, le=1)
    device: str | None = None

    model_config = ConfigDict(extra="forbid")
