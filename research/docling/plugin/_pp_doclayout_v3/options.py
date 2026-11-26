from typing import ClassVar, Literal

from pydantic import ConfigDict, Field

from docling.datamodel.pipeline_options import BaseLayoutOptions


class PPDocLayoutV3Options(BaseLayoutOptions):
    kind: ClassVar[Literal["pp-doclayout-v3"]] = "pp-doclayout-v3"
    model_dir: str | None = None
    threshold: float = Field(default=0.5, ge=0, le=1)
    img_size: int | None = Field(default=None, ge=320)
    layout_nms: bool = True
    device: str | None = None

    model_config = ConfigDict(extra="forbid")
