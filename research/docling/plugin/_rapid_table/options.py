from typing import ClassVar, Literal

from pydantic import ConfigDict, Field

from docling.datamodel.pipeline_options import BaseTableStructureOptions


class RapidTableOptions(BaseTableStructureOptions):
    kind: ClassVar[Literal["rapidtable-slanet-plus"]] = "rapidtable-slanet-plus"
    scale: float = Field(default=2.0, gt=0)
    model_path: str | None = None
    use_ocr: bool = True
    engine_cfg: dict[str, object] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")
