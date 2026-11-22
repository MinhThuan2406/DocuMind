from pathlib import Path
from collections.abc import Iterable, Iterator
from typing import Sequence

import numpy as np

from docling_core.types.doc import DocItemLabel
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Cluster, Page, Table, TableStructurePrediction
from docling.datamodel.document import ConversionResult
from docling.models.base_table_model import BaseTableStructureModel
from docling.utils.profiling import TimeRecorder

from .mapping import rapid_cells_to_docling
from .options import RapidTableOptions


try:
    from docling.utils.text_normalization import normalize_extracted_text
except ModuleNotFoundError:
    # Docling 2.135 did not normalize TableFormer text either.
    def normalize_extracted_text(text: str) -> str:
        return text


try:
    from docling.models.base_table_model import (
        keep_table_text_as_child as _keep_table_text_as_child,
    )
except ImportError:
    # Compatibility with Docling 2.135, before this helper became public.
    def _clusters_in_tree(clusters: Iterable[Cluster]) -> Iterator[Cluster]:
        for cluster in clusters:
            yield cluster
            yield from _clusters_in_tree(cluster.children)

    def _keep_table_text_as_child(table_cluster: Cluster, page: Page) -> None:
        layout = page.predictions.layout
        clusters = list(
            _clusters_in_tree(
                [*(layout.clusters if layout is not None else []), *table_cluster.children]
            )
        )
        held = {
            cell.index
            for cluster in clusters
            if cluster is not table_cluster
            for cell in cluster.cells
        }
        cells = [
            cell
            for cell in page.cells
            if cell.text.strip()
            and cell.index not in held
            and cell.rect.to_bounding_box().intersection_over_self(table_cluster.bbox)
            > 0.5
        ]
        if cells:
            table_cluster.children.append(
                Cluster(
                    id=max([table_cluster.id, *(cluster.id for cluster in clusters)]) + 1,
                    label=DocItemLabel.TEXT,
                    bbox=table_cluster.bbox,
                    confidence=table_cluster.confidence,
                    cells=cells,
                )
            )


class RapidTableModel(BaseTableStructureModel):
    def __init__(
        self,
        *,
        enabled: bool,
        artifacts_path: Path | None,
        options: RapidTableOptions,
        accelerator_options: AcceleratorOptions,
        **_: object,
    ) -> None:
        self.enabled = enabled
        self.options = options
        self.scale = options.scale
        self.engine = None
        if enabled:
            try:
                from rapid_table import ModelType, RapidTable, RapidTableInput
            except ImportError as exc:
                raise ImportError(
                    "RapidTable plugin requires `rapid-table` and `rapidocr`."
                ) from exc
            model_path = options.model_path
            if model_path is None and artifacts_path is not None:
                candidate = artifacts_path / "rapid-table"
                if candidate.exists():
                    model_path = str(candidate)
            config = RapidTableInput(
                model_type=ModelType.SLANETPLUS,
                model_dir_or_path=model_path,
                use_ocr=options.use_ocr,
                engine_cfg=options.engine_cfg,
            )
            self.engine = RapidTable(config)

    @classmethod
    def get_options_type(cls) -> type[RapidTableOptions]:
        return RapidTableOptions

    def predict_tables(
        self, conv_res: ConversionResult, pages: Sequence[Page]
    ) -> Sequence[TableStructurePrediction]:
        predictions: list[TableStructurePrediction] = []
        for page in pages:
            prediction = TableStructurePrediction()
            if not self.enabled or self.engine is None:
                predictions.append(prediction)
                continue
            assert page._backend is not None
            if not page._backend.is_valid() or page.predictions.layout is None:
                predictions.append(page.predictions.tablestructure or prediction)
                continue
            with TimeRecorder(conv_res, "table_structure"):
                for cluster in page.predictions.layout.clusters:
                    if cluster.label not in (DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX):
                        continue
                    image = page.get_image(scale=self.scale, cropbox=cluster.bbox)
                    if image is None:
                        _keep_table_text_as_child(cluster, page)
                        continue
                    output = self.engine(np.asarray(image))
                    if not output.cell_bboxes or not output.logic_points:
                        _keep_table_text_as_child(cluster, page)
                        continue
                    cells, rows, cols = rapid_cells_to_docling(
                        cell_bboxes=output.cell_bboxes[0],
                        logic_points=output.logic_points[0],
                        crop=cluster.bbox,
                        scale=self.scale,
                    )
                    for cell in cells:
                        if cell.bbox is not None:
                            cell.text = normalize_extracted_text(
                                page._backend.get_text_in_rect(cell.bbox)
                            )
                    prediction.table_map[cluster.id] = Table(
                        otsl_seq=[],
                        table_cells=cells,
                        num_rows=rows,
                        num_cols=cols,
                        id=cluster.id,
                        page_no=page.page_no,
                        cluster=cluster,
                        label=cluster.label,
                    )
            predictions.append(prediction)
        return predictions
