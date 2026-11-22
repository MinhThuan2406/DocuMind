from importlib.metadata import version

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options import (
    HeadingHierarchyOptions,
    PdfPipelineOptions,
)
from _doclayout_yolo import DocLayoutYoloOptions
from _paddle_ocr import PaddleOcrOptions
from _rapid_table import RapidTableOptions


print("docling:", version("docling"))
print("docling-core:", version("docling-core"))


# %% AcceleratorOptions
accelerator_options = AcceleratorOptions(
    device=AcceleratorDevice.CUDA,
    num_threads=4,
    cuda_use_flash_attention2=False,
)


# %% PaddleOCR plugin
ocr_options = PaddleOcrOptions(
    lang=["en", "ja"],
    scale=3.0,
    device="gpu",
)


# %% DocLayout-YOLO plugin
layout_options = DocLayoutYoloOptions(
    image_size=1024,
    confidence=0.2,
    device="cuda:0",
)


# %% RapidTable plugin
table_options = RapidTableOptions(
    scale=2.0,
    use_ocr=True,
)


# %% HeadingHierarchyOptions
heading_options = HeadingHierarchyOptions(
    enabled=True,
    use_bookmarks=True,
    use_numbering=True,
    use_style=True,
    use_font_style=True,
    max_level=6,
)


# %% PdfPipelineOptions
pdf_options = PdfPipelineOptions(
    # Runtime & security
    document_timeout=None,
    accelerator_options=accelerator_options,
    enable_remote_services=False,
    allow_external_plugins=True,
    artifacts_path=None,  # Set to a local model directory for offline inference.

    # OCR, layout & table structure
    do_ocr=True,
    ocr_options=ocr_options,
    do_table_structure=True,
    table_structure_options=table_options,
    layout_options=layout_options,
    force_backend_text=False,

    # Reading order / heading hierarchy
    use_reading_order_separators=True,
    heading_hierarchy_options=heading_options,

    # Assets cho validator, crop & recovery
    images_scale=2.0,
    generate_page_images=True,
    generate_picture_images=True,
    generate_table_images=True,
    generate_parsed_pages=True,

    # Built-in enrichment
    do_picture_classification=True,
    do_picture_description=False,
    do_chart_extraction=True,
    do_code_enrichment=True,
    do_formula_enrichment=True,

    # Batching / throughput
    ocr_batch_size=2,
    layout_batch_size=2,
    table_batch_size=2,
    batch_polling_interval_seconds=0.5,
    queue_max_size=100,
    stage_shutdown_timeout_seconds=60.0,
)

# pdf_options.layout_options
# pdf_options.picture_classification_options
# pdf_options.picture_description_options
# pdf_options.chart_extraction_options
# pdf_options.code_formula_options
