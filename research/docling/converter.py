# %%
from pathlib import Path
from dotenv import load_dotenv

docling_dir = Path.cwd()
load_dotenv(docling_dir / ".env", override=True)

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from config import pdf_options


# %%
converter = DocumentConverter(
    allowed_formats=[
        InputFormat.PDF, InputFormat.DOCX,
        InputFormat.PPTX, InputFormat.XLSX,
    ],
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
    },
)
