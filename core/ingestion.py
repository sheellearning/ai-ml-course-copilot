from pathlib import Path

from core.chunking import chunk_records
from core.loaders.docx_loader import load_docx_file
from core.loaders.image_loader import load_image_file
from core.loaders.ipynb_loader import load_ipynb_file
from core.loaders.pdf_loader import load_pdf_pages
from core.loaders.pptx_loader import load_pptx_file
from core.loaders.spreadsheet_loader import load_spreadsheet_file


SUPPORTED_EXTENSIONS = [
    ".pdf",
    ".docx",
    ".pptx",
    ".ipynb",
    ".jpg",
    ".jpeg",
    ".png",
    ".xlsx",
    ".xls",
    ".csv",
]


# Select the correct loader based on the file extension.
def load_file(file_path):
    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return load_pdf_pages(str(file_path))

    if extension == ".docx":
        return load_docx_file(str(file_path))

    if extension == ".pptx":
        return load_pptx_file(str(file_path))

    if extension == ".ipynb":
        return load_ipynb_file(str(file_path))

    if extension in [".jpg", ".jpeg", ".png"]:
        return load_image_file(str(file_path))

    if extension in [".xlsx", ".xls", ".csv"]:
        return load_spreadsheet_file(str(file_path))

    return []


# Load supported files from the selected folder and its subfolders.
def load_folder(folder_path):
    folder = Path(folder_path)
    all_records = []

    for file_path in folder.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        records = load_file(file_path)
        all_records.extend(records)

    return all_records


# Load the documents and convert their records into searchable chunks.
def ingest_folder(folder_path):
    records = load_folder(folder_path)
    chunks = chunk_records(records)

    return chunks