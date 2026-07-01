import os
from typing import List, Dict, Any

import fitz


# Load a PDF and extract text from each page with source metadata.
def load_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    pdf_doc = fitz.open(pdf_path)
    pages: List[Dict[str, Any]] = []

    filename = os.path.basename(pdf_path)

    for page_index in range(len(pdf_doc)):
        page = pdf_doc[page_index]
        page_text = page.get_text("text")
        page_text = (page_text or "").strip()

        # Skip pages that do not contain readable text.
        if not page_text:
            continue

        metadata = {
            "source": "course",
            "doc_type": "pdf",
            "filename": filename,
            "page": page_index + 1,
            "path": pdf_path,
        }

        record = {
            "text": page_text,
            "metadata": metadata,
        }

        pages.append(record)

    pdf_doc.close()

    return pages