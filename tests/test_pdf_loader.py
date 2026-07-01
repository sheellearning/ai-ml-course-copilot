from pathlib import Path
from tempfile import TemporaryDirectory

import fitz

from core.loaders.pdf_loader import load_pdf_pages


# Create a small temporary PDF for testing.
def create_sample_pdf(pdf_path):
    pdf_doc = fitz.open()
    page = pdf_doc.new_page()
    page.insert_text((72, 72), "Artificial intelligence test content.")
    pdf_doc.save(str(pdf_path))
    pdf_doc.close()


# Verify that the PDF loader extracts text and metadata correctly.
def test_pdf_loader():
    with TemporaryDirectory() as temporary_folder:
        pdf_path = Path(temporary_folder) / "sample.pdf"
        create_sample_pdf(pdf_path)

        pages = load_pdf_pages(str(pdf_path))

        assert len(pages) == 1

        first_page = pages[0]

        assert "Artificial intelligence test content." in first_page["text"]
        assert first_page["metadata"]["source"] == "course"
        assert first_page["metadata"]["doc_type"] == "pdf"
        assert first_page["metadata"]["filename"] == "sample.pdf"
        assert first_page["metadata"]["page"] == 1
        assert first_page["metadata"]["path"] == str(pdf_path)


if __name__ == "__main__":
    test_pdf_loader()
    print("PDF LOADER TEST PASSED")
