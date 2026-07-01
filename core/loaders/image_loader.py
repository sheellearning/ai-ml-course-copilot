from typing import List, Dict, Any
import pytesseract
from PIL import Image
import os


# Configure the local Tesseract OCR executable.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Extract text from an image using OCR and return it with source metadata.
def load_image_file(image_path: str) -> List[Dict[str, Any]]:
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image).strip()

    if not extracted_text:
        return []

    filename = os.path.basename(image_path)

    metadata = {
        "source": "course",
        "doc_type": "image_ocr",
        "filename": filename,
        "path": image_path,
    }

    return [{"text": extracted_text, "metadata": metadata}]