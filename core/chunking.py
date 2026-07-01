from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


# Split generic text into smaller overlapping chunks.
def recursive_chunk_text(text, chunk_size=500, chunk_overlap=80):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    return splitter.split_text(text)


# Split Q&A content using question boundaries.
def chunk_qa_text(text):
    parts = text.split("\nQ:")
    chunks = []

    for i, part in enumerate(parts):
        # Add back the question prefix removed during splitting.
        if i == 0:
            chunk_text = part.strip()
        else:
            chunk_text = "Q:" + part.strip()

        if not chunk_text:
            continue

        chunks.append(chunk_text)

    return chunks


# Detect whether the document contains structured Q&A content.
def looks_like_qa(text):
    return text.count("Q:") >= 2 and text.count("A:") >= 2


# Apply Q&A or recursive chunking to PDF records.
def chunk_pdf_record(record):
    text = record["text"]
    metadata = record["metadata"]

    if looks_like_qa(text):
        raw_chunks = chunk_qa_text(text)
        strategy = "qa"
    else:
        raw_chunks = recursive_chunk_text(text)
        strategy = "recursive"

    chunks = []

    for i, chunk_text in enumerate(raw_chunks):
        chunks.append(
            {
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "chunk_strategy": strategy,
                },
            }
        )

    return chunks


# Apply recursive chunking to DOCX records.
def chunk_docx_record(record):
    text = record["text"]
    metadata = record["metadata"]
    raw_chunks = recursive_chunk_text(text)

    chunks = []

    for i, chunk_text in enumerate(raw_chunks):
        chunks.append(
            {
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "chunk_strategy": "recursive_docx",
                },
            }
        )

    return chunks


# Chunk PowerPoint content while preserving slide metadata.
def chunk_pptx_record(record):
    text = record["text"]
    metadata = record["metadata"]
    raw_chunks = recursive_chunk_text(text, chunk_size=700, chunk_overlap=60)

    chunks = []

    for i, chunk_text in enumerate(raw_chunks):
        chunks.append(
            {
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "chunk_strategy": "slide",
                },
            }
        )

    return chunks


# Chunk notebook content while preserving cell metadata.
def chunk_ipynb_record(record):
    text = record["text"]
    metadata = record["metadata"]
    raw_chunks = recursive_chunk_text(text, chunk_size=700, chunk_overlap=60)

    chunks = []

    for i, chunk_text in enumerate(raw_chunks):
        chunks.append(
            {
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "chunk_strategy": "cell",
                },
            }
        )

    return chunks


# Chunk OCR text extracted from image files.
def chunk_image_record(record):
    text = record["text"]
    metadata = record["metadata"]
    raw_chunks = recursive_chunk_text(text, chunk_size=500, chunk_overlap=50)

    chunks = []

    for i, chunk_text in enumerate(raw_chunks):
        chunks.append(
            {
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "chunk_strategy": "image_ocr",
                },
            }
        )

    return chunks


# Chunk spreadsheet records using smaller chunk sizes.
def chunk_spreadsheet_record(record):
    text = record["text"]
    metadata = record["metadata"]
    raw_chunks = recursive_chunk_text(text, chunk_size=400, chunk_overlap=40)

    chunks = []

    for i, chunk_text in enumerate(raw_chunks):
        chunks.append(
            {
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "chunk_strategy": "spreadsheet",
                },
            }
        )

    return chunks


# Use recursive chunking for unknown document types.
def chunk_generic_record(record):
    text = record["text"]
    metadata = record["metadata"]
    raw_chunks = recursive_chunk_text(text)

    chunks = []

    for i, chunk_text in enumerate(raw_chunks):
        chunks.append(
            {
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "chunk_strategy": "generic_recursive",
                },
            }
        )

    return chunks


# Route each record to the correct chunking strategy.
def chunk_one_record(record):
    doc_type = record["metadata"].get("doc_type", "unknown")

    if doc_type == "pdf":
        return chunk_pdf_record(record)

    if doc_type == "docx":
        return chunk_docx_record(record)

    if doc_type == "pptx":
        return chunk_pptx_record(record)

    if doc_type == "ipynb":
        return chunk_ipynb_record(record)

    if doc_type in ["image_ocr", "jpg", "jpeg", "png"]:
        return chunk_image_record(record)

    if doc_type in ["xlsx", "xls", "csv", "spreadsheet"]:
        return chunk_spreadsheet_record(record)

    return chunk_generic_record(record)


# Process all extracted records and return the final searchable chunks.
def chunk_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_chunks = []

    for record in records:
        record_chunks = chunk_one_record(record)
        all_chunks.extend(record_chunks)

    return all_chunks