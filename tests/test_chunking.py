from core.chunking import chunk_records


# Verify that PDF content with Q&A formatting uses the Q&A strategy.
def test_pdf_qa_chunking():
    record = {
        "text": (
            "Q: What is artificial intelligence?\n"
            "A: Artificial intelligence allows machines to perform intelligent tasks.\n"
            "Q: What is machine learning?\n"
            "A: Machine learning allows systems to learn patterns from data."
        ),
        "metadata": {
            "doc_type": "pdf",
            "filename": "sample.pdf",
            "page": 1,
        },
    }

    chunks = chunk_records([record])

    assert len(chunks) == 2
    assert chunks[0]["metadata"]["chunk_strategy"] == "qa"
    assert chunks[1]["metadata"]["chunk_strategy"] == "qa"
    assert chunks[0]["metadata"]["filename"] == "sample.pdf"


# Verify that unknown document types use recursive chunking.
def test_generic_chunking():
    record = {
        "text": "Machine learning uses data to identify patterns and make predictions.",
        "metadata": {
            "doc_type": "unknown",
            "filename": "sample.txt",
        },
    }

    chunks = chunk_records([record])

    assert len(chunks) == 1
    assert chunks[0]["metadata"]["chunk_strategy"] == "generic_recursive"
    assert chunks[0]["text"] == record["text"]


if __name__ == "__main__":
    test_pdf_qa_chunking()
    test_generic_chunking()

    print("CHUNKING TESTS PASSED")
