from core.embeddings import embed_chunks, embed_query, load_embedding_model


# Verify that document chunks and queries are converted into embeddings.
def test_embeddings():
    records = [
        {
            "text": "Machine learning allows systems to learn patterns from data.",
            "metadata": {
                "doc_type": "sample",
                "filename": "sample.txt",
            },
        }
    ]

    model = load_embedding_model()

    embedded_records = embed_chunks(model, records)
    query_embedding = embed_query(model, "What is machine learning?")

    assert len(embedded_records) == 1
    assert "embedding" in embedded_records[0]

    chunk_embedding = embedded_records[0]["embedding"]

    assert len(chunk_embedding) == 384
    assert query_embedding.shape == (1, 384)


if __name__ == "__main__":
    test_embeddings()
    print("EMBEDDING TEST PASSED")
