from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from core.vector_store import (
    build_faiss_index,
    load_faiss_index,
    save_faiss_index,
    search_faiss,
    search_index,
)


# Create small records with predictable embeddings for FAISS testing.
def create_sample_records():
    return [
        {
            "text": "Supervised learning uses labelled training data.",
            "metadata": {
                "doc_type": "sample",
                "filename": "supervised.txt",
            },
            "embedding": np.array([1.0, 0.0], dtype="float32"),
        },
        {
            "text": "Unsupervised learning identifies patterns without labels.",
            "metadata": {
                "doc_type": "sample",
                "filename": "unsupervised.txt",
            },
            "embedding": np.array([0.0, 1.0], dtype="float32"),
        },
    ]


# Verify that FAISS returns the most similar record first.
def test_vector_search():
    records = create_sample_records()
    index, indexed_records = build_faiss_index(records)

    query_embedding = np.array([1.0, 0.0], dtype="float32")
    results = search_index(query_embedding, index, indexed_records, top_k=2)

    assert len(results) == 2
    assert results[0]["metadata"]["filename"] == "supervised.txt"
    assert results[0]["score"] >= results[1]["score"]


# Verify that the saved index and records can be loaded and searched.
def test_save_and_load_index():
    records = create_sample_records()
    index, indexed_records = build_faiss_index(records)

    with TemporaryDirectory() as temporary_folder:
        index_path = Path(temporary_folder) / "faiss.index"
        records_path = Path(temporary_folder) / "records.pkl"

        save_faiss_index(
            index,
            indexed_records,
            index_path=str(index_path),
            records_path=str(records_path),
        )

        loaded_index, loaded_records = load_faiss_index(
            index_path=str(index_path),
            records_path=str(records_path),
        )

        query_embedding = np.array([[1.0, 0.0]], dtype="float32")
        results = search_faiss(
            loaded_index,
            loaded_records,
            query_embedding,
            top_k=1,
        )

        assert len(results) == 1
        assert results[0]["metadata"]["filename"] == "supervised.txt"
        assert "labelled training data" in results[0]["text"]


if __name__ == "__main__":
    test_vector_search()
    test_save_and_load_index()

    print("VECTOR STORE TESTS PASSED")
