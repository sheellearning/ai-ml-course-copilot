import faiss
import numpy as np
import os
import pickle


# Build a normalized FAISS index from the document embeddings.
def build_faiss_index(records):
    embeddings = [record["embedding"] for record in records]
    embedding_matrix = np.array(embeddings).astype("float32")

    # Normalization allows inner-product search to work like cosine similarity.
    faiss.normalize_L2(embedding_matrix)

    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embedding_matrix)

    return index, records


# Search the index and return matching records with similarity scores.
def search_index(query_embedding, index, records, top_k=3):
    query_vector = np.array([query_embedding]).astype("float32")
    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        record = records[idx].copy()
        record["score"] = float(score)
        results.append(record)

    return results


# Save the FAISS index and its matching records to disk.
def save_faiss_index(index, records, index_path="data/index/faiss.index", records_path="data/index/records.pkl"):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    faiss.write_index(index, index_path)

    with open(records_path, "wb") as file:
        pickle.dump(records, file)


# Load the saved FAISS index and records.
def load_faiss_index(index_path="data/index/faiss.index", records_path="data/index/records.pkl"):
    index = faiss.read_index(index_path)

    with open(records_path, "rb") as file:
        records = pickle.load(file)

    return index, records


# Search FAISS and return the matching text, metadata, and similarity scores.
def search_faiss(index, records, query_embedding, top_k=5):
    distances, indices = index.search(query_embedding, top_k)

    results = []

    for i, idx in enumerate(indices[0]):
        # FAISS returns -1 when a valid matching record is unavailable.
        if idx == -1:
            continue

        record = records[idx]

        result = {
            "text": record["text"],
            "metadata": record["metadata"],
            "score": float(distances[0][i]),
        }

        results.append(result)

    return results
