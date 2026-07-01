from sentence_transformers import SentenceTransformer


# Load the embedding model used for semantic search.
def load_embedding_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


# Generate and store an embedding for each document chunk.
def embed_chunks(model, records):
    for record in records:
        text = record["text"]
        embedding = model.encode(text)
        record["embedding"] = embedding

    return records


# Convert one user question into the 2D vector expected by FAISS.
def embed_query(model, query: str):
    embedding = model.encode([query])
    return embedding