from core.embeddings import embed_chunks, load_embedding_model
from core.ingestion import ingest_folder
from core.vector_store import build_faiss_index, save_faiss_index


UPLOAD_FOLDER = "data/uploads"


# Load and chunk all supported documents.
chunks = ingest_folder(UPLOAD_FOLDER)
print("Chunks created:", len(chunks))


# Generate an embedding for each document chunk.
model = load_embedding_model()
embedded_chunks = embed_chunks(model, chunks)


# Build and save the FAISS index and matching records.
index, records = build_faiss_index(embedded_chunks)
save_faiss_index(index, records)


print("FAISS index saved successfully.")
print("Saved to: data/index/faiss.index")
print("Saved records to: data/index/records.pkl")