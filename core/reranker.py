from sentence_transformers import CrossEncoder


RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MIN_CHUNK_LENGTH = 40


# Load the cross-encoder model used to rerank FAISS results.
def load_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL_NAME)


# Identify chunks that are empty, too short, or mostly numerical.
def is_noisy_chunk(record: dict) -> bool:
    text = record.get("text", "").strip()

    if len(text) < MIN_CHUNK_LENGTH:
        return True

    letter_count = sum(character.isalpha() for character in text)
    digit_count = sum(character.isdigit() for character in text)

    return letter_count == 0 or digit_count > letter_count


# Remove retrieved results that contain identical text.
def remove_duplicate_results(results: list[dict]) -> list[dict]:
    unique_results = []
    seen_text = set()

    for result in results:
        normalized_text = " ".join(result.get("text", "").lower().split())

        if not normalized_text or normalized_text in seen_text:
            continue

        seen_text.add(normalized_text)
        unique_results.append(result)

    return unique_results


# Filter FAISS candidates and rerank them using the cross-encoder.
def rerank_results(
    query: str,
    results: list[dict],
    reranker: CrossEncoder,
    top_k: int = 5,
) -> list[dict]:
    clean_results = [
        result.copy()
        for result in results
        if not is_noisy_chunk(result)
    ]

    clean_results = remove_duplicate_results(clean_results)

    if not clean_results:
        return []

    # Score each question-and-chunk pair for relevance.
    query_chunk_pairs = [
        (query, result.get("text", ""))
        for result in clean_results
    ]

    rerank_scores = reranker.predict(query_chunk_pairs)

    for result, score in zip(clean_results, rerank_scores):
        result["rerank_score"] = float(score)

    clean_results.sort(
        key=lambda result: result["rerank_score"],
        reverse=True,
    )

    return clean_results[:top_k]