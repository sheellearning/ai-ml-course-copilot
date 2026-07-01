import json
import math
import re
import time
from pathlib import Path

from core.embeddings import embed_query, load_embedding_model
from core.rag_pipeline import generate_answer
from core.reranker import load_reranker, rerank_results
from core.vector_store import load_faiss_index, search_faiss


# Retrieval and confidence settings used by the evaluation.
FAISS_THRESHOLD = 0.55
RERANKER_THRESHOLD = 4.0
CANDIDATE_COUNT = 30
FINAL_RESULT_COUNT = 5

QUESTIONS_PATH = Path("evaluation_questions.json")
RESULTS_PATH = Path("evaluation_results.json")

NO_ANSWER_TEXT = "No reliable answer found in uploaded documents."


# Clean the question before retrieval.
def normalize_query(query: str) -> str:
    cleaned_query = " ".join(query.strip().split())

    return re.sub(r"\bAL\b", "AI", cleaned_query, flags=re.IGNORECASE)


# Normalize generated text before checking expected concepts.
def normalize_text(text: str) -> str:
    normalized = text.lower().replace("-", " ")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)

    return " ".join(normalized.split())


# Retrieve FAISS candidates, rerank them, and apply the confidence gate.
def retrieve_results(question, index, records, embedding_model, reranker):
    query_embedding = embed_query(embedding_model, question)

    candidate_results = search_faiss(
        index,
        records,
        query_embedding,
        top_k=CANDIDATE_COUNT,
    )

    results = rerank_results(
        question,
        candidate_results,
        reranker,
        top_k=FINAL_RESULT_COUNT,
    )

    if not results:
        return {
            "results": [],
            "candidate_count": len(candidate_results),
            "top_faiss_score": 0.0,
            "top_reranker_score": 0.0,
            "can_answer": False,
        }

    top_faiss_score = max(float(result.get("score", 0)) for result in results)
    top_reranker_score = max(float(result.get("rerank_score", 0)) for result in results)

    # Evidence is accepted when either retrieval score passes its threshold.
    can_answer = (
        top_faiss_score >= FAISS_THRESHOLD
        or top_reranker_score >= RERANKER_THRESHOLD
    )

    return {
        "results": results,
        "candidate_count": len(candidate_results),
        "top_faiss_score": top_faiss_score,
        "top_reranker_score": top_reranker_score,
        "can_answer": can_answer,
    }


# Compare the generated answer with the expected concepts.
def find_expected_terms(answer: str, expected_terms: list[str]) -> tuple[list[str], list[str]]:
    normalized_answer = normalize_text(answer)

    matched_terms = []
    missing_terms = []

    for term in expected_terms:
        normalized_term = normalize_text(term)

        if normalized_term in normalized_answer:
            matched_terms.append(term)
        else:
            missing_terms.append(term)

    return matched_terms, missing_terms


# Collect source and score details for the saved evaluation report.
def get_source_details(results: list[dict]) -> list[dict]:
    sources = []

    for result in results:
        metadata = result.get("metadata", {})

        sources.append(
            {
                "filename": metadata.get("filename", "Unknown"),
                "doc_type": metadata.get("doc_type", "unknown"),
                "page": metadata.get("page"),
                "slide": metadata.get("slide"),
                "cell_index": metadata.get("cell_index"),
                "sheet_name": metadata.get("sheet_name"),
                "faiss_score": round(float(result.get("score", 0)), 4),
                "reranker_score": round(float(result.get("rerank_score", 0)), 4),
            }
        )

    return sources


# Run one evaluation question and return its complete result.
def evaluate_question(test_case, index, records, embedding_model, reranker):
    question = test_case["question"]
    expected_behavior = test_case["expected_behavior"]
    expected_terms = test_case.get("expected_terms", [])

    start_time = time.perf_counter()
    normalized_question = normalize_query(question)

    retrieval = retrieve_results(
        normalized_question,
        index,
        records,
        embedding_model,
        reranker,
    )

    results = retrieval["results"]

    if retrieval["can_answer"]:
        answer = generate_answer(normalized_question, results)
    else:
        answer = NO_ANSWER_TEXT

    refused = "no reliable answer found" in answer.lower()

    matched_terms, missing_terms = find_expected_terms(answer, expected_terms)

    # Supported answers must include at least 60% of the expected concepts.
    required_term_count = (
        math.ceil(len(expected_terms) * 0.6)
        if expected_terms
        else 0
    )

    if expected_behavior == "answer":
        behavior_correct = not refused
        term_coverage_correct = len(matched_terms) >= required_term_count
        passed = behavior_correct and term_coverage_correct
    else:
        behavior_correct = refused
        term_coverage_correct = True
        passed = behavior_correct

    elapsed_seconds = time.perf_counter() - start_time

    return {
        "id": test_case["id"],
        "category": test_case["category"],
        "question": question,
        "expected_behavior": expected_behavior,
        "actual_behavior": "refuse" if refused else "answer",
        "passed": passed,
        "behavior_correct": behavior_correct,
        "term_coverage_correct": term_coverage_correct,
        "matched_terms": matched_terms,
        "missing_terms": missing_terms,
        "required_term_count": required_term_count,
        "top_faiss_score": round(retrieval["top_faiss_score"], 4),
        "top_reranker_score": round(retrieval["top_reranker_score"], 4),
        "confidence_gate_passed": retrieval["can_answer"],
        "candidate_count": retrieval["candidate_count"],
        "final_result_count": len(results),
        "response_time_seconds": round(elapsed_seconds, 2),
        "answer": answer,
        "sources": get_source_details(results),
        "notes": test_case.get("notes", ""),
    }


# Print one evaluation result in the terminal.
def print_result(result: dict) -> None:
    if result.get("actual_behavior") == "error":
        print("\n" + "=" * 70)
        print(f'ERROR: {result.get("id", "unknown")}')
        print("Question:", result.get("question", ""))
        print("Error:", result.get("error", ""))
        return

    status = "PASS" if result["passed"] else "FAIL"

    print("\n" + "=" * 70)
    print(f'{status}: {result["id"]}')
    print(f'Question: {result["question"]}')
    print("Expected behavior:", result["expected_behavior"])
    print("Actual behavior:", result["actual_behavior"])
    print("Top FAISS score:", result["top_faiss_score"])
    print("Top reranker score:", result["top_reranker_score"])
    print("Confidence gate passed:", result["confidence_gate_passed"])
    print("Response time:", result["response_time_seconds"], "seconds")

    if result["expected_behavior"] == "answer":
        print("Matched terms:", result["matched_terms"])
        print("Missing terms:", result["missing_terms"])

    print("Answer:")
    print(result["answer"])


# Calculate the percentage of evaluation cases that passed.
def calculate_pass_rate(results: list[dict]) -> float:
    if not results:
        return 0.0

    passed_count = sum(1 for result in results if result.get("passed"))

    return passed_count / len(results) * 100


# Load the models, run all tests, and save the evaluation report.
def main() -> None:
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"Evaluation file not found: {QUESTIONS_PATH}")

    with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        test_cases = json.load(file)

    print("Loading FAISS index...")
    index, records = load_faiss_index()

    print("Loading embedding model...")
    embedding_model = load_embedding_model()

    print("Loading reranker model...")
    reranker = load_reranker()

    evaluation_results = []

    # One failed test should not stop the remaining evaluations.
    for test_case in test_cases:
        try:
            result = evaluate_question(
                test_case,
                index,
                records,
                embedding_model,
                reranker,
            )
        except Exception as error:
            result = {
                "id": test_case.get("id", "unknown"),
                "category": test_case.get("category", "unknown"),
                "question": test_case.get("question", ""),
                "expected_behavior": test_case.get("expected_behavior", ""),
                "actual_behavior": "error",
                "passed": False,
                "error": str(error),
            }

        evaluation_results.append(result)
        print_result(result)

    answer_results = [
        result
        for result in evaluation_results
        if result.get("expected_behavior") == "answer"
    ]

    refusal_results = [
        result
        for result in evaluation_results
        if result.get("expected_behavior") == "refuse"
    ]

    total_tests = len(evaluation_results)
    passed_tests = sum(1 for result in evaluation_results if result.get("passed"))
    failed_tests = total_tests - passed_tests

    overall_pass_rate = calculate_pass_rate(evaluation_results)
    answer_pass_rate = calculate_pass_rate(answer_results)
    refusal_pass_rate = calculate_pass_rate(refusal_results)

    report = {
        "summary": {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "overall_pass_rate": round(overall_pass_rate, 2),
            "supported_answer_pass_rate": round(answer_pass_rate, 2),
            "unsupported_refusal_rate": round(refusal_pass_rate, 2),
        },
        "results": evaluation_results,
    }

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Overall pass rate: {overall_pass_rate:.2f}%")
    print(f"Supported-answer pass rate: {answer_pass_rate:.2f}%")
    print(f"Unsupported-question refusal rate: {refusal_pass_rate:.2f}%")
    print(f"Detailed results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()