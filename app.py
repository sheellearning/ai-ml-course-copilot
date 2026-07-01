import re

import streamlit as st

from core.embeddings import embed_query, load_embedding_model
from core.rag_pipeline import generate_answer
from core.reranker import load_reranker, rerank_results
from core.usage_logger import log_failure, log_success
from core.vector_store import load_faiss_index, search_faiss


# Retrieval settings calibrated using the current evaluation questions.
FAISS_THRESHOLD = 0.55
RERANKER_THRESHOLD = 4.0
CANDIDATE_COUNT = 30
FINAL_RESULT_COUNT = 5

NO_ANSWER_MESSAGE = (
    "No reliable answer found in uploaded documents. "
    "Please try rephrasing your question or ask about content "
    "available in the AI/ML course documents."
)


st.set_page_config(
    page_title="AI/ML Course Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# Load the index and models once and reuse them across Streamlit reruns.
@st.cache_resource
def load_system():
    index, records = load_faiss_index()
    embedding_model = load_embedding_model()
    reranker_model = load_reranker()

    return index, records, embedding_model, reranker_model


# Clean the user question before retrieval.
def normalize_query(user_query: str) -> str:
    cleaned_query = " ".join(user_query.strip().split())
    return re.sub(r"\bAL\b", "AI", cleaned_query, flags=re.IGNORECASE)


# Return the available location information for a retrieved source.
def get_result_location(metadata: dict):
    for field in ["page", "slide", "cell_index", "sheet_name"]:
        value = metadata.get(field)

        if value is not None and value != "":
            return value

    return "N/A"


# Return the strongest FAISS and reranker scores.
def get_retrieval_scores(results):
    if not results:
        return 0.0, 0.0

    top_faiss_score = max(float(result.get("score", 0)) for result in results)
    top_reranker_score = max(float(result.get("rerank_score", 0)) for result in results)

    return top_faiss_score, top_reranker_score


# Display the files used to generate the answer.
def display_sources(results):
    with st.expander("📚 View Sources"):
        for result in results:
            metadata = result.get("metadata", {})

            filename = metadata.get("filename", "Unknown")
            doc_type = metadata.get("doc_type", "unknown")
            location = get_result_location(metadata)

            st.write(f"• {filename} | type: {doc_type} | location: {location}")


# Display the retrieved document passages supporting the answer.
def display_evidence(results):
    with st.expander("🔍 View Supporting Evidence"):
        for position, result in enumerate(results, start=1):
            st.markdown(f"### Evidence {position}")
            st.write(result.get("text", "")[:1000])


index, records, embedding_model, reranker = load_system()


header_column, button_column = st.columns([6, 1])

with header_column:
    st.title("🤖 AI/ML Course Copilot")
    st.caption(
        "Ask questions about your indexed AI and machine-learning "
        "course materials."
    )

with button_column:
    st.write("")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []


if not st.session_state.messages:
    st.info(
        "Ask a question about concepts, models, algorithms, "
        "course notes, presentations, notebooks, or other "
        "indexed learning materials."
    )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


query = st.chat_input("Ask a question about your course documents...")


if query:
    normalized_query = normalize_query(query)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Searching the course materials..."):
        # Retrieve broad FAISS candidates before applying precise reranking.
        query_embedding = embed_query(embedding_model, normalized_query)

        candidate_results = search_faiss(
            index,
            records,
            query_embedding,
            top_k=CANDIDATE_COUNT,
        )

        results = rerank_results(
            normalized_query,
            candidate_results,
            reranker,
            top_k=FINAL_RESULT_COUNT,
        )

        top_faiss_score, top_reranker_score = get_retrieval_scores(results)

        # Allow generation when either retrieval method finds strong evidence.
        can_answer = bool(results) and (
            top_faiss_score >= FAISS_THRESHOLD
            or top_reranker_score >= RERANKER_THRESHOLD
        )

        if can_answer:
            answer = generate_answer(normalized_query, results)
            is_no_answer = "no reliable answer found" in answer.lower()
        else:
            answer = NO_ANSWER_MESSAGE
            is_no_answer = True

    with st.chat_message("assistant"):
        if is_no_answer:
            st.warning("No reliable answer found in the indexed documents.")
            st.write(
                "Try rephrasing the question or asking about "
                "content covered in the uploaded course materials."
            )
        else:
            st.markdown(answer)
            display_sources(results)
            display_evidence(results)

    if is_no_answer:
        log_failure(query, top_faiss_score)
        assistant_message = NO_ANSWER_MESSAGE
    else:
        log_success(query, top_faiss_score)
        assistant_message = answer

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_message,
        }
    )