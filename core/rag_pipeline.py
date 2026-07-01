from langchain_openai import ChatOpenAI


MODEL_NAME = "gpt-4o-mini"
NO_ANSWER_TEXT = "No reliable answer found in uploaded documents."


# Reuse one model client for all questions.
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)


# Build a prompt that restricts the model to the retrieved document context.
def build_prompt(query: str, context_chunks: list[dict]) -> str:
    context_text = "\n\n".join(chunk.get("text", "") for chunk in context_chunks)

    return f"""
You are a document-grounded AI assistant.

Answer the question using only the provided context.

Rules:
- Do not use outside knowledge.
- Do not make assumptions beyond the context.
- If the context does not clearly answer the question, respond exactly with:
  "{NO_ANSWER_TEXT}"
- Keep the answer clear and concise.

Context:
{context_text}

Question:
{query}

Answer:
""".strip()


# Generate an answer using only the retrieved document chunks.
def generate_answer(query: str, context_chunks: list[dict]) -> str:
    # Skip the API call when retrieval returns no supporting context.
    if not context_chunks:
        return NO_ANSWER_TEXT

    prompt = build_prompt(query, context_chunks)
    response = llm.invoke(prompt)

    return response.content.strip()
