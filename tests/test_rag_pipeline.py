import core.rag_pipeline as rag_pipeline


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self):
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        return FakeResponse("Transformers use attention mechanisms.")


# Verify that the prompt contains the question and retrieved context.
def test_build_prompt():
    context_chunks = [
        {
            "text": "Transformers use attention mechanisms to process sequences."
        }
    ]

    prompt = rag_pipeline.build_prompt(
        "What are Transformers?",
        context_chunks,
    )

    assert "What are Transformers?" in prompt
    assert "Transformers use attention mechanisms" in prompt
    assert "Do not use outside knowledge" in prompt
    assert rag_pipeline.NO_ANSWER_TEXT in prompt


# Verify answer generation without making a real API request.
def test_generate_answer():
    fake_llm = FakeLLM()
    original_llm = rag_pipeline.llm
    rag_pipeline.llm = fake_llm

    try:
        context_chunks = [
            {
                "text": "Transformers use attention mechanisms to process sequences."
            }
        ]

        answer = rag_pipeline.generate_answer(
            "What are Transformers?",
            context_chunks,
        )

        assert answer == "Transformers use attention mechanisms."
        assert fake_llm.last_prompt is not None
        assert "What are Transformers?" in fake_llm.last_prompt
    finally:
        rag_pipeline.llm = original_llm


# Verify that missing context returns the standard refusal message.
def test_no_context_answer():
    answer = rag_pipeline.generate_answer(
        "What is the latest stock price?",
        [],
    )

    assert answer == rag_pipeline.NO_ANSWER_TEXT


if __name__ == "__main__":
    test_build_prompt()
    test_generate_answer()
    test_no_context_answer()

    print("RAG PIPELINE TESTS PASSED")
