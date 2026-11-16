from src.retrieve import retrieve
from src.llm import generate_answer


def build_context(results):

    context_parts = []

    for result in results:
        context_parts.append(
            result["chunk"]
        )

    return "\n\n".join(context_parts)


def ask(question, model, top_k=3):

    # 1. Retrieve
    results = retrieve(
        question,
        model,
        top_k=top_k
    )

    # 2. No relevant information
    if not results:
        return (
            "I don't have enough information "
            "in the provided documents.",
            []
        )

    # 3. Build context
    context = build_context(results)

    # 4. Send context + question to LLM
    answer = generate_answer(
        question,
        context
    )

    return answer, results