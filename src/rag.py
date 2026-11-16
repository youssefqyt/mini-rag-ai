from src.retrieve import retrieve
from src.llm import generate_answer


def build_context(results):
    return "\n\n".join(f"[{i}] {r['source']}" + (f", page {r['page']}" if r.get("page") else "") + f"\n{r['chunk']}" for i, r in enumerate(results, 1))


def ask(question, model, top_k=5, history=None, min_score=0.12):
    results = retrieve(question, model, top_k=top_k, min_score=min_score)
    if not results:
        return "I couldn't find sufficiently relevant information in the indexed documents.", []
    return generate_answer(question, build_context(results), history), results
