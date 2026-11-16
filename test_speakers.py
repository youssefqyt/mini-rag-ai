"""Manual retrieval smoke check against the current local index."""
from src.embeddings import create_embedding_model
from src.retrieve import retrieve


if __name__ == "__main__":
    question = "Who are the keynote speakers?"
    results = retrieve(question, create_embedding_model(), top_k=5, min_score=0.0)
    print(f"QUESTION: {question}")
    for rank, result in enumerate(results, 1):
        location = result["source"]
        if result.get("page"):
            location += f", page {result['page']}"
        print(f"\n--- RESULT {rank} | score {result['score']:.3f} | {location} ---")
        print(result["chunk"])
