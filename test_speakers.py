from src.embeddings import create_embedding_model
from src.retrieve import retrieve


model = create_embedding_model()

query = "Who are the keynote speakers?"

results = retrieve(
    query,
    model,
    top_k=5,
    min_score=0.0
)

print("\nQUESTION:")
print(query)

print("\nRESULTS:")

for i, result in enumerate(results):

    print(f"\n--- RESULT {i + 1} ---")
    print("Score:", result["score"])
    print("Source:", result["source"])
    print("Text:")
    print(result["chunk"])