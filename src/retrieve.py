import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.vector_store import load_vector_store


def retrieve(
    query,
    model,
    top_k=3,
min_score=0.25):
    chunks, embeddings, sources = load_vector_store()

    query_embedding = model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        score = float(similarities[index])

        if score >= min_score:

            results.append({
                "chunk": chunks[index],
                "score": score,
                "source": sources[index]
            })

    return results