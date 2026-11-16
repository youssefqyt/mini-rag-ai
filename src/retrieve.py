import numpy as np
import math
import re
from collections import Counter
from src.vector_store import load_vector_store


def retrieve(query, model, top_k=5, min_score=0.12, source_filter=None, store_path=None):
    chunks, embeddings, sources = load_vector_store(store_path) if store_path else load_vector_store()
    if not chunks:
        return []
    indices = [i for i, source in enumerate(sources) if not source_filter or source.get("name") == source_filter]
    if not indices:
        return []
    texts = [chunks[i] for i in indices]
    dense = np.zeros(len(indices), dtype=np.float32)
    if embeddings.ndim == 2 and embeddings.shape[0] == len(chunks) and embeddings.shape[1]:
        q = np.asarray(model.encode([query]), dtype=np.float32)
        q = q[0]
        candidates = embeddings[indices]
        denom = np.linalg.norm(candidates, axis=1) * max(float(np.linalg.norm(q)), 1e-12)
        dense = np.divide(candidates @ q, denom, out=np.zeros(len(indices), dtype=np.float32), where=denom > 0)
    # Lightweight TF-IDF with word unigrams and bigrams keeps query-time ranking self-contained.
    def terms(value):
        words = re.findall(r"[a-z0-9]+", value.lower())
        return words + [f"{a} {b}" for a, b in zip(words, words[1:])]
    token_docs = [terms(text) for text in texts]
    query_terms = set(terms(query))
    df = {}
    for tokens in token_docs:
        for token in set(tokens): df[token] = df.get(token, 0) + 1
    qweights = {t: math.log((1 + len(texts)) / (1 + df.get(t, 0))) + 1 for t in query_terms if df.get(t, 0)}
    qnorm = math.sqrt(sum(w*w for w in qweights.values())) or 1
    lexical = []
    for tokens in token_docs:
        counts = Counter(tokens)
        weights = {}
        for token, count in counts.items():
            if token in qweights: weights[token] = (1 + math.log(count)) * qweights[token]
        denom = math.sqrt(sum(w*w for w in weights.values())) * qnorm or 1
        lexical.append(sum(weights.get(t, 0) * w for t, w in qweights.items()) / denom)
    lexical = np.asarray(lexical, dtype=np.float32)
    # Blend semantic and lexical relevance, then normalize as an interpretable confidence signal.
    scores = 0.75 * dense + 0.25 * lexical
    order = np.argsort(scores)[::-1][:max(1, top_k)]
    results = []
    for pos in order:
        score = float(scores[pos])
        if score >= min_score:
            source = sources[indices[pos]]
            if isinstance(source, str):
                source = {"name": source, "page": None}
            results.append({"chunk": texts[pos], "score": score, "source": source.get("name", "Unknown"), "page": source.get("page"), "metadata": source})
    return results
