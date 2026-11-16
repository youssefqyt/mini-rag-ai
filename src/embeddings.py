import os


MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def create_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install project dependencies with `pip install -r requirements.txt` to use embeddings") from exc
    return SentenceTransformer(MODEL_NAME)


def create_embeddings(model, texts):
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=len(texts) > 16)
