import json
from pathlib import Path
import numpy as np

VECTOR_STORE_PATH = Path("data/vector_store.json")


def save_vector_store(chunks, embeddings, sources, path=VECTOR_STORE_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"version": 2, "chunks": chunks, "embeddings": np.asarray(embeddings).tolist(), "sources": sources}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_vector_store(path=VECTOR_STORE_PATH):
    path = Path(path)
    if not path.exists():
        return [], np.empty((0, 0), dtype=np.float32), []
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks, embeddings, sources = data.get("chunks", []), data.get("embeddings", []), data.get("sources", [])
    if not (len(chunks) == len(embeddings) == len(sources)):
        raise ValueError(f"Corrupt vector store: chunks, embeddings and sources differ in length ({path})")
    array = np.asarray(embeddings, dtype=np.float32)
    if array.size == 0:
        array = np.empty((0, 0), dtype=np.float32)
    return chunks, array, sources
