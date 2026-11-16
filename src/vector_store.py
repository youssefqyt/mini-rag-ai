import json
from pathlib import Path

import numpy as np


VECTOR_STORE_PATH = Path(
    "data/vector_store.json"
)


def save_vector_store(
    chunks,
    embeddings,
    sources
):
    VECTOR_STORE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data = {
        "chunks": chunks,
        "embeddings": embeddings.tolist(),
        "sources": sources
    }

    with open(
        VECTOR_STORE_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False
        )

    print(
        f"Vector store saved to: "
        f"{VECTOR_STORE_PATH}"
    )


def load_vector_store():

    with open(
        VECTOR_STORE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    chunks = data["chunks"]

    embeddings = np.array(
        data["embeddings"]
    )

    sources = data["sources"]

    return (
        chunks,
        embeddings,
        sources
    )