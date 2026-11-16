from pathlib import Path

from pypdf import PdfReader

from chunking import create_chunks
from embeddings import (
    create_embedding_model,
    create_embeddings
)
from vector_store import save_vector_store


DOCUMENTS_DIR = Path("data/documents")


def load_documents():

    documents = []

    for file_path in DOCUMENTS_DIR.iterdir():

        # =========================
        # TXT
        # =========================
        if file_path.suffix.lower() == ".txt":

            print(f"Loading TXT: {file_path}")

            text = file_path.read_text(
                encoding="utf-8"
            )

            documents.append({
                "source": file_path.name,
                "text": text
            })

        # =========================
        # PDF
        # =========================
        elif file_path.suffix.lower() == ".pdf":

            print(f"Loading PDF: {file_path}")

            reader = PdfReader(file_path)

            for page_number, page in enumerate(
                reader.pages,
                start=1
            ):

                text = page.extract_text()

                if text:

                    documents.append({
                        "source": (
                            f"{file_path.name} "
                            f"— page {page_number}"
                        ),
                        "text": text
                    })

    return documents


def build_index():

    documents = load_documents()

    all_chunks = []
    all_sources = []

    for document in documents:

        chunks = create_chunks(
            document["text"],
            chunk_size=300
        )

        all_chunks.extend(chunks)

        all_sources.extend(
            [document["source"]] * len(chunks)
        )

    print(
        "Number of chunks:",
        len(all_chunks)
    )

    model = create_embedding_model()

    embeddings = create_embeddings(
        model,
        all_chunks
    )

    print(
        "Embeddings shape:",
        embeddings.shape
    )

    save_vector_store(
        all_chunks,
        embeddings,
        all_sources
    )


if __name__ == "__main__":
    build_index()