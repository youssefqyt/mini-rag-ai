from pathlib import Path
from pypdf import PdfReader
from src.chunking import create_chunks
from src.embeddings import create_embedding_model, create_embeddings
from src.vector_store import save_vector_store

DOCUMENTS_DIR = Path("data/documents")
SUPPORTED = {".txt", ".md", ".markdown", ".pdf", ".docx"}


def extract_documents(directory=DOCUMENTS_DIR):
    documents = []
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        try:
            if path.suffix.lower() == ".pdf":
                for page_no, page in enumerate(PdfReader(str(path)).pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():
                        documents.append({"text": text, "source": {"name": path.name, "page": page_no, "type": "pdf"}})
            elif path.suffix.lower() == ".docx":
                from docx import Document
                text = "\n".join(p.text for p in Document(str(path)).paragraphs)
                if text.strip(): documents.append({"text": text, "source": {"name": path.name, "page": None, "type": "docx"}})
            else:
                text = path.read_text(encoding="utf-8-sig", errors="replace")
                if text.strip(): documents.append({"text": text, "source": {"name": path.name, "page": None, "type": path.suffix[1:]}})
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")
    return documents


def build_index(directory=DOCUMENTS_DIR, output=None, chunk_size=900, overlap=120):
    entries = []
    for doc in extract_documents(directory):
        for n, chunk in enumerate(create_chunks(doc["text"], chunk_size, overlap)):
            entries.append((chunk, {**doc["source"], "chunk": n + 1}))
    if not entries:
        raise ValueError(f"No readable documents found in {directory}")
    model = create_embedding_model()
    vectors = create_embeddings(model, [entry[0] for entry in entries])
    save_vector_store([e[0] for e in entries], vectors, [e[1] for e in entries], output) if output else save_vector_store([e[0] for e in entries], vectors, [e[1] for e in entries])
    return len(entries)


if __name__ == "__main__":
    print(f"Indexed {build_index()} chunks")
