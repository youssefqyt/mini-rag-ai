"""Local single-user web app and JSON API for the Mini RAG knowledge base."""
import base64, json, mimetypes, sqlite3, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from src.build_index import DOCUMENTS_DIR, SUPPORTED, build_index
from src.embeddings import create_embedding_model
from src.rag import ask

ROOT = Path(__file__).parent
DB = ROOT / "data" / "app.sqlite3"
MAX_UPLOAD = 15 * 1024 * 1024
_model = None
_lock = threading.Lock()


def db():
    DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, role TEXT NOT NULL, content TEXT NOT NULL, sources TEXT NOT NULL DEFAULT '[]', created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
    return conn


def get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None: _model = create_embedding_model()
    return _model


def documents():
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    return [{"name": p.name, "size": p.stat().st_size, "type": p.suffix[1:].lower()} for p in sorted(DOCUMENTS_DIR.iterdir()) if p.is_file() and p.suffix.lower() in SUPPORTED]


class Handler(BaseHTTPRequestHandler):
    def send_json(self, value, status=200):
        raw = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)

    def body(self):
        size = int(self.headers.get("Content-Length", 0))
        if size > MAX_UPLOAD + 1024 * 1024: raise ValueError("Request is too large")
        return json.loads(self.rfile.read(size) or b"{}")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/status":
            from src.vector_store import load_vector_store
            chunks, _, _ = load_vector_store()
            return self.send_json({"status": "ok", "documents": len(documents()), "chunks": len(chunks), "model": "Ollama (local)"})
        if path == "/api/documents": return self.send_json(documents())
        if path == "/api/history":
            with db() as conn: rows = conn.execute("SELECT role, content, sources, created_at FROM messages ORDER BY id").fetchall()
            return self.send_json([{"role": r[0], "content": r[1], "sources": json.loads(r[2]), "created_at": r[3]} for r in rows])
        if path == "/api/openapi.json":
            return self.send_json({"openapi":"3.0.0","info":{"title":"Mini RAG API","version":"1.0.0"},"paths":{"/api/status":{"get":{"summary":"Health and index status"}},"/api/documents":{"get":{"summary":"List documents"}},"/api/upload":{"post":{"summary":"Upload and index a document"}},"/api/chat":{"post":{"summary":"Ask a grounded question"}},"/api/documents/{name}":{"delete":{"summary":"Delete and reindex a document"}}}})
        if path == "/" or path == "/index.html":
            raw = (ROOT / "web" / "index.html").read_bytes(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(raw))); self.end_headers(); return self.wfile.write(raw)
        return self.send_json({"error":"Not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            payload = self.body()
            if path == "/api/upload":
                name = Path(str(payload.get("name", ""))).name
                suffix = Path(name).suffix.lower()
                if not name or suffix not in SUPPORTED: return self.send_json({"error":"Supported files: PDF, DOCX, TXT, Markdown"}, 400)
                content = base64.b64decode(payload.get("content", ""), validate=True)
                if not content or len(content) > MAX_UPLOAD: return self.send_json({"error":"File is empty or exceeds 15 MB"}, 413)
                if suffix == ".pdf" and not content.startswith(b"%PDF-"): return self.send_json({"error":"Invalid PDF file"}, 400)
                if suffix == ".docx" and not content.startswith(b"PK"): return self.send_json({"error":"Invalid DOCX file"}, 400)
                DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
                target = DOCUMENTS_DIR / name
                target.write_bytes(content)
                try: count = build_index()
                except Exception:
                    target.unlink(missing_ok=True); raise
                return self.send_json({"ok":True,"name":name,"chunks":count}, 201)
            if path == "/api/chat":
                question = str(payload.get("question", "")).strip()
                if not question or len(question) > 4000: return self.send_json({"error":"Question must be between 1 and 4000 characters"}, 400)
                with db() as conn: history = [{"user":r[0],"assistant":r[1]} for r in conn.execute("SELECT u.content, a.content FROM messages u JOIN messages a ON a.id = u.id + 1 WHERE u.role='user' AND a.role='assistant' ORDER BY u.id DESC LIMIT 4").fetchall()][::-1]
                answer, sources = ask(question, get_model(), history=history)
                with db() as conn:
                    conn.execute("INSERT INTO messages(role,content,sources) VALUES('user',?,'[]')", (question,))
                    conn.execute("INSERT INTO messages(role,content,sources) VALUES('assistant',?,?)", (answer, json.dumps(sources, ensure_ascii=False)))
                return self.send_json({"answer":answer,"sources":sources})
            return self.send_json({"error":"Not found"}, 404)
        except (ValueError, json.JSONDecodeError) as exc: return self.send_json({"error":str(exc)}, 400)
        except Exception as exc: return self.send_json({"error":str(exc)}, 500)

    def do_DELETE(self):
        path = urlparse(self.path).path
        if path == "/api/history":
            with db() as conn: conn.execute("DELETE FROM messages")
            return self.send_json({"ok":True})
        if path.startswith("/api/documents/"):
            name = Path(unquote(path.rsplit("/",1)[-1])).name
            target = DOCUMENTS_DIR / name
            if not target.is_file() or target.suffix.lower() not in SUPPORTED: return self.send_json({"error":"Document not found"}, 404)
            target.unlink()
            try:
                if documents(): build_index()
                else: from src.vector_store import save_vector_store; save_vector_store([], [], [])
            except Exception as exc: return self.send_json({"error":str(exc)}, 500)
            return self.send_json({"ok":True})
        return self.send_json({"error":"Not found"}, 404)


if __name__ == "__main__":
    print("Mini RAG web app: http://127.0.0.1:8000 (local single-user access)")
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
