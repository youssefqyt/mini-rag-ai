import unittest
from pathlib import Path
import numpy as np

from src.chunking import create_chunks
from src.vector_store import load_vector_store, save_vector_store
from src.retrieve import retrieve


class FakeModel:
    def encode(self, texts):
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(__file__).parent.parent / "test-index.tmp.json"

    def tearDown(self):
        self.path.unlink(missing_ok=True)
    def test_chunk_overlap_and_empty_input(self):
        text = "A sentence. " * 100
        chunks = create_chunks(text, 100, 20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunks[i][-15:] in chunks[i + 1] for i in range(len(chunks) - 1)))
        self.assertEqual(create_chunks("  "), [])

    def test_chunk_configuration_validation(self):
        with self.assertRaises(ValueError): create_chunks("text", 10, 10)

    def test_store_round_trip_and_missing_store(self):
        path = self.path
        self.assertEqual(load_vector_store(path)[0], [])
        save_vector_store(["alpha"], [[1, 0]], [{"name":"a.txt", "page":None}], path)
        chunks, vectors, sources = load_vector_store(path)
        self.assertEqual(chunks, ["alpha"])
        self.assertEqual(vectors.shape, (1, 2))
        self.assertEqual(sources[0]["name"], "a.txt")

    def test_hybrid_retrieval_and_source_metadata(self):
        path = self.path
        save_vector_store(["The keynote speaker is Alice."], [[1, 0]], [{"name":"agenda.txt", "page":2}], path)
        results = retrieve("Who is the keynote speaker?", FakeModel(), min_score=0, store_path=path)
        self.assertEqual(results[0]["source"], "agenda.txt")
        self.assertEqual(results[0]["page"], 2)


if __name__ == "__main__": unittest.main()
