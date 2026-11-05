import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from docx import Document

from rag import RAG, RAGError, Store, normalize


class FakeModels:
    def __init__(self):
        self.embed_calls = 0
        self.fail = False

    def embed_content(self, **kwargs):
        self.embed_calls += 1
        embeddings = [] if self.fail else [
            SimpleNamespace(values=[1.0] + [0.0] * 767) for _ in kwargs['contents']
        ]
        return SimpleNamespace(embeddings=embeddings)

    def generate_content(self, **kwargs):
        return SimpleNamespace(text='Employees get 20 days of leave [1].')


class RAGTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = self.root / '.env'
        self.env.write_text('GEMINI_API_KEY=fake\nRAG_PROMPT="You are an HR assistant."\n'
                            f'DATABASE_PATH={self.root.as_posix()}/rag.db\n')
        self.models = FakeModels()
        self.rag = RAG(env_file=self.env, client=SimpleNamespace(models=self.models))
        self.addCleanup(self.rag.close)
        document = Document()
        document.add_paragraph('Employees get 20 days of leave.')
        stream = BytesIO()
        document.save(stream)
        self.document = self.root / 'policy.docx'
        self.document.write_bytes(stream.getvalue())

    def test_ingest_query_duplicate_persistence_and_delete(self):
        self.assertIn('HR assistant', self.rag.prompt)
        result = self.rag.ingest(self.document)
        self.assertTrue(self.rag.ingest(self.document)['duplicate'])
        self.assertEqual(self.models.embed_calls, 1)
        answer = self.rag.ask('How much leave?')
        self.assertEqual(answer['sources'][0]['filename'], 'policy.docx')
        self.assertIn('20 days', answer['sources'][0]['text'])
        reopened = RAG(env_file=self.env, client=SimpleNamespace(models=self.models))
        self.assertEqual(len(reopened.list_documents()), 1)
        self.assertTrue(reopened.delete(result['document_id']))
        self.assertEqual(self.rag.list_documents(), [])
        reopened.close()

    def test_empty_and_invalid_queries(self):
        self.assertEqual(self.rag.ask('Leave?')['sources'], [])
        self.assertEqual(self.models.embed_calls, 0)
        for question, top_k in [(' ', 5), ('Leave?', 0), ('a' * 8001, 5)]:
            with self.subTest(question=question[:10], top_k=top_k):
                with self.assertRaises(ValueError):
                    self.rag.ask(question, top_k=top_k)

    def test_failed_embedding_does_not_commit(self):
        self.models.fail = True
        with self.assertRaises(RAGError):
            self.rag.ingest(self.document)
        self.assertEqual(self.rag.list_documents(), [])

    def test_invalid_document(self):
        with self.assertRaises(ValueError):
            self.rag.ingest_bytes(b'legacy Word data', 'old.doc')

    def test_ranking_threshold_and_project_isolation(self):
        store = Store(self.root / 'vectors.db', 'one')
        store.put('doc', 'policy.pdf', [
            {'text': 'first', 'page': 1, 'vector': normalize([1, 0])},
            {'text': 'second', 'page': 2, 'vector': normalize([0, 1])},
        ])
        matches = store.search([1, 0], 5, 0.5)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]['text'], 'first')
        self.assertEqual(Store(self.root / 'vectors.db', 'two').search([1, 0], 5, 0), [])


if __name__ == '__main__':
    unittest.main()
