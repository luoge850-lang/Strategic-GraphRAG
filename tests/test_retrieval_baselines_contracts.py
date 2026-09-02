import unittest

from scripts.run_retrieval_baselines import _retrieved_contexts


class RetrievalBaselineContractTests(unittest.TestCase):
    def test_vector_contexts_keep_chunk_identity(self):
        result = {
            "metadata": {
                "retrieval": {
                    "hits": [
                        {"metadata": {"chunk_id": "2025:12:0", "page": 12, "doc_id": "2025-10-K.pdf"}}
                    ]
                }
            }
        }
        contexts = _retrieved_contexts("vector", result)

        self.assertEqual(contexts["unit_type"], "chunk_id")
        self.assertEqual(contexts["contexts"][0]["id"], "2025:12:0")

    def test_graph_contexts_keep_evidence_claim_identity(self):
        result = {
            "paths": [
                {
                    "evidence_ids": ["claim_v2_abc"],
                    "pages": [52],
                    "filings": ["2025-10-K.pdf"],
                }
            ]
        }
        contexts = _retrieved_contexts("graph", result)

        self.assertEqual(contexts["unit_type"], "evidence_claim_id")
        self.assertEqual(contexts["contexts"][0]["id"], "claim_v2_abc")
