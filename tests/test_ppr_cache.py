import unittest

from strategic_graphrag.engine.retrieval import Neo4jPPRRetriever


class _FakeSession:
    def __init__(self, rows, calls):
        self.rows = rows
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, *_args, **_kwargs):
        self.calls["run"] += 1
        return list(self.rows)


class _FakeDriver:
    def __init__(self, rows):
        self.rows = rows
        self.calls = {"session": 0, "run": 0}

    def session(self):
        self.calls["session"] += 1
        return _FakeSession(self.rows, self.calls)


class PPRCacheTests(unittest.TestCase):
    def test_repeated_frozen_graph_lookup_uses_bounded_process_cache(self):
        driver = _FakeDriver([{"source_id": "A", "target_id": "B"}])
        retriever = Neo4jPPRRetriever(driver, cache_ttl_seconds=60, cache_max_entries=2)

        first = retriever.rank(["A"], limit=2)
        second = retriever.rank(["a"], limit=2)

        self.assertEqual(first, second)
        self.assertEqual(driver.calls["session"], 1)
        self.assertEqual(driver.calls["run"], 1)

    def test_cache_key_preserves_temporal_and_filing_scope(self):
        driver = _FakeDriver([{"source_id": "A", "target_id": "B"}])
        retriever = Neo4jPPRRetriever(driver, cache_ttl_seconds=60)

        retriever.rank(["A"], source_filing="2025-10-K.pdf", year_start=2025, year_end=2025)
        retriever.rank(["A"], source_filing="2024-10-K.pdf", year_start=2024, year_end=2024)

        self.assertEqual(driver.calls["session"], 2)


if __name__ == "__main__":
    unittest.main()
