import unittest
from unittest.mock import patch

from strategic_graphrag.pipeline import ingestor as ingestor_module
from strategic_graphrag.pipeline.ingestor import GraphIngestor
from strategic_graphrag.pipeline.pipeline import KnowledgeGraphPipeline


class _FakeDriver:
    def __init__(self, alive=True):
        self.alive = alive
        self.closed = False
        self.verify_calls = 0

    def verify_connectivity(self):
        self.verify_calls += 1
        if not self.alive:
            raise RuntimeError("stale driver")

    def close(self):
        self.closed = True


class _FailingCommitIngestor:
    def __init__(self):
        self.replace_calls = 0

    def ensure_connection(self):
        return False

    def replace_filing(self, _filename):
        self.replace_calls += 1


class IngestorReconnectTests(unittest.TestCase):
    def test_live_driver_is_reused(self):
        ingestor = GraphIngestor(uri="bolt://test", user="neo4j", password="password")
        live_driver = _FakeDriver(alive=True)
        ingestor.driver = live_driver

        with patch.object(ingestor_module.GraphDatabase, "driver") as driver_factory:
            self.assertTrue(ingestor.ensure_connection())

        self.assertIs(ingestor.driver, live_driver)
        self.assertFalse(live_driver.closed)
        driver_factory.assert_not_called()

    def test_dead_driver_is_closed_and_reconnected(self):
        ingestor = GraphIngestor(uri="bolt://test", user="neo4j", password="password")
        stale_driver = _FakeDriver(alive=False)
        fresh_driver = _FakeDriver(alive=True)
        ingestor.driver = stale_driver

        with patch.object(
            ingestor_module.GraphDatabase,
            "driver",
            return_value=fresh_driver,
        ) as driver_factory:
            self.assertTrue(ingestor.ensure_connection())

        self.assertTrue(stale_driver.closed)
        self.assertIs(ingestor.driver, fresh_driver)
        self.assertEqual(fresh_driver.verify_calls, 1)
        driver_factory.assert_called_once()

    def test_reconnect_failure_leaves_no_driver(self):
        ingestor = GraphIngestor(uri="bolt://test", user="neo4j", password="password")
        stale_driver = _FakeDriver(alive=False)
        ingestor.driver = stale_driver

        with patch.object(
            ingestor_module.GraphDatabase,
            "driver",
            side_effect=RuntimeError("routing unavailable"),
        ):
            self.assertFalse(ingestor.ensure_connection())

        self.assertTrue(stale_driver.closed)
        self.assertIsNone(ingestor.driver)

    def test_commit_guard_blocks_replace_when_connection_fails(self):
        pipeline = KnowledgeGraphPipeline.__new__(KnowledgeGraphPipeline)
        failing_ingestor = _FailingCommitIngestor()
        pipeline.ingestor = failing_ingestor

        with self.assertRaisesRegex(RuntimeError, "before filing commit"):
            pipeline._ensure_commit_connection()

        self.assertEqual(failing_ingestor.replace_calls, 0)


if __name__ == "__main__":
    unittest.main()
