import unittest
from unittest.mock import patch

from strategic_graphrag.schema.manager import SchemaManager


class _Record:
    def __init__(self, payload):
        self.payload = payload

    def data(self):
        return self.payload


class _Session:
    def __init__(self, *, error=None, rows=None):
        self.error = error
        self.rows = rows or []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def run(self, query, **params):
        if self.error is not None:
            raise self.error
        return [_Record(row) for row in self.rows]


class _Driver:
    def __init__(self, *, session_error=None, rows=None):
        self.session_error = session_error
        self.rows = rows
        self.closed = False

    def verify_connectivity(self):
        return None

    def session(self, **kwargs):
        return _Session(error=self.session_error, rows=self.rows)

    def close(self):
        self.closed = True


class SchemaManagerResilienceTests(unittest.TestCase):
    def test_read_replaces_stale_driver_once(self):
        stale = _Driver(session_error=OSError("defunct connection"))
        replacement = _Driver(rows=[{"ok": 1}])
        manager = SchemaManager()
        manager.driver = stale

        with patch(
            "strategic_graphrag.schema.manager.GraphDatabase.driver",
            return_value=replacement,
        ) as factory:
            rows = manager._read("RETURN 1 AS ok")

        self.assertEqual(rows, [{"ok": 1}])
        self.assertTrue(stale.closed)
        self.assertIs(manager.driver, replacement)
        factory.assert_called_once()

    def test_connect_cleans_up_failed_replacement(self):
        failed = _Driver()
        failed.verify_connectivity = lambda: (_ for _ in ()).throw(
            OSError("database waking")
        )
        manager = SchemaManager()

        with patch(
            "strategic_graphrag.schema.manager.GraphDatabase.driver",
            return_value=failed,
        ):
            self.assertFalse(manager.connect())

        self.assertIsNone(manager.driver)
        self.assertTrue(failed.closed)


if __name__ == "__main__":
    unittest.main()
