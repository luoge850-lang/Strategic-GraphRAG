from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import inventory_rebuildable_artifacts as inventory_script


class InventoryRebuildableArtifactsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "project"
        self.root.mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write(self, relative_path: str, content: bytes) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_classifies_candidates_and_protected_material_and_counts_logical_sizes(self) -> None:
        self._write("__pycache__/module.pyc", b"abc")
        self._write("frontend/node_modules/.vite/cache.bin", b"12345")
        self._write(".venv/lib/package.py", b"env!")
        self._write("archive/legacy-venv-2026-09-02/site.py", b"old")
        self._write("reports/isolatedstaging/venv/site.py", b"staged")
        self._write("reports/__pycache__/module.pyc", b"keep-cache")
        self._write("documents/source.pdf", b"pdf-bytes")
        self._write(".env", b"secret-value-must-not-appear")
        self._write("requirements-lock-2026.txt", b"lock")
        self._write("unknown-purpose.bin", b"unknown")

        result = inventory_script.build_inventory(self.root)
        files = {Path(item["path"]).relative_to(self.root).as_posix(): item for item in result["files"]}
        directories = {item["name"]: item for item in result["top_level_directories"]}

        self.assertEqual(files["__pycache__/module.pyc"]["classification"], "rebuildable_cache")
        self.assertEqual(files["frontend/node_modules/.vite/cache.bin"]["classification"], "rebuildable_cache")
        self.assertEqual(files[".venv/lib/package.py"]["classification"], "rebuildable_environment")
        self.assertEqual(
            files["archive/legacy-venv-2026-09-02/site.py"]["classification"],
            "rebuildable_environment",
        )
        self.assertEqual(files["reports/isolatedstaging/venv/site.py"]["classification"], "rebuildable_environment")
        self.assertEqual(files["reports/__pycache__/module.pyc"]["classification"], "protected_material")
        self.assertEqual(files["documents/source.pdf"]["classification"], "protected_material")
        self.assertIn("PDF", files["documents/source.pdf"]["reason"])
        self.assertEqual(files[".env"]["classification"], "protected_material")
        self.assertIn("不读取内容", files[".env"]["reason"])
        self.assertEqual(files["requirements-lock-2026.txt"]["classification"], "protected_material")
        self.assertFalse(files["unknown-purpose.bin"]["candidate"])
        self.assertIn("不因未发现引用", files["unknown-purpose.bin"]["reason"])
        self.assertNotIn("secret-value-must-not-appear", json.dumps(result, ensure_ascii=False))

        self.assertEqual(directories["__pycache__"]["logical_size_bytes"], 3)
        self.assertEqual(directories["__pycache__"]["file_count"], 1)
        self.assertEqual(directories["frontend"]["logical_size_bytes"], 5)
        self.assertEqual(directories["archive"]["logical_size_bytes"], 3)
        self.assertEqual(directories["reports"]["logical_size_bytes"], 16)
        self.assertTrue(result["scanned_at"].endswith("Z"))
        self.assertIsInstance(result["disk_free_bytes_before_scan"], int)
        self.assertTrue(Path(result["root"]).is_absolute())
        recorded_paths = [item["path"] for item in result["top_level_directories"]]
        recorded_paths.extend(item["path"] for item in result["candidate_directories"])
        recorded_paths.extend(item["path"] for item in result["files"])
        recorded_paths.extend(item["path"] for item in result["reparse_points"])
        recorded_paths.extend(item["path"] for item in result["special_entries"])
        recorded_paths.extend(item["path"] for item in result["errors"])
        for path in recorded_paths:
            self.assertTrue(os.path.isabs(path), path)
            self.assertEqual(path, os.path.normpath(path))

    def test_does_not_treat_other_archive_paths_as_environments(self) -> None:
        self._write("archive/legacyvenv/site.py", b"preserve")

        result = inventory_script.build_inventory(self.root)

        file_record = next(item for item in result["files"] if item["path"].endswith("site.py"))
        self.assertEqual(file_record["classification"], "protected_material")
        self.assertEqual(result["candidate_directories"], [])

    def test_records_link_without_traversing_its_target(self) -> None:
        external = Path(self.temporary_directory.name) / "outside"
        external.mkdir()
        sentinel = external / "must-not-be-inventoried.txt"
        sentinel.write_text("outside", encoding="utf-8")
        link = self.root / "linked-directory"
        expected_kind = "symlink"
        try:
            if os.name == "nt":
                expected_kind = "junction"
                completed = subprocess.run(
                    ["cmd.exe", "/c", "mklink", "/J", str(link), str(external)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if completed.returncode:
                    self.skipTest(f"junction creation is unavailable: {completed.stderr or completed.stdout}")
            else:
                os.symlink(external, link, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"directory link creation is unavailable: {error}")

        result = inventory_script.build_inventory(self.root)

        self.assertEqual(len(result["reparse_points"]), 1)
        self.assertEqual(result["reparse_points"][0]["path"], os.path.normpath(str(link.absolute())))
        self.assertEqual(result["reparse_points"][0]["kind"], expected_kind)
        listed_paths = {item["path"] for item in result["files"]}
        self.assertNotIn(os.path.normpath(str(sentinel.absolute())), listed_paths)

    def test_records_directory_read_error(self) -> None:
        blocked = self.root / "unreadable"
        blocked.mkdir()
        original_scandir = os.scandir
        blocked_path = os.path.normcase(os.path.normpath(str(blocked.absolute())))

        def scandir_with_denial(path: os.PathLike[str] | str):
            if os.path.normcase(os.path.normpath(os.fspath(path))) == blocked_path:
                raise PermissionError("test access denied")
            return original_scandir(path)

        with mock.patch.object(inventory_script.os, "scandir", side_effect=scandir_with_denial):
            result = inventory_script.build_inventory(self.root)

        matching = [item for item in result["errors"] if item["path"] == os.path.normpath(str(blocked.absolute()))]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["operation"], "scandir")
        unreadable = next(item for item in result["top_level_directories"] if item["name"] == "unreadable")
        self.assertFalse(unreadable["complete"])

    def test_non_directory_root_is_rejected(self) -> None:
        input_file = Path(self.temporary_directory.name) / "not-a-directory.txt"
        input_file.write_text("plain file", encoding="utf-8")

        with self.assertRaises(NotADirectoryError):
            inventory_script.build_inventory(input_file)

    def test_cli_writes_new_json_without_overwriting_existing_output(self) -> None:
        self._write("__pycache__/one.pyc", b"x")
        output = Path(self.temporary_directory.name) / "inventory.json"

        self.assertEqual(
            inventory_script.main(["--root", str(self.root), "--output", str(output)]),
            0,
        )
        saved = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(saved["inventory_path"], os.path.normpath(str(output.absolute())))
        self.assertNotIn(saved["inventory_path"], {item["path"] for item in saved["files"]})

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            status = inventory_script.main(["--root", str(self.root), "--output", str(output)])
        self.assertEqual(status, 2)
        self.assertIn("refusing to overwrite", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
