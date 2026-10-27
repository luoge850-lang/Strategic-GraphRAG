#!/usr/bin/env python3
"""Create a read-only inventory of rebuildable candidates and protected files.

The scanner reads filesystem metadata only. It never opens file contents and
never descends into symbolic links, junctions, or other reparse points.
"""

from __future__ import annotations

import argparse
import datetime as dt
import errno
import json
import os
import shutil
import stat
import sys
from typing import Any, Iterable


_FILE_ATTRIBUTE_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_IO_REPARSE_TAG_MOUNT_POINT = 0xA0000003
_IO_REPARSE_TAG_SYMLINK = 0xA000000C

_CACHE_NAMES = {"__pycache__", ".pytest_cache"}
_PROTECTED_DIR_NAMES = {
    "annotation",
    "annotations",
    "human",
    "human_material",
    "human_materials",
    "label",
    "labels",
    "manual_material",
    "manual_materials",
    "handwritten",
    "report",
    "reports",
    "build",
    "builds",
    "dist",
    "release",
}
_LOCK_NAMES = {
    "cargo.lock",
    "composer.lock",
    "gemfile.lock",
    "go.sum",
    "npm-shrinkwrap.json",
    "package-lock.json",
    "pipfile.lock",
    "poetry.lock",
    "pnpm-lock.yaml",
    "uv.lock",
    "yarn.lock",
}
_PACKAGE_SUFFIXES = (
    ".whl",
    ".egg",
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".7z",
    ".rar",
)

_CANDIDATE_REASON = {
    "cache": "位于明确识别的工具缓存目录；仅作为可评估重建候选，不代表可以直接删除。",
    "environment": "位于明确识别的 Python 虚拟环境目录；重建前需核对 Python 版本、依赖锁文件及本地安装内容。",
}
_CANDIDATE_RECOVERY = {
    "cache": "先确认对应工具及构建流程，再使用该工具的标准命令重新生成；确认前保留原路径。",
    "environment": "依据已审核的依赖锁文件和 Python 版本，在隔离位置重建并验证；发现本地包或改动时先备份。",
}


def _absolute_normalized(path: os.PathLike[str] | str) -> str:
    """Return an absolute lexical path without resolving links."""
    return os.path.normpath(os.path.abspath(os.fspath(path)))


def _reparse_kind(metadata: os.stat_result) -> str | None:
    """Identify links and Windows reparse points using no-follow metadata."""
    tag = getattr(metadata, "st_reparse_tag", None)
    attributes = getattr(metadata, "st_file_attributes", 0)
    is_reparse = bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)

    if stat.S_ISLNK(metadata.st_mode) or tag == _IO_REPARSE_TAG_SYMLINK:
        return "symlink"
    if is_reparse and tag == _IO_REPARSE_TAG_MOUNT_POINT:
        return "junction"
    if is_reparse:
        return "reparse_point"
    return None


def _validate_real_directory(path: str) -> None:
    """Validate each path component without crossing a reparse point."""
    drive, _ = os.path.splitdrive(path)
    if drive:
        anchor = drive + os.sep
    else:
        anchor = os.sep

    relative = os.path.relpath(path, anchor)
    components = [] if relative == os.curdir else relative.split(os.sep)
    current = anchor
    final_metadata: os.stat_result | None = None

    for index, component in enumerate(components):
        current = os.path.join(current, component)
        metadata = os.lstat(current)
        kind = _reparse_kind(metadata)
        if kind:
            raise OSError(
                errno.ELOOP,
                f"path crosses a {kind}; refusing to follow it",
                current,
            )
        if index < len(components) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise NotADirectoryError(errno.ENOTDIR, "path component is not a directory", current)
        final_metadata = metadata

    if final_metadata is None:
        final_metadata = os.lstat(anchor)
    if not stat.S_ISDIR(final_metadata.st_mode):
        raise NotADirectoryError(errno.ENOTDIR, "--root is not a directory", path)


def _is_environment_path(parts: tuple[str, ...]) -> bool:
    folded = tuple(part.casefold() for part in parts)
    if any(part in {".venv", "venv"} for part in folded):
        return True
    for index in range(max(0, len(folded) - 1)):
        if folded[index : index + 2] == ("archive", "legacy-venv-2026-09-02"):
            return True
    return False


def _cache_kind(parts: tuple[str, ...]) -> str | None:
    folded = tuple(part.casefold() for part in parts)
    if any(part in _CACHE_NAMES for part in folded):
        return "cache"
    for index in range(max(0, len(folded) - 2)):
        if folded[index : index + 3] == ("frontend", "node_modules", ".vite"):
            return "cache"
    return None


def _protected_reason(parts: tuple[str, ...]) -> str | None:
    folded = tuple(part.casefold() for part in parts)
    filename = folded[-1] if folded else ""

    if ".git" in folded:
        return "Git 元数据受保护；扫描器不会建议移除。"
    if any(part == ".env" or part.startswith(".env.") for part in folded):
        return "环境变量文件或目录受保护；只读取路径和文件元数据，不读取内容。"
    if filename.endswith(".pdf"):
        return "PDF 原始材料受保护；不会根据引用情况推断可删除。"
    if filename in _LOCK_NAMES or filename.endswith(".lock") or "-lock-" in filename:
        return "依赖锁定文件受保护，因为它可能用于复现环境。"
    if any(filename.endswith(suffix) for suffix in _PACKAGE_SUFFIXES):
        return "构建包或归档文件受保护；用途和恢复成本需人工核对。"
    if any(part in _PROTECTED_DIR_NAMES for part in folded if part not in {"report", "reports"}):
        return "报告、构建输出或人工标注材料受保护。"
    if any(part in {"report", "reports"} for part in folded) and not _is_environment_path(parts):
        return "报告目录内容受保护。"
    return None


def _classify_file(parts: tuple[str, ...]) -> dict[str, Any]:
    protected = _protected_reason(parts)
    if protected:
        return {
            "classification": "protected_material",
            "candidate": False,
            "reason": protected,
            "recovery_suggestion": "保留原件；若需处置，先由材料负责人确认归属并制作独立备份。",
        }

    if _is_environment_path(parts):
        kind = "environment"
    else:
        kind = _cache_kind(parts)

    if kind:
        return {
            "classification": f"rebuildable_{kind}",
            "candidate": True,
            "reason": _CANDIDATE_REASON[kind],
            "recovery_suggestion": _CANDIDATE_RECOVERY[kind],
        }

    return {
        "classification": "protected_material",
        "candidate": False,
        "reason": "用途无法仅凭路径判断；默认受保护，不因未发现引用而推断无用。",
        "recovery_suggestion": "保留原件；先确认用途、归属和恢复方法，再由负责人决定后续处理。",
    }


def _classify_directory(parts: tuple[str, ...]) -> dict[str, Any] | None:
    if _protected_reason(parts):
        return None
    if _is_environment_path(parts):
        return {"kind": "environment", "classification": "rebuildable_environment"}
    if _cache_kind(parts):
        return {"kind": "cache", "classification": "rebuildable_cache"}
    return None


def _relative_parts(path: str, root: str) -> tuple[str, ...]:
    relative = os.path.relpath(path, root)
    if relative == os.curdir:
        return ()
    return tuple(relative.split(os.sep))


def _new_error(path: str, operation: str, error: OSError) -> dict[str, str]:
    return {
        "path": _absolute_normalized(path),
        "operation": operation,
        "error_type": type(error).__name__,
        "message": str(error),
    }


def build_inventory(root: os.PathLike[str] | str) -> dict[str, Any]:
    """Inventory a directory without opening files or following reparse points."""
    root_path = _absolute_normalized(root)
    _validate_real_directory(root_path)

    scanned_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    errors: list[dict[str, str]] = []
    try:
        free_bytes: int | None = shutil.disk_usage(root_path).free
    except OSError as error:
        free_bytes = None
        errors.append(_new_error(root_path, "disk_usage", error))

    files: list[dict[str, Any]] = []
    reparse_points: list[dict[str, str]] = []
    special_entries: list[dict[str, str]] = []
    top_directories: dict[str, dict[str, Any]] = {}
    candidate_directories: dict[str, dict[str, Any]] = {}
    root_level_files = {"logical_size_bytes": 0, "file_count": 0}

    # Each work item carries the top-level directory and candidate directories
    # whose totals include the files below it.
    pending: list[tuple[str, str | None, tuple[str, ...]]] = [(root_path, None, ())]
    while pending:
        directory, top_path, candidate_ancestors = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: (item.name.casefold(), item.name))
        except OSError as error:
            errors.append(_new_error(directory, "scandir", error))
            if top_path is not None:
                top_directories[top_path]["error_count"] += 1
            continue

        for entry in entries:
            path = _absolute_normalized(entry.path)
            parts = _relative_parts(path, root_path)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as error:
                errors.append(_new_error(path, "lstat", error))
                if top_path is not None:
                    top_directories[top_path]["error_count"] += 1
                continue

            reparse_kind = _reparse_kind(metadata)
            if reparse_kind:
                reparse_points.append(
                    {
                        "path": path,
                        "kind": reparse_kind,
                        "classification": "protected_material",
                        "reason": "链接或 reparse point 不会被跟随或计入目标内容。",
                        "recovery_suggestion": "保留链接本身；如需盘点目标，请将目标作为单独根路径审阅。",
                    }
                )
                continue

            if stat.S_ISDIR(metadata.st_mode):
                child_top_path = top_path
                if top_path is None:
                    child_top_path = path
                    top_directories[path] = {
                        "path": path,
                        "name": entry.name,
                        "logical_size_bytes": 0,
                        "file_count": 0,
                        "error_count": 0,
                        "complete": True,
                    }

                child_candidates = candidate_ancestors
                directory_class = _classify_directory(parts)
                if directory_class:
                    record = {
                        "path": path,
                        "classification": directory_class["classification"],
                        "candidate": True,
                        "reason": _CANDIDATE_REASON[directory_class["kind"]],
                        "recovery_suggestion": _CANDIDATE_RECOVERY[directory_class["kind"]],
                        "logical_size_bytes": 0,
                        "file_count": 0,
                        "protected_file_count": 0,
                    }
                    candidate_directories[path] = record
                    child_candidates = (*candidate_ancestors, path)
                pending.append((path, child_top_path, child_candidates))
                continue

            if stat.S_ISREG(metadata.st_mode):
                classification = _classify_file(parts)
                file_record = {
                    "path": path,
                    "size_bytes": metadata.st_size,
                    **classification,
                }
                files.append(file_record)
                size = metadata.st_size
                if top_path is None:
                    root_level_files["logical_size_bytes"] += size
                    root_level_files["file_count"] += 1
                else:
                    top_directories[top_path]["logical_size_bytes"] += size
                    top_directories[top_path]["file_count"] += 1
                for candidate_path in candidate_ancestors:
                    candidate = candidate_directories[candidate_path]
                    candidate["logical_size_bytes"] += size
                    candidate["file_count"] += 1
                    if not classification["candidate"]:
                        candidate["protected_file_count"] += 1
                continue

            special_entries.append(
                {
                    "path": path,
                    "kind": "special_filesystem_entry",
                    "classification": "protected_material",
                    "reason": "非普通文件系统对象；扫描器不读取或操作其内容。",
                    "recovery_suggestion": "保留原对象并由负责人确认用途。",
                }
            )

    for record in top_directories.values():
        record["complete"] = record["error_count"] == 0

    return {
        "schema_version": 1,
        "scanned_at": scanned_at,
        "root": root_path,
        "disk_free_bytes_before_scan": free_bytes,
        "size_basis": "普通文件 st_size 逻辑字节数按路径累加；不计目录、reparse point 或特殊文件，硬链接可能按路径重复计数。",
        "top_level_directories": sorted(
            top_directories.values(), key=lambda item: (item["name"].casefold(), item["name"])
        ),
        "root_level_files": root_level_files,
        "candidate_directories": sorted(candidate_directories.values(), key=lambda item: item["path"]),
        "files": sorted(files, key=lambda item: item["path"]),
        "reparse_points": sorted(reparse_points, key=lambda item: item["path"]),
        "special_entries": sorted(special_entries, key=lambda item: item["path"]),
        "errors": sorted(errors, key=lambda item: (item["path"], item["operation"])),
    }


def _write_new_inventory(path: str, inventory: dict[str, Any]) -> None:
    output_path = _absolute_normalized(path)
    parent = os.path.dirname(output_path) or os.curdir
    _validate_real_directory(_absolute_normalized(parent))
    try:
        os.lstat(output_path)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(errno.EEXIST, "refusing to overwrite an existing inventory", output_path)

    inventory["inventory_path"] = output_path
    payload = json.dumps(inventory, ensure_ascii=False, indent=2) + "\n"
    with open(output_path, "x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="directory to inventory")
    parser.add_argument("--output", help="new JSON file path; omitted means write JSON to stdout")
    arguments = parser.parse_args(argv)

    try:
        inventory = build_inventory(arguments.root)
        if arguments.output:
            _write_new_inventory(arguments.output, inventory)
        else:
            json.dump(inventory, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
    except (OSError, ValueError) as error:
        print(f"inventory failed: {error}", file=sys.stderr)
        return 2
    return 1 if inventory["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
