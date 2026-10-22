# Proposed project organization

This is a migration policy, not a completed migration or cleanup operation. Existing historical commands, filenames, manifests and hashes remain authoritative until a separately tested migration is approved.

## Four content classes

| Class | Keep in Git | Suggested owner/location | Rules |
|---|---|---|---|
| Source and configuration schemas | Yes, after credential review | Existing Python package, frontend/src, scripts, deployment, tests, dependency locks | Do not add live configuration or environments. Untracked source is not disposable. |
| Immutable experiment evidence | Small reviewed records | experiments/run-id: protocol, labels, original results, scorer/version, judgments, summaries, failures, manifests | Never overwrite past results or relabel exposed questions as independent validation. |
| Large reproducibility artifacts | Manifest in Git, bytes in a controlled artifact store | builds/build-id: immutable package, source snapshot, frontend bundle, document manifest; separately retained original PDFs | Preserve source identity, licensing and byte-level hashes. Do not deduplicate SQLite/HNSW or immutable packages in place. |
| Rebuildable runtime data | No | One clearly owned runtime directory per build: database state, vector copy, cache, logs, PID/port record | Separate from immutable builds. Removal requires process ownership, dependency review and a demonstrated reconstruction. |

Moving files on the same disk does not free disk space. OneDrive synchronization is not an independently verified backup. A future cold archive needs a manifest, hashes, independent storage and sample restoration.

## Worktree roles

- Keep one clearly designated active development checkout. Preserve its dirty state until a reviewed source commit is prepared.
- Retain the original checkout while user configs, temporary business materials and shared Python/runtime dependencies still live there.
- Detached reproduction and README-only worktrees are archival candidates, not automatically safe deletion targets. Preserve their exact HEAD, clean/dirty status and purpose first; prove recreation before an approved removal.
- Treat nested reproduction clones as a separate category; `git worktree list` does not enumerate them.
- Keep a worktree registry containing path/role/HEAD/status and a separate build registry containing source fingerprint, corpus, package, database, vector collection, frontend and experiment identity. Multiple builds inside one worktree are not multiple worktrees.

## Cleanup batches

1. Rebuildable caches and idle dependencies: demonstrate installation from the matching package lock and a successful build before proposing removal. Reparse targets require separate review.
2. Historical environments and runtime copies: capture interpreter version, installed packages, local installs and modifications. Preserve and restore-test a cold archive first. A Python3.12 lock does not establish that a Python3.14 environment is reproducible.
3. Dependency-bound historical artifacts: keep until validators, dynamic paths, source registries and previous-package hash checks are accounted for. Missing a text-search hit is not proof of independence.
4. Protected materials: source, user configs, original PDFs, frozen labels, raw results, failures, current/rollback packages and required credentials. Credentials remain private and are not read or published as part of a size audit.

Two large historical inventory JSON files may represent different snapshots. Different sizes already rule out byte identity; compressing or archiving them requires preserving original bytes and checking downstream paths. Duplicate PDF bytes may serve different source registries; use a future manifest-based source registry migration instead of removing copies in place.

Historical dated scripts remain valid entry points. Extract common build/evaluation logic into reusable modules in a separate source change with a new build identity, while retaining compatible wrappers. Do not combine that refactor with a supposedly behavior-neutral cleanup.

## Verification before and after a future approved migration

Check Git differences and untracked source, current and rollback package hashes, original-score recomputation, original PDF identities, frontend assets and actual citation pages, service/build isolation, reconstruction commands and free disk space. Report estimated logical bytes, bytes actually removed and observed free-space changes separately. A failed cleanup is not completion.

No blanket `git clean -fdx`, cross-shell recursive deletion, forced live database deletion or removal based only on old timestamps is permitted by this policy.
