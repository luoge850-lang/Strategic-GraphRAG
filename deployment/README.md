# Protected public demo deployment (prepared, not deployed)

This Compose stack serves the existing FastAPI and built React frontend through
Caddy. It is a deployment recipe, not evidence that the service is live. A Linux
host, a real DNS name, Docker Compose, reachable TCP 80/443, an isolated Neo4j
candidate database, and a complete verified candidate package are still needed.
No command here imports data from the old database or creates an empty candidate
database. The handoff reports that the old Aura data lacks `build_id` on all 234 observations;
pointing this deployment at that database must fail production preflight.

## Inputs and gates

1. Create a *dedicated* Neo4j database containing only the real candidate
   import. The same canonical `GRAPHRAG_BUILD_ID` must bind every EvidenceClaim,
   business relationship, FinancialObservation and its source, and every Chroma
   record. Same-database label separation is insufficient because existing
   API paths contain global queries. The production preflight is
   responsible for rejecting mixed, missing, or empty data.
2. Supply the immutable candidate package built with this source tree, including
   build manifests, ledgers, and `vector_index/` with the real 843 chunks. Mount
   it at `/candidate` through `CANDIDATE_HOST_DIR`; it is never writable by the
   containers. The three PDF paths must contain the exact bytes recorded by the
   candidate build. The image includes `strategic_graphrag/`, `scripts/`,
   `tests/`, `frontend/src/`, all `requirements*.txt`, and the root
   `.env.example` so the source fingerprint under `ROOT=/app` can be checked.
3. `deployment/preflight.py` must exist and support
   `--production --output /runtime/preflight.json`. The API entrypoint copies
   `/candidate/vector_index` into writable `/app/data/chroma_db`, runs that
   preflight, and starts Uvicorn only if it exits successfully. The writable
   Chroma copy and preflight report live in tmpfs and are rebuilt on restart.
   The copy is a runtime derivative, never an import source of truth. A failed
   preflight leaves the API stopped. Verify that the preflight checks
   the *runtime copy* and immutable package, source fingerprint, PDF hashes,
   collection/build metadata and full dedicated Neo4j contents before rollout.
4. Copy `deployment/.env.example` to `deployment/.env` on the host and fill all
   required fields with real values. Use absolute Linux host paths for the
   candidate and PDFs. `DEMO_DOMAIN` must resolve to the server, not a sample
   domain. Give `NEO4J_*` credentials access only to the isolated candidate
   database. Generate a random `API_KEY` and do not put it in frontend code or
   browser storage. For example:

   ```sh
   cp deployment/.env.example deployment/.env
   chmod 600 deployment/.env
   docker run --rm caddy:2.10-alpine caddy hash-password --plaintext 'CHOOSE_A_LONG_UNIQUE_PASSWORD'
   ```

   Put the resulting **hash only** in `BASIC_AUTH_HASH` and surround it with
   single quotes in `.env` so Compose retains the dollar signs. The plaintext
   example above is a command placeholder; avoid putting a real password in
   shared shell history. Fill `BASIC_AUTH_USER`, `DEMO_DOMAIN`, canonical build
   ID, collection, dedicated Neo4j connection, and one working LLM provider.
   Compose derives `CORS_ORIGINS=https://DEMO_DOMAIN` and fixes
   `API_AUTH_ENABLED=true`, `GRAPH_EMBEDDING_BACKEND=chroma_onnx`, and
   `GRAPH_VECTOR_DB_PATH=/app/data/chroma_db`. The current `/health/ready`
   requires an available LLM even when queries use `synthesize=false`.

The Python lock `requirements-lock-2026-09-19.txt` records a Windows x64
Python 3.12.14 environment. Linux wheel resolution and runtime behavior have
not been established for it, so the Dockerfile uses `requirements.txt`.
That means Python dependencies are not fully pinned for this deployment. The
frontend uses `npm ci` against `frontend/package-lock.json`. Build a tested
Linux lock before claiming reproducible image builds.

## Deploy on a suitable Linux host

Run from the repository root after the inputs above are actually available:

```sh
docker compose --env-file deployment/.env -f deployment/compose.yaml config --quiet
docker compose --env-file deployment/.env -f deployment/compose.yaml build
docker compose --env-file deployment/.env -f deployment/compose.yaml up -d
docker compose --env-file deployment/.env -f deployment/compose.yaml ps
docker compose --env-file deployment/.env -f deployment/compose.yaml logs --tail=100 api
docker compose --env-file deployment/.env -f deployment/compose.yaml exec -T api cat /runtime/preflight.json
```

The API has no published port. Caddy alone publishes TCP 80/443 and requests
certificates for `DEMO_DOMAIN`. The Caddy image listens on unprivileged
container ports 8080/8443; both containers run as UID/GID 10001, with a
read-only root filesystem, all capabilities dropped, `no-new-privileges`,
bounded tmpfs writes, and read-only candidate/PDF mounts. Caddy's certificate
and configuration volumes persist across restarts. Check real host firewall,
DNS, and certificate issuance separately.

After Caddy starts, authenticate with the demo username and password:

```sh
curl -fsS -u 'YOUR_DEMO_USER' https://YOUR_DOMAIN/health/ready
curl -fsS -u 'YOUR_DEMO_USER' -H 'Content-Type: application/json' \
  --data '{"question":"What risks are discussed in the 2025 filing?","synthesize":false}' \
  https://YOUR_DOMAIN/query
curl -fsS -u 'YOUR_DEMO_USER' https://YOUR_DOMAIN/
```

The `POST /query` example uses `synthesize=false` and does not incur an LLM
generation request. The frontend also defaults to `synthesize=false`; turning
that option on may use the configured paid model. The model still has to pass current readiness. A 200 on
`/health/ready` is a runtime check, not a substitute for the production
preflight report or a review of the candidate database.

Caddy applies HTTPS Basic Auth to every route, injects the API key upstream,
removes the browser's Authorization header, and caps `POST /query` bodies at
16 KB. It allows only GET for `/`, `/assets/*`, `/graph/*`, `/evidence/*`,
`/health`, `/health/live`, `/health/ready`, and three exact PDF source URLs.
All other methods and routes return 403, including every evaluation annotation
endpoint, `POST /query/vector`, PATCH, PUT, and DELETE. The PDF files are
read-only; the API's own annotation routes still exist internally, so keep
port 8000 unpublished and do not add another ingress to the private network.
The production preflight also requires `QUERY_CACHE_TTL_SECONDS=0`, fixed in
Compose, to prevent cross-build answer reuse.

## Stop and roll back

Before replacing a working release, retain its image tags, candidate package,
PDF files, and a protected copy of its `.env` outside the repository. The image
tags use `GRAPHRAG_BUILD_ID`; do not overwrite an existing tag with a different
source tree. To stop public access while retaining Caddy certificate volumes:

```sh
docker compose --env-file deployment/.env -f deployment/compose.yaml down
```

To roll back on the same source release directory, restore the **previous**
`.env` values (including previous canonical ID, dedicated DB and package/PDF
paths), then start the retained previous image tags without rebuilding:

```sh
docker compose --env-file /secure/location/previous.env -f deployment/compose.yaml config --quiet
docker compose --env-file /secure/location/previous.env -f deployment/compose.yaml up -d --no-build --force-recreate
docker compose --env-file /secure/location/previous.env -f deployment/compose.yaml ps
```

If source or Compose config changed, use the matching previous checkout and
its Compose file. Rollback must point to a complete previous candidate database
and package; preflight will reject mismatched or legacy data. Do not use
`down -v`: that would remove Caddy certificate storage. No rollback command
rewrites Neo4j or the immutable package.

Status for this handoff: Docker build, Caddy adaptation, container startup,
remote preflight, HTTPS issuance, and live endpoint checks are **NOT_EXECUTED**
on this machine. There is currently no server or domain.

The local Windows HTTP/browser run is separate from this unexecuted Linux
deployment recipe. Its raw records and actual screenshots are in
[`experiments/public-demo-delivery-2026-09-30/`](../experiments/public-demo-delivery-2026-09-30/).
Five requests returned HTTP 200, but only one of five scenario assertions passed.
The revenue citation opened physical PDF page 80; numeric calculation remained
`INSUFFICIENT_EVIDENCE`. Do not substitute this legacy-store run for candidate
or production acceptance.

The ONNX embedding model is not bundled in the candidate vector snapshot.
Chroma may download its default model on the first query; network availability,
model artifact hashes, cache persistence and cold-start behavior must be checked
on the target host before release. The current recipe does not meet offline
or fully pinned deployment requirements. No public endpoint is live.
