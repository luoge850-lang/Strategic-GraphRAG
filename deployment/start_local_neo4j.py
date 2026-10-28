"""Start an isolated, authenticated loopback-only Neo4j trial; never touch Aura."""
from __future__ import annotations
import argparse
import json
import os
import secrets
import shutil
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", type=Path, required=True)
    parser.add_argument("--java-home", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--bolt-port", type=int, default=17687)
    parser.add_argument("--http-port", type=int, default=17474)
    args = parser.parse_args()
    distribution, state = args.distribution.resolve(), args.state_dir.resolve()
    if state.exists():
        parser.error("state directory must be new; existing databases are never reused")
    for required in (distribution / "bin/neo4j.bat", args.java_home / "bin/java.exe"):
        if not required.is_file():
            parser.error("distribution or JDK executable missing")
    state.mkdir(parents=True)
    shutil.copytree(distribution / "conf", state / "conf")
    settings = {
        "server.default_listen_address": "127.0.0.1",
        "server.bolt.listen_address": f"127.0.0.1:{args.bolt_port}",
        "server.bolt.advertised_address": f"127.0.0.1:{args.bolt_port}",
        "server.http.listen_address": f"127.0.0.1:{args.http_port}",
        "server.http.advertised_address": f"127.0.0.1:{args.http_port}",
        "server.https.enabled": "false",
        "server.memory.heap.initial_size": "256m",
        "server.memory.heap.max_size": "512m",
        "server.memory.pagecache.size": "256m",
        "server.directories.data": (state / "data").as_posix(),
        "server.directories.logs": (state / "logs").as_posix(),
        "server.directories.run": (state / "run").as_posix(),
        "dbms.security.auth_enabled": "true",
    }
    config = state / "conf/neo4j.conf"
    original = config.read_text(encoding="utf-8")
    retained = [line for line in original.splitlines() if line.split("=", 1)[0].strip() not in settings]
    config.write_text("\n".join(retained + [f"{key}={value}" for key, value in settings.items()]) + "\n", encoding="utf-8")
    env = {**os.environ, "JAVA_HOME": str(args.java_home.resolve()), "NEO4J_CONF": str(state / "conf")}
    password = secrets.token_hex(24)
    admin = subprocess.run([str(distribution / "bin/neo4j-admin.bat"), "dbms", "set-initial-password", password],
        env=env, cwd=distribution, capture_output=True, text=True, timeout=60)
    if admin.returncode:
        raise RuntimeError("local password initialization failed; no remote database was contacted")
    credentials = {"uri": f"bolt://127.0.0.1:{args.bolt_port}", "username": "neo4j", "password": password,
                   "database": "neo4j", "scope": "disposable local trial"}
    with (state / "private_credentials.json").open("x", encoding="utf-8") as stream:
        json.dump(credentials, stream)
    log = (state / "console.log").open("xb")
    process = subprocess.Popen([str(distribution / "bin/neo4j.bat"), "console"], env=env,
        cwd=distribution, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    log.close()
    (state / "launcher_pid.json").write_text(json.dumps({"pid": process.pid}), encoding="utf-8")
    from neo4j import GraphDatabase
    deadline = time.monotonic() + 90
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("local server exited; inspect local console.log")
        try:
            with GraphDatabase.driver(credentials["uri"], auth=("neo4j", password), connection_timeout=2) as driver:
                driver.verify_connectivity()
            print(json.dumps({"status": "READY", "pid": process.pid, "uri": credentials["uri"],
                              "state_dir": str(state), "remote_store_mutations": 0}))
            return
        except Exception as exc:
            last_error = type(exc).__name__
            time.sleep(1)
    raise RuntimeError(f"local startup deadline exceeded: {last_error}")


if __name__ == "__main__":
    main()
