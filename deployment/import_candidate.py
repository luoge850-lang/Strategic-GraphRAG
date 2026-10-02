"""Import a verified package into an EMPTY loopback Neo4j database atomically."""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare(graph: dict) -> dict:
    from strategic_graphrag.staging_index import StagingGraphIndex
    index = StagingGraphIndex(graph)
    entities, documents, claims, sentences, observations = {}, {}, {}, {}, {}
    relations = defaultdict(list)
    seen_edges = set()
    for edge in index.edges:
        edge_id = edge["edge_id"]
        if edge_id in seen_edges:
            raise ValueError("duplicate fact edge id")
        seen_edges.add(edge_id)
        relation = edge["relation"]
        if not re.fullmatch(r"[A-Z][A-Z_]*", relation):
            raise ValueError("invalid relation identifier")
        for position in ("source", "target"):
            category = edge[position + "_category"]
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", category):
                raise ValueError("invalid entity label")
            node = entities.setdefault(edge[position], {"id": edge[position], "name": edge[position].replace("_", " "),
                                                       "build_id": index.build_id, "labels": set()})
            node["labels"].add(category)
        doc_id = edge["source_filing"].removesuffix(".pdf")
        documents[doc_id] = {"doc_id": doc_id, "filename": edge["source_filing"],
            "document_sha256": edge["document_sha256"], "build_id": index.build_id,
            "filing_fiscal_year": edge["filing_year"]}
        sentences[edge["sentence_id"]] = {"id": edge["sentence_id"], "text": edge["evidence"],
            "page": edge["page"], "doc_id": doc_id, "build_id": index.build_id}
        claim = claims.setdefault(edge["claim_id"], {
            "id": edge["claim_id"], "text": edge["evidence"], "page": edge["page"],
            "doc_id": doc_id, "source_filing": edge["source_filing"],
            "document_sha256": edge["document_sha256"], "build_id": index.build_id,
            "source_id": edge["source"], "target_id": edge["target"],
            "relation_type": relation, "verification_status": "VERBATIM",
            "filing_fiscal_year": edge["filing_year"], "sentence_id": edge["sentence_id"],
            "metric_unit": edge.get("metric_unit"), "_metric_values": []})
        if relation == "REPORTS_METRIC":
            claim["_metric_values"].append({"period": f"FY{edge['fact_year']}", "value": edge["metric_value"]})
        typed = StagingGraphIndex._to_path([edge], edge_id).financial_observations
        if relation == "REPORTS_METRIC" and len(typed) != 1:
            raise ValueError("metric edge must produce exactly one canonical observation")
        for observation in typed:
            row = observation.to_dict()
            if row["id"] in observations and observations[row["id"]] != row:
                raise ValueError("conflicting observation identity")
            observations[row["id"]] = row
        relations[relation].append({"source": edge["source"], "target": edge["target"],
            "props": {"edge_id": edge_id, "evidence_id": edge["claim_id"], "build_id": index.build_id,
                "source_filing": edge["source_filing"], "filing": edge["source_filing"],
                "page": edge["page"], "year": edge["fact_year"], "filing_fiscal_year": edge["filing_year"],
                "evidence_sentence": edge["evidence"], "causal_strength": edge["causal_strength"],
                "causal_form": edge["causal_form"], "metric_value": edge.get("metric_value"),
                "metric_unit": edge.get("metric_unit"), "observation_id": typed[0].id if typed else None}})
    for claim in claims.values():
        claim["metric_values_json"] = json.dumps(claim.pop("_metric_values"))
    return {"build_id": index.build_id, "entities": list(entities.values()), "documents": list(documents.values()),
            "claims": list(claims.values()), "sentences": list(sentences.values()),
            "observations": list(observations.values()), "relations": dict(relations)}


def write_empty(tx, plan: dict):
    count = tx.run("MATCH (n) RETURN count(n) AS count").single()["count"]
    if count:
        raise ValueError("refusing to import into a nonempty database; no deletion is performed")
    tx.run("CREATE (:CandidateImportLock {id:'only', build_id:$build})", build=plan["build_id"]).consume()
    by_label = defaultdict(list)
    for item in plan["entities"]:
        for label in item["labels"]:
            by_label[label].append({key: value for key, value in item.items() if key != "labels"})
    for label, items in by_label.items():
        tx.run(f"UNWIND $items AS item MERGE (n:Entity {{id:item.id}}) SET n:{label}, n += item", items=items).consume()
    for label, key, items in (("Document", "doc_id", plan["documents"]), ("Sentence", "id", plan["sentences"]),
                             ("EvidenceClaim", "id", plan["claims"]), ("FinancialObservation", "id", plan["observations"])):
        tx.run(f"UNWIND $items AS item CREATE (n:{label}) SET n += item", items=items).consume()
    tx.run("""MATCH (c:EvidenceClaim), (s:Sentence {id:c.sentence_id}), (d:Document {doc_id:c.doc_id})
        MERGE (c)-[:SUPPORTED_BY]->(s) MERGE (s)-[:BELONGS_TO]->(d)
        WITH c MATCH (a:Entity {id:c.source_id}), (b:Entity {id:c.target_id})
        MERGE (c)-[:ABOUT_SOURCE]->(a) MERGE (c)-[:ABOUT_TARGET]->(b)""").consume()
    for relation, items in plan["relations"].items():
        tx.run(f"""UNWIND $items AS item MATCH (a:Entity {{id:item.source}}), (b:Entity {{id:item.target}})
            CREATE (a)-[r:{relation}]->(b) SET r += item.props""", items=items).consume()
    tx.run("""MATCH (o:FinancialObservation), (c:EvidenceClaim {id:o.claim_id}),
        (a:Entity {id:o.company_id}), (m:Entity {id:o.metric_id}),
        (d:Document {doc_id:replace(o.source_filing,'.pdf','')})
        MERGE (a)-[:HAS_FINANCIAL_OBSERVATION]->(o) MERGE (o)-[:OBSERVES_METRIC]->(m)
        MERGE (o)-[:SUPPORTED_BY_CLAIM]->(c) MERGE (o)-[:DISCLOSED_IN]->(d)
        MERGE (y:Year {year:o.fiscal_year}) SET y.build_id=o.build_id
        MERGE (o)-[:VALID_DURING]->(y)""").consume()
    return {"claims": len(plan["claims"]), "observations": len(plan["observations"]),
            "fact_edges": sum(map(len, plan["relations"].values()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, required=True, help="ignored private local trial credentials")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new path")
    credentials = json.loads(args.credentials.read_text())
    if urlsplit(credentials["uri"]).hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("only an explicitly isolated loopback database is supported")
    from scripts.run_isolated_staging import verify_package
    from neo4j import GraphDatabase
    before = verify_package(args.candidate)
    if before["status"] != "PASS":
        raise ValueError("immutable source package rejected before import")
    graph = json.loads((args.candidate / "graph.json").read_text(encoding="utf-8"))
    plan = prepare(graph)
    started = time.perf_counter()
    with GraphDatabase.driver(credentials["uri"], auth=(credentials["username"], credentials["password"]),
                              max_transaction_retry_time=0) as driver:
        with driver.session(database=credentials["database"]) as session:
            if session.run("MATCH (n) RETURN count(n) AS count").single()["count"]:
                raise ValueError("database is not empty")
            session.run("CREATE CONSTRAINT candidate_import_lock IF NOT EXISTS FOR (n:CandidateImportLock) REQUIRE n.id IS UNIQUE").consume()
            for label, key in (("Entity", "id"), ("Document", "doc_id"), ("Sentence", "id"),
                               ("EvidenceClaim", "id"), ("FinancialObservation", "id"), ("Year", "year")):
                session.run(f"CREATE CONSTRAINT candidate_{label.lower()} IF NOT EXISTS FOR (n:{label}) REQUIRE n.{key} IS UNIQUE").consume()
            session.run("CALL db.awaitIndexes(30)").consume()
            counts = session.execute_write(write_empty, plan)
            session.run("CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS FOR (n:Entity) ON EACH [n.name]").consume()
            session.run("CALL db.awaitIndexes(30)").consume()
    after = verify_package(args.candidate)
    report = {"schema": "real-candidate-import/v1", "status": "PASS" if after["status"] == "PASS" else "FAIL",
              "build_id": plan["build_id"], "counts": counts, "elapsed_ms": round((time.perf_counter()-started)*1000,3),
              "source_package_before": before["status"], "source_package_after": after["status"],
              "remote_store_mutations": 0, "scope": "real loopback Neo4j, atomic empty-store import",
              "semantic_quality": "import identity is not independent PDF-based fact accuracy"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
