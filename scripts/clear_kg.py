"""Clear Neo4j knowledge graph for re-extraction."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j import GraphDatabase
from dotenv import load_dotenv
load_dotenv()

uri = os.getenv('NEO4J_URI')
user = os.getenv('NEO4J_USERNAME')
pwd = os.getenv('NEO4J_PASSWORD')

print(f"Connecting to {uri}...")
driver = GraphDatabase.driver(uri, auth=(user, pwd), max_connection_lifetime=1800, keep_alive=True)
driver.verify_connectivity()
print("Connected.")

with driver.session() as s:
    result = s.run('MATCH (n) DETACH DELETE n')
    summary = result.consume()
    print(f"Deleted all nodes and relationships.")

driver.close()
print("Neo4j database cleared. Ready for re-extraction.")
