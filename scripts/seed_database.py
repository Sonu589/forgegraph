#!/usr/bin/env python3
"""Create ForgeGraph tables and ingest sample documents into AlloyDB Omni."""
import json
import os
from pathlib import Path
import psycopg2
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

ROOT = Path(__file__).resolve().parents[1]
DSN = os.getenv("DATABASE_URL")
MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "embeddinggemma")
GRAPH_PATH = ROOT / "data" / "maintenance_graph.json"
if not DSN:
    raise SystemExit("Set DATABASE_URL to the AlloyDB Omni demo database DSN first.")

conn = psycopg2.connect(DSN, connect_timeout=5)
try:
    with conn.cursor() as cur:
        cur.execute((ROOT / "sql" / "schema.sql").read_text())
        docs = json.loads((ROOT / "data" / "maintenance_documents.json").read_text())
        graph = json.loads(GRAPH_PATH.read_text())
        for node in graph["nodes"]:
            cur.execute(
                """INSERT INTO knowledge_nodes(node_key,node_type,label,description,document_key)
                   VALUES (%s,%s,%s,%s,%s)
                   ON CONFLICT(node_key) DO UPDATE SET node_type=EXCLUDED.node_type,
                     label=EXCLUDED.label, description=EXCLUDED.description, document_key=EXCLUDED.document_key""",
                (node["node_key"], node["node_type"], node["label"], node["description"], node.get("document_key")),
            )
        for edge in graph["edges"]:
            cur.execute(
                """INSERT INTO knowledge_edges(source_key,target_key,relation,confidence,source_note)
                   VALUES (%s,%s,%s,%s,%s)
                   ON CONFLICT(source_key,target_key,relation) DO UPDATE SET
                     confidence=EXCLUDED.confidence, source_note=EXCLUDED.source_note""",
                (edge["source_key"], edge["target_key"], edge["relation"], edge["confidence"], edge["source_note"]),
            )
        for doc in docs:
            cur.execute(
                """INSERT INTO maintenance_documents(document_key,title,source,chunk_text,embedding)
                   VALUES (%s,%s,%s,%s,google_ml.embedding(%s,%s)::vector)
                   ON CONFLICT(document_key,title,chunk_text) DO NOTHING""",
                (doc["id"], doc["title"], doc["source"], doc["text"], MODEL_ID, doc["text"]),
            )
    conn.commit()
    print(f"Schema ready; attempted to ingest {len(docs)} sample documents, {len(graph['nodes'])} graph nodes and {len(graph['edges'])} graph edges.")
except Exception:
    conn.rollback()
    raise
finally:
    conn.close()
