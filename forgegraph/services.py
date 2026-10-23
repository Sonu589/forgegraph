"""Infrastructure adapters for demo mode, AlloyDB Omni, and local/GKE Gemma."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

from .core import chunk_text, extract_model_text, rank_documents
from .graph import local_graph_context, start_nodes

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCS = json.loads((ROOT / "data" / "maintenance_documents.json").read_text())


def mode() -> str:
    configured = os.getenv("APP_MODE", "demo").strip().lower()
    return configured if configured in {"demo", "alloydb"} else "demo"


def _connect():
    try:
        import psycopg2
    except ImportError as exc:
        raise RuntimeError("psycopg2 is missing; install requirements.txt") from exc
    dsn = os.getenv("DATABASE_URL", "").strip()
    if not dsn:
        raise RuntimeError("DATABASE_URL is required when APP_MODE=alloydb")
    return psycopg2.connect(dsn, connect_timeout=5)


def search_documents(query: str, limit: int = 4) -> list[dict[str, Any]]:
    if mode() == "demo":
        return rank_documents(query, DEFAULT_DOCS, limit)
    model_id = os.getenv("EMBEDDING_MODEL_ID", "embeddinggemma")
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """WITH q AS (
                     SELECT google_ml.embedding(%s, %s)::vector AS embedding
                   )
                   SELECT d.document_key, d.title, d.source, d.chunk_text,
                          1 - (d.embedding <=> q.embedding) AS similarity
                   FROM maintenance_documents d CROSS JOIN q
                   ORDER BY d.embedding <=> q.embedding LIMIT %s""",
                (model_id, query, max(1, min(limit, 10))),
            )
            rows = cur.fetchall()
        return [
            {"id": r[0], "title": r[1], "source": r[2], "text": r[3], "similarity": round(float(r[4]), 4)}
            for r in rows
        ]
    finally:
        conn.close()


def graph_context(flags: list[str]) -> dict[str, Any]:
    """Traverse the curated relationship graph; DB mode reads its seeded graph tables."""
    if mode() == "demo":
        return local_graph_context(flags)
    starts = start_nodes(flags)
    if not starts:
        return {"nodes": [], "edges": [], "facts": []}
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """WITH RECURSIVE walk(node_key, depth, path) AS (
                     SELECT n.node_key, 0, ARRAY[n.node_key]
                     FROM knowledge_nodes n WHERE n.node_key = ANY(%s)
                     UNION ALL
                     SELECT CASE WHEN e.source_key = w.node_key THEN e.target_key ELSE e.source_key END,
                            w.depth + 1,
                            w.path || (CASE WHEN e.source_key = w.node_key THEN e.target_key ELSE e.source_key END)
                     FROM walk w
                     JOIN knowledge_edges e ON e.source_key = w.node_key OR e.target_key = w.node_key
                     WHERE w.depth < 2
                       AND (CASE WHEN e.source_key = w.node_key THEN e.target_key ELSE e.source_key END) <> ALL(w.path)
                   ), visited AS (
                     SELECT node_key, MIN(depth) AS depth FROM walk GROUP BY node_key
                   )
                   SELECT n.node_key,n.node_type,n.label,n.description,n.document_key,v.depth
                   FROM visited v JOIN knowledge_nodes n USING (node_key)
                   ORDER BY v.depth, n.node_type, n.label LIMIT 12""",
                (starts,),
            )
            nodes = [
                {"node_key": r[0], "node_type": r[1], "label": r[2], "description": r[3], "document_key": r[4], "depth": r[5]}
                for r in cur.fetchall()
            ]
            keys = [n["node_key"] for n in nodes]
            cur.execute(
                """SELECT source_key,target_key,relation,confidence,source_note
                   FROM knowledge_edges WHERE source_key = ANY(%s) AND target_key = ANY(%s)
                   ORDER BY source_key,target_key LIMIT 30""",
                (keys, keys),
            )
            edges = [
                {"source_key": r[0], "target_key": r[1], "relation": r[2], "confidence": float(r[3]), "source_note": r[4]}
                for r in cur.fetchall()
            ]
        by_key = {n["node_key"]: n["label"] for n in nodes}
        facts = [
            f"{by_key.get(e['source_key'], e['source_key'])} --{e['relation']}--> "
            f"{by_key.get(e['target_key'], e['target_key'])} (confidence={e['confidence']:.2f}; {e['source_note']})"
            for e in edges
        ]
        return {"nodes": nodes, "edges": edges, "facts": facts}
    finally:
        conn.close()


def ingest_document(title: str, source: str, text: str) -> int:
    if not title.strip() or not source.strip() or not text.strip():
        raise ValueError("title, source and text are required")
    pieces = chunk_text(text)
    if mode() == "demo":
        # Demo uploads are intentionally process-local; restart resets the demo corpus.
        DEMO_EXTRA_DOCS.extend({"id": f"upload-{len(DEMO_EXTRA_DOCS)+1}", "title": title.strip(),
                               "source": source.strip(), "text": piece} for piece in pieces)
        return len(pieces)
    model_id = os.getenv("EMBEDDING_MODEL_ID", "embeddinggemma")
    conn = _connect()
    try:
        with conn.cursor() as cur:
            for piece in pieces:
                cur.execute(
                    """INSERT INTO maintenance_documents
                       (document_key, title, source, chunk_text, embedding)
                       VALUES (%s, %s, %s, %s, google_ml.embedding(%s, %s)::vector)
                       ON CONFLICT (document_key, title, chunk_text) DO NOTHING""",
                    (source.strip(), title.strip(), source.strip(), piece, model_id, piece),
                )
        conn.commit()
        return len(pieces)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


DEMO_EXTRA_DOCS: list[dict[str, Any]] = []


def _demo_search(query: str, limit: int = 4) -> list[dict[str, Any]]:
    return rank_documents(query, [*DEFAULT_DOCS, *DEMO_EXTRA_DOCS], limit)


def evidence_for_query(query: str, limit: int = 4) -> list[dict[str, Any]]:
    if mode() == "demo":
        return _demo_search(query, limit)
    return search_documents(query, limit)


def ask_gemma(prompt: str) -> tuple[str | None, str]:
    """Return (answer, backend-label). Never silently treats demo rules as Gemma."""
    endpoint = os.getenv("GEMMA_API_URL", "").strip()
    if not endpoint:
        return None, "not-configured"
    timeout = float(os.getenv("GEMMA_TIMEOUT_SECONDS", "35"))
    try:
        response = requests.post(
            endpoint,
            json={"messages": [
                {"role": "system", "content": "You are ForgeGraph, an evidence-grounded industrial maintenance assistant. Follow safety instructions in the user prompt."},
                {"role": "user", "content": prompt},
            ], "temperature": 0.2, "max_tokens": 500},
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
        answer = extract_model_text(response.json())
        if not answer:
            return None, "empty-model-response"
        return answer, "gemma"
    except requests.RequestException:
        return None, "model-unavailable"
    except (ValueError, TypeError):
        return None, "invalid-model-response"


def save_telemetry(cycle_id: str, pressure: float, temperature: float, vibration: float,
                   flags: list[str]) -> None:
    if mode() == "demo":
        return
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO forging_press_logs(cycle_id, hyd_pressure_bar, oil_temp_c, vib_z_rms, flags) VALUES (%s,%s,%s,%s,%s::jsonb)",
                (cycle_id, pressure, temperature, vibration, json.dumps(flags)),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def db_health() -> tuple[bool, str]:
    if mode() == "demo":
        return True, "demo-mode"
    try:
        conn = _connect()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True, "connected"
        finally:
            conn.close()
    except Exception as exc:
        return False, str(exc)[:180]
