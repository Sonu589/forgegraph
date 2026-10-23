-- ForgeGraph schema for AlloyDB Omni. Run after the codelab has enabled
-- google_ml_integration and registered the EmbeddingGemma model.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS forging_press_logs (
    log_id BIGSERIAL PRIMARY KEY,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    cycle_id TEXT NOT NULL,
    hyd_pressure_bar DOUBLE PRECISION NOT NULL,
    oil_temp_c DOUBLE PRECISION NOT NULL,
    vib_z_rms DOUBLE PRECISION NOT NULL,
    flags JSONB NOT NULL DEFAULT '[]'::jsonb
);

CREATE TABLE IF NOT EXISTS maintenance_documents (
    id BIGSERIAL PRIMARY KEY,
    document_key TEXT NOT NULL,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_key, title, chunk_text)
);

CREATE INDEX IF NOT EXISTS maintenance_documents_embedding_hnsw
    ON maintenance_documents USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS forging_press_logs_recorded_at_idx
    ON forging_press_logs (recorded_at DESC);


-- Small curated incident/component relationship graph (GraphRAG context).
CREATE TABLE IF NOT EXISTS knowledge_nodes (
    node_key TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    label TEXT NOT NULL,
    description TEXT NOT NULL,
    document_key TEXT NULL
);

CREATE TABLE IF NOT EXISTS knowledge_edges (
    source_key TEXT NOT NULL REFERENCES knowledge_nodes(node_key) ON DELETE CASCADE,
    target_key TEXT NOT NULL REFERENCES knowledge_nodes(node_key) ON DELETE CASCADE,
    relation TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.5 CHECK (confidence >= 0 AND confidence <= 1),
    source_note TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (source_key, target_key, relation)
);
CREATE INDEX IF NOT EXISTS knowledge_edges_source_idx ON knowledge_edges (source_key);
CREATE INDEX IF NOT EXISTS knowledge_edges_target_idx ON knowledge_edges (target_key);
