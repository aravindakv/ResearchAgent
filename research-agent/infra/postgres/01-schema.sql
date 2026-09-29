-- Runs once, when the Postgres data volume is first created.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE jobs (
    id          UUID PRIMARY KEY,
    user_id     TEXT NOT NULL,
    topic       TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'queued',
    error       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX jobs_user_idx ON jobs (user_id, created_at DESC);

CREATE TABLE chunks (
    id            BIGSERIAL PRIMARY KEY,
    job_id        UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    url           TEXT NOT NULL,
    content       TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    embedding     vector(1536) NOT NULL,
    tsv           tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (job_id, content_hash)
);
CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);

CREATE TABLE eval_results (
    job_id  UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    metric  TEXT NOT NULL,
    score   REAL NOT NULL,
    PRIMARY KEY (job_id, metric)
);

-- LangGraph checkpoints move here in chapter 10, owned by a least-privilege worker role.
CREATE SCHEMA lg;