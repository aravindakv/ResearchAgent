# 02 — Postgres (pgvector) and Redis

**Goal:** Postgres with the pgvector extension and Redis running in Docker, reachable only from your machine, with the complete database schema in place.

---

## Concepts first

### Why pgvector instead of a dedicated vector database?

The agent needs to store four kinds of data: jobs, text chunks with their embeddings, evaluation scores, and LangGraph checkpoints. pgvector adds a `vector` column type and similarity operators to ordinary Postgres, so all four live in one database:

- one thing to back up, secure and monitor;
- joins and transactions across vectors and normal data;
- hybrid search (keyword + vector) in plain SQL.

A dedicated vector database (Qdrant, Weaviate) becomes worth it at much larger scale or when you need features Postgres lacks. For this project, and for many real ones, one Postgres is the better trade-off. The data model diagram is in `docs/images/data-model.svg`.

### What the schema contains

| Table | Purpose |
|---|---|
| `jobs` | One row per research request: who asked, the topic, status, error |
| `chunks` | Source text split into pieces, each with an embedding vector and a full-text search column |
| `eval_results` | Quality scores per job (faithfulness, sufficiency) |
| schema `lg` | Empty for now; LangGraph's checkpoint tables go here in chapter 10 |

Two index types matter:

- **HNSW** on `chunks.embedding`: an approximate nearest-neighbour graph index. It finds the most similar vectors without comparing against every row. "Approximate" means it can occasionally miss the exact best match, in exchange for being dramatically faster at scale.
- **GIN** on `chunks.tsv`: an inverted index for keyword search, ready for hybrid retrieval later.

`tsv` is a **generated column**: Postgres computes it from `content` on every insert and update, so it can never drift out of sync with the text.

### Why Redis?

In chapter 09, Redis becomes the job queue (Redis Streams) and holds the API's rate-limit counters. We start it now so the infrastructure is complete, and so you can check the password setup early.

---

## Step 1: The schema

File: `infra/postgres/01-schema.sql`

```sql
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
```

**Why these details:**

- `vector(1536)` matches the output size of `text-embedding-3-small`. If you switch embedding models, the dimension must match or inserts fail.
- `UNIQUE (job_id, content_hash)` deduplicates: the same paragraph found by two searches is stored once per job.
- `ON DELETE CASCADE` means deleting a job deletes its chunks and scores. That makes data retention trivial.

---

## Step 2: Docker Compose for development

File: `docker-compose.yml`

```yaml
# Development infrastructure (chapters 02-09). Chapter 10 replaces this file with the full stack.
name: research-agent

services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: agent
      POSTGRES_DB: agent
      POSTGRES_PASSWORD_FILE: /run/secrets/pg_password
    secrets: [pg_password]
    ports:
      - "127.0.0.1:5432:5432"      # reachable from this machine only
    volumes:
      - pg_data:/var/lib/postgresql/data
      - ./infra/postgres/01-schema.sql:/docker-entrypoint-initdb.d/01-schema.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U agent -d agent"]
      interval: 5s
      retries: 10

  redis:
    image: redis:7-alpine
    command: ["sh", "-c", "exec redis-server --requirepass \"$$(cat /run/secrets/redis_password)\" --appendonly yes"]
    secrets: [redis_password]
    ports:
      - "127.0.0.1:6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD-SHELL", "redis-cli -a \"$$(cat /run/secrets/redis_password)\" --no-auth-warning ping | grep -q PONG"]
      interval: 5s
      retries: 10

volumes:
  pg_data:
  redis_data:

secrets:
  pg_password:    {file: ./secrets/pg_password}
  redis_password: {file: ./secrets/redis_password}
```

**Why:**

- **`127.0.0.1:` in every port mapping.** Without it, Docker publishes the port on all interfaces, and Docker's iptables rules bypass `ufw`. Your database would be reachable from your whole network even with a firewall "on". This is one of the most common real-world Docker security mistakes.
- **`POSTGRES_PASSWORD_FILE`** makes the official image read the password from the mounted secret file, so the password never appears in `docker inspect`.
- **`$$` in the Redis command:** Compose treats `$` as its own variable syntax, so `$$` passes a literal `$` to the shell inside the container.
- **Healthchecks** let later chapters wait for "ready", not just "started".

---

## Step 3: Makefile targets

File: `Makefile`

```makefile
.RECIPEPREFIX = >
PY := .venv/bin/python
PY_MCP := .venv-mcp/bin/python
.PHONY: venv venv-mcp secrets infra-up infra-down psql redis-cli mcp mcp-check run resume show graph api worker smoke-dev

# ---- setup ----
venv:
> uv venv .venv --python 3.12
> uv pip install --python $(PY) -r requirements-dev.txt

venv-mcp:
> uv venv .venv-mcp --python 3.12
> uv pip install --python $(PY_MCP) -r mcp_server/requirements.txt

secrets:
> ./scripts/init-secrets.sh

# ---- infrastructure (chapter 02) ----
infra-up:
> docker compose up -d postgres redis

infra-down:
> docker compose down

psql:
> docker compose exec postgres psql -U agent -d agent

redis-cli:
> docker compose exec redis sh -c 'redis-cli -a "$$(cat /run/secrets/redis_password)" --no-auth-warning'

# ---- host development (chapters 04-09) ----
mcp:
> $(PY_MCP) mcp_server/server.py

mcp-check:
> ./scripts/mcp-tools-check.sh

run:
> $(PY) worker/run_local.py "$(TOPIC)"

resume:
> $(PY) worker/run_local.py --resume $(ID)

show:
> $(PY) worker/run_local.py --show $(ID)

graph:
> $(PY) worker/run_local.py --graph

api:
> $(PY) -m uvicorn --app-dir api main:app --host 127.0.0.1 --port 8100

worker:
> $(PY) worker/main.py

smoke-dev:
> BASE=http://127.0.0.1:8100 ./scripts/smoke-test.sh "$(TOPIC)"
```

The later targets refer to files you haven't written yet. They're here now so this Makefile doesn't need to change again until chapter 10. `venv-mcp` and `mcp` use a second Python environment for the MCP server; chapter 05 explains why.

---

## Step 4: Start and inspect

**Do:**

```bash
make infra-up
docker compose ps                  # wait until both show "healthy"
make psql
```

Inside `psql`:

```sql
\dx                    -- extensions: vector should be listed
\dt                    -- tables: chunks, eval_results, jobs
\d chunks              -- note the vector(1536) and generated tsv columns
\di                    -- indexes: chunks_embedding_idx (hnsw), chunks_tsv_idx (gin)
SELECT '[1,2,3]'::vector <=> '[1,2,4]'::vector AS cosine_distance;
\q
```

Then Redis:

```bash
make redis-cli
```

```text
PING           -> PONG
SET hello world
GET hello      -> "world"
DEL hello
exit
```

Also prove that Redis refuses unauthenticated clients:

```bash
docker compose exec redis redis-cli ping      # NOAUTH Authentication required
```

---

## Verify

| Check | Pass condition |
|---|---|
| `docker compose ps` | `postgres` and `redis` both `healthy` |
| `\dx` | `vector` listed |
| `\dt` | `chunks`, `eval_results`, `jobs` |
| cosine distance query | a small positive number (about 0.008) |
| Redis with password | `PONG` |
| Redis without password | `NOAUTH` |
| `ss -tlnp \| grep -E '5432\|6379'` | both bound to `127.0.0.1`, not `0.0.0.0` |

**Troubleshooting:**

| Symptom | Fix |
|---|---|
| `port is already allocated` on 5432 | Another Postgres is running on your machine. Change the mapping to `"127.0.0.1:55432:5432"` and set `POSTGRES_PORT=55432` in `.env`. |
| Tables missing | Init scripts only run on an **empty** data volume. `docker compose down -v` (deletes data) then `make infra-up`. |
| Redis healthcheck never passes | The secret file is empty or missing: check `wc -c secrets/redis_password`. |

**Commit:**

```bash
git add -A && git commit -m "ch02: postgres with pgvector, redis, schema"
```

**Checkpoint questions:**

1. Why is `tsv` a generated column rather than something your code computes?
2. What trade-off does "approximate" in HNSW's approximate nearest-neighbour search make?
3. What would `ports: ["5432:5432"]` expose that `"127.0.0.1:5432:5432"` does not, and why doesn't `ufw` help?
