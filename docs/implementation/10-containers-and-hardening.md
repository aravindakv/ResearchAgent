# 10 — Containers and Hardening

**Goal:** run the whole system in Docker Compose with real security boundaries: each service in its own container with the least privilege it needs, networks that encode who may talk to whom, least-privilege database roles, secrets mounted as files, and Caddy terminating HTTPS in front. Then prove the boundaries with a test script.

The target topology is `docs/images/local-topology.svg`, and the cloud equivalent is `docs/images/aws-trust-zones.svg`.

---

## Concepts first

### Trust zones and least power

Ask, for each component, what it reads and what it can reach:

| Component | Reads untrusted content | Can reach data | Has LLM keys | Has internet |
|---|---|---|---|---|
| Caddy | user requests | no | no | inbound only |
| API | user topics | `jobs` (read/insert), `eval_results` (read) | no | **no** |
| Worker | web text (via chunks) | jobs, chunks, evals, checkpoints | yes | yes |
| MCP server | **raw web pages** | **nothing** | search key only | yes |

The design principle: **the component that reads the most dangerous content gets the least power.** The MCP server reads arbitrary web pages, so it can't reach Postgres, Redis or the API at all. Even a full compromise of it yields a Tavily key and nothing else.

### Docker networks as security groups

Five networks, each a boundary:

| Network | Members | Internet? |
|---|---|---|
| `public` | Caddy | yes (ports published here) |
| `edge` | Caddy, API | no (`internal: true`) |
| `data` | API, worker, Postgres, Redis | no |
| `tools` | worker, MCP server | no |
| `egress` | worker, MCP server | yes |

A container on only internal networks has **no route out**. The API is on `edge` + `data`, so it can't reach the internet or the MCP server. Networks replace "please don't call that service" with "you can't".

### Container hardening

Every image you build runs as a non-root user (UID 10001), and Compose adds:

- `read_only: true` with a `tmpfs` for `/tmp`: an attacker can't drop files into the image.
- `cap_drop: [ALL]`: no Linux capabilities (no raw sockets, no changing file ownership, ...).
- `no-new-privileges`: setuid binaries can't escalate.

### Database roles

Until now, everything used the Postgres superuser. From this chapter:

- `app_api` can `SELECT, INSERT` on `jobs` and `SELECT` on `eval_results`. Nothing else.
- `app_worker` can read and update jobs, manage chunks and eval results, and owns the `lg` schema where LangGraph writes its checkpoint tables.
- Neither can create roles or drop tables, and `PUBLIC` can't even connect.

A SQL injection in the API couldn't read chunks or drop a table.

---

## Step 1: One requirements file and Dockerfile per service

Each image contains only its own dependencies.

File: `api/requirements.txt`

```text
fastapi>=0.115
uvicorn[standard]>=0.30
psycopg[binary]>=3.2
psycopg-pool>=3.2
redis>=5.0
pydantic>=2.7
python-dotenv>=1.0
```

File: `api/Dockerfile`

```dockerfile
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && useradd --uid 10001 --no-create-home app \
 && mkdir /reports && chown 10001 /reports
COPY . .
USER 10001
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

File: `api/.dockerignore`

```text
__pycache__/
*.pyc
.env
```

File: `worker/requirements.txt`

```text
langgraph>=0.6
langgraph-checkpoint-postgres>=2.0
langchain-core>=0.3
langchain-openai>=0.3
langchain-mcp-adapters>=0.1
langchain-text-splitters>=0.3
langchain-typesafe
langsmith>=0.3
openai>=1.40
psycopg[binary]>=3.2
psycopg-pool>=3.2
redis>=5.0
pydantic>=2.7
python-dotenv>=1.0
```

File: `worker/Dockerfile`

```dockerfile
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && useradd --uid 10001 --no-create-home app \
 && mkdir /reports && chown 10001 /reports
COPY . .
USER 10001
CMD ["python", "main.py"]
```

File: `worker/.dockerignore`

```text
__pycache__/
*.pyc
.env
```

File: `mcp_server/requirements.txt`

```text
mcp>=1.10
uvicorn>=0.30
starlette>=0.37
httpx>=0.27
tavily-python>=0.5
trafilatura>=1.12
markdown>=3.6
nh3>=0.2
weasyprint>=62
python-dotenv>=1.0
```

File: `mcp_server/Dockerfile`

```dockerfile
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends libpango-1.0-0 libpangoft2-1.0-0 fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && useradd --uid 10001 --no-create-home app
COPY . .
USER 10001
EXPOSE 8000
CMD ["python", "server.py"]
```

File: `mcp_server/.dockerignore`

```text
__pycache__/
*.pyc
.env
```

**Why:**

- **Dependencies before code.** `COPY requirements.txt` + `pip install` come first, so Docker caches that layer and rebuilds after a code change take seconds.
- **The worker image has no Tavily or WeasyPrint;** the API image has no LangChain. Less code in an image means fewer vulnerabilities to patch.
- **`/reports` is created and owned by UID 10001 in both the API and worker images.** Docker initializes a new named volume from the first container that mounts it, so the ownership carries over.
- **Build contexts are the service folders**, so the repo-root `.env` and `secrets/` can never be copied into an image. The `.dockerignore` files are a second guard.

---

## Step 2: Database roles

File: `infra/postgres/02-roles.sh`

```bash
#!/bin/bash
# Least-privilege roles. Runs once, after 01-schema.sql, when the data volume is first created.
set -euo pipefail
API_PW=$(cat /run/secrets/pg_api_password)
WORKER_PW=$(cat /run/secrets/pg_worker_password)

psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" <<SQL
CREATE ROLE app_api LOGIN PASSWORD '${API_PW}';
CREATE ROLE app_worker LOGIN PASSWORD '${WORKER_PW}';

REVOKE ALL ON DATABASE ${POSTGRES_DB} FROM PUBLIC;
GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO app_api, app_worker;

GRANT SELECT, INSERT ON jobs TO app_api;
GRANT SELECT ON eval_results TO app_api;

GRANT SELECT, UPDATE ON jobs TO app_worker;
GRANT SELECT, INSERT, DELETE ON chunks TO app_worker;
GRANT SELECT, INSERT, UPDATE ON eval_results TO app_worker;
GRANT USAGE ON SEQUENCE chunks_id_seq TO app_worker;

ALTER SCHEMA lg OWNER TO app_worker;
ALTER ROLE app_worker SET search_path = lg, public;
SQL
```

**Why the `lg` schema:** `saver.setup()` needs to create tables. Rather than granting `CREATE` on `public`, the worker owns one dedicated schema, and its `search_path` makes LangGraph create the checkpoint tables there.

---

## Step 3: Caddy

File: `infra/caddy/Caddyfile`

```text
{
	admin off
}

localhost {
	tls internal
	encode gzip

	header {
		Strict-Transport-Security "max-age=31536000"
		X-Content-Type-Options "nosniff"
		X-Frame-Options "DENY"
		Referrer-Policy "no-referrer"
		-Server
	}

	request_body {
		max_size 64KB
	}

	reverse_proxy api:8000
}
```

`tls internal` makes Caddy issue a certificate for `localhost` from its own local CA. `admin off` disables Caddy's admin API. `max_size 64KB` stops oversized bodies before they reach Python. Indentation in a Caddyfile is cosmetic; tabs or spaces both work.

---

## Step 4: The full Compose file

File: `docker-compose.yml`

```yaml
name: research-agent

# Container-side values that override the host-development values in .env.
x-container-env: &container-env
  SECRETS_DIR: /run/secrets
  POSTGRES_HOST: postgres
  POSTGRES_PORT: "5432"
  REDIS_HOST: redis
  MCP_BIND: 0.0.0.0
  MCP_PORT: "8000"
  MCP_URL: http://mcp:8000/mcp
  REPORTS_DIR: /reports
  HOME: /tmp
  XDG_CACHE_HOME: /tmp

# Hardening for the images we build: read-only root FS, no privilege escalation, no capabilities.
x-hardening: &hardening
  read_only: true
  tmpfs: [/tmp]
  security_opt: ["no-new-privileges:true"]
  cap_drop: [ALL]
  restart: unless-stopped
  env_file: .env

services:
  caddy:
    image: caddy:2
    ports:
      - "127.0.0.1:443:443"   # localhost only; see chapter 12 before exposing to your LAN
      - "127.0.0.1:80:80"
    volumes:
      - ./infra/caddy/Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy_data:/data
      - caddy_config:/config
    networks: [public, edge]
    depends_on:
      api: {condition: service_healthy}
    security_opt: ["no-new-privileges:true"]
    restart: unless-stopped

  api:
    <<: *hardening
    build: ./api
    environment:
      <<: *container-env
      DB_USER: app_api
      DB_PASSWORD_SECRET: pg_api_password
      ENABLE_DOCS: "0"
    secrets: [pg_api_password, redis_password, api_tokens]
    volumes:
      - reports:/reports:ro
    networks: [edge, data]
    depends_on:
      postgres: {condition: service_healthy}
      redis: {condition: service_healthy}
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=3)"]
      interval: 10s
      retries: 5

  worker:
    <<: *hardening
    build: ./worker
    environment:
      <<: *container-env
      DB_USER: app_worker
      DB_PASSWORD_SECRET: pg_worker_password
    secrets: [pg_worker_password, redis_password, mcp_token, openai_api_key, langsmith_api_key, typesafe_api_key]
    volumes:
      - reports:/reports
    networks: [data, tools, egress]
    depends_on:
      postgres: {condition: service_healthy}
      redis: {condition: service_healthy}
      mcp: {condition: service_healthy}
    deploy:
      replicas: 2

  mcp:
    <<: *hardening
    build: ./mcp_server
    environment: *container-env
    secrets: [mcp_token, tavily_api_key]
    networks: [tools, egress]          # no route to postgres, redis, or the API
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=3)"]
      interval: 10s
      retries: 5

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: agent
      POSTGRES_DB: agent
      POSTGRES_PASSWORD_FILE: /run/secrets/pg_password
    secrets: [pg_password, pg_api_password, pg_worker_password]
    volumes:
      - pg_data:/var/lib/postgresql/data
      - ./infra/postgres/01-schema.sql:/docker-entrypoint-initdb.d/01-schema.sql:ro
      - ./infra/postgres/02-roles.sh:/docker-entrypoint-initdb.d/02-roles.sh:ro
    networks: [data]
    security_opt: ["no-new-privileges:true"]
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U agent -d agent"]
      interval: 5s
      retries: 10

  redis:
    image: redis:7-alpine
    command: ["sh", "-c", "exec redis-server --requirepass \"$$(cat /run/secrets/redis_password)\" --appendonly yes"]
    secrets: [redis_password]
    volumes:
      - redis_data:/data
    networks: [data]
    security_opt: ["no-new-privileges:true"]
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "redis-cli -a \"$$(cat /run/secrets/redis_password)\" --no-auth-warning ping | grep -q PONG"]
      interval: 5s
      retries: 10

networks:
  public: {}                 # only Caddy; host ports are published here
  edge: {internal: true}     # Caddy <-> API
  data: {internal: true}     # API/worker <-> Postgres, Redis
  tools: {internal: true}    # worker <-> MCP server
  egress: {}                 # internet: worker (OpenAI, TypeSafe, LangSmith) and MCP (web)

volumes:
  pg_data:
  redis_data:
  caddy_data:
  caddy_config:
  reports:

secrets:
  pg_password:        {file: ./secrets/pg_password}
  pg_api_password:    {file: ./secrets/pg_api_password}
  pg_worker_password: {file: ./secrets/pg_worker_password}
  redis_password:     {file: ./secrets/redis_password}
  mcp_token:          {file: ./secrets/mcp_token}
  api_tokens:         {file: ./secrets/api_tokens}
  openai_api_key:     {file: ./secrets/openai_api_key}
  tavily_api_key:     {file: ./secrets/tavily_api_key}
  langsmith_api_key:  {file: ./secrets/langsmith_api_key}
  typesafe_api_key:   {file: ./secrets/typesafe_api_key}
```

**Why:**

- **Two YAML anchors.** `container-env` overrides every host-specific value from `.env` (`environment` beats `env_file`), and each service merges it with its own database role.
- **Each service lists only the secrets it needs.** The API never sees an OpenAI key; the MCP server never sees a database password.
- **Postgres and Redis publish no ports.** Nothing outside Docker can reach them. Step 5 adds an opt-in override for host development.
- **`deploy.replicas: 2`** on the worker gives you competing consumers from the start.

---

## Step 5: Keep host development working

Because Postgres and Redis are now on internal networks with no published ports, the host-development targets (`make run`, `make api`, `make worker`) need an override that publishes them on localhost again.

File: `docker-compose.dev.yml`

```yaml
# Host development override: publish Postgres and Redis on localhost so `make run`, `make api`
# and `make worker` keep working. Internal networks can't publish ports, so both also join devnet.
services:
  postgres:
    ports: ["127.0.0.1:5432:5432"]
    networks: [data, devnet]
  redis:
    ports: ["127.0.0.1:6379:6379"]
    networks: [data, devnet]

networks:
  devnet: {}
```

---

## Step 6: Isolation tests

File: `scripts/check-isolation.sh`

```bash
#!/usr/bin/env bash
# Verifies network boundaries and the SSRF guard. Run after `make up`.
set -uo pipefail
cd "$(dirname "$0")/.."

can_reach() {
  docker compose exec -T "$1" python -c \
    "import socket; socket.create_connection(('$2', $3), timeout=3)" >/dev/null 2>&1
}
expect_blocked() { if can_reach "$1" "$2" "$3"; then echo "FAIL  $1 -> $2:$3 is reachable"; else echo "ok    $1 -> $2:$3 blocked"; fi; }
expect_open()    { if can_reach "$1" "$2" "$3"; then echo "ok    $1 -> $2:$3 reachable"; else echo "FAIL  $1 -> $2:$3 unreachable"; fi; }

echo "== Should be blocked"
expect_blocked api    example.com 443
expect_blocked api    mcp         8000
expect_blocked mcp    postgres    5432
expect_blocked mcp    redis       6379
expect_blocked mcp    api         8000
echo "== Should be reachable"
expect_open    worker postgres    5432
expect_open    worker mcp         8000
expect_open    mcp    example.com 443

echo "== SSRF guard"
for url in http://169.254.169.254/ http://127.0.0.1:8000/ http://postgres:5432/ file:///etc/passwd; do
  if docker compose exec -T mcp python -c "from server import assert_public_url; assert_public_url('$url')" >/dev/null 2>&1; then
    echo "FAIL  $url allowed"; else echo "ok    $url rejected"; fi
done
```

---

## Step 7: The final Makefile

File: `Makefile`

```makefile
.RECIPEPREFIX = >
PY := .venv/bin/python
DEV := docker compose -f docker-compose.yml -f docker-compose.dev.yml
ENGINE ?= jev
.PHONY: venv secrets infra-up infra-down psql redis-cli mcp run resume show graph api worker smoke-dev \
        up up-openjev down ps logs smoke isolation trust-ca backup reset eval-guardrail eval-benchmark

# ---- setup (chapter 01) ----
venv:
> uv venv .venv --python 3.12
> uv pip install --python $(PY) -r requirements-dev.txt

secrets:
> ./scripts/init-secrets.sh

# ---- host development (chapters 02-09) ----
infra-up:
> $(DEV) up -d postgres redis

infra-down:
> docker compose stop postgres redis

psql:
> docker compose exec postgres psql -U agent -d agent

redis-cli:
> docker compose exec redis sh -c 'redis-cli -a "$$(cat /run/secrets/redis_password)" --no-auth-warning'

mcp:
> $(PY) mcp_server/server.py

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

# ---- containers (chapter 10) ----
up:
> docker compose up -d --build

down:
> docker compose down

ps:
> docker compose ps

logs:
> docker compose logs -f --tail=100 worker mcp api

up-openjev:
> docker compose -f docker-compose.yml -f docker-compose.openjev.yml up -d --build

smoke:
> ./scripts/smoke-test.sh "$(TOPIC)"

isolation:
> ./scripts/check-isolation.sh

trust-ca:
> docker compose cp caddy:/data/caddy/pki/authorities/local/root.crt infra/caddy-root.crt
> @echo "Saved infra/caddy-root.crt"

backup:
> mkdir -p backups
> docker compose exec -T postgres pg_dump -U agent agent | gzip > backups/agent-$$(date +%F-%H%M).sql.gz
> @ls -lh backups | tail -n 3

reset:
> docker compose down -v

# ---- evaluation (chapter 11) ----
eval-guardrail:
> $(PY) evals/guardrail_eval.py --engine $(ENGINE)

eval-benchmark:
> $(PY) evals/benchmark_eval.py
```

File: `README.md`

```markdown
# Research Agent

A learning project: an AI agent that researches a technical topic and returns a cited PDF.
Built step by step following `research-agent-guide/`.

## Quick reference

    make venv secrets          # setup
    make up                    # full stack in containers (https://localhost)
    make up-openjev            # ... plus self-hosted OpenJev on a GPU (chapter 10, step 9)
    make isolation             # verify network boundaries and the SSRF guard
    make smoke TOPIC="..."     # end-to-end test through Caddy
    make logs                  # follow worker, MCP and API logs

    make infra-up              # host development: Postgres + Redis on localhost
    make mcp / worker / api    # run components on the host
    make run TOPIC="..."       # run the graph directly

    make eval-guardrail ENGINE=jev|llm
    make eval-benchmark
    make backup | reset
```

---

## Step 8: Start fresh and bring it all up

**Do:** stop the host processes from chapter 09 (Ctrl+C in each terminal). The new roles only exist in a **fresh** database, so reset the volumes (this deletes your chapter 02–09 data):

```bash
chmod +x infra/postgres/02-roles.sh scripts/*.sh
make reset
make up                      # first build takes a few minutes (the MCP image installs PDF libraries)
make ps                      # postgres, redis, api, mcp healthy; two workers running; caddy up
make logs                    # workers log "ready (decision engine: jev)"
```

Trust Caddy's local certificate (optional; the scripts fall back to `-k`):

```bash
make trust-ca
```

---

## Step 9 (optional): the decision server inside the stack

How the containerized worker reaches its decision server depends on the option you chose in chapter 08:

| Chapter 08 option | In containers |
|---|---|
| A (TypeSafe) or B (OpenJev on Codiv) | Works as is: `.env` reaches the worker through `env_file`, and the worker is on the `egress` network. |
| C (self-hosted OpenJev on a GPU) | Use the override below: OpenJev runs as a service in the stack. `127.0.0.1` inside a container is the container itself, so the host URL from `.env` can't work there. |
| D (small encoder on the host CPU) | For the containers, switch `.env` to option B. The host-only server isn't reachable from the stack. |

For option C, OpenJev gets its own internal network, `decide`, shared only with the worker:

File: `docker-compose.openjev.yml`

```yaml
# Optional: self-hosted OpenJev as part of the stack (chapter 08, option C).
# Needs an NVIDIA GPU with >= 24 GB and the NVIDIA Container Toolkit.
#   make up-openjev      (or: docker compose -f docker-compose.yml -f docker-compose.openjev.yml up -d)
services:
  openjev:
    image: razorback16/openjev:0.5.0
    ipc: host                        # vLLM needs large shared memory; OpenJev's own setup runs it this way
    volumes:
      - hf_cache:/root/.cache/huggingface
    networks: [decide, egress]       # decide: only the worker can reach it; egress: first-start weight download
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: all, capabilities: [gpu]}]
    security_opt: ["no-new-privileges:true"]
    restart: unless-stopped

  worker:
    environment:
      TYPESAFE_BASE_URL: http://openjev:8080
    networks: [data, tools, egress, decide]

networks:
  decide: {internal: true}

volumes:
  hf_cache:
```

**Why:**

- **A separate `decide` network.** The MCP server is also on `tools`, and there's no reason it should reach the decision server. Only the worker joins `decide`.
- **`egress` for OpenJev** only because the first start downloads about 18 GB of weights into the `hf_cache` volume. Once they're cached, you can remove `egress` from OpenJev's networks.
- **No API key on the server.** Network isolation is the control here: only the worker can reach it. `secrets/typesafe_api_key` just needs any non-empty value to select the engine.
- **Third-party image, lighter hardening.** A GPU inference server needs writable caches and shared memory, so it doesn't get the read-only and capability rules of your own images. Isolate it instead.

**Do:**

```bash
make up-openjev
docker compose -f docker-compose.yml -f docker-compose.openjev.yml logs -f openjev    # wait for the model to load
docker compose exec worker python -c "import urllib.request; print(urllib.request.urlopen('http://openjev:8080/v1/models').read()[:200])"
make logs                                   # worker: "decision engine: jev:openjev-latest@openjev:8080"
```

To make every `docker compose` command include the override, add `COMPOSE_FILE=docker-compose.yml:docker-compose.openjev.yml` to `.env`; then plain `make up` includes OpenJev too.

---

## Verify

```bash
make isolation
make smoke TOPIC="How does Raft consensus work"
```

| Check | Pass condition |
|---|---|
| `make isolation` | every line `ok` |
| `make smoke` | job completes through `https://localhost`, PDF saved |
| No token | `curl -sk -o /dev/null -w "%{http_code}\n" https://localhost/jobs/x` → `401` |
| Oversized body | `python3 -c "print('{\"topic\":\"' + 'a'*70000 + '\"}')" \| curl -sk -o /dev/null -w "%{http_code}\n" -H "Content-Type: application/json" --data-binary @- https://localhost/research` → `413` |
| API has no internet | covered by `make isolation` (`api -> example.com blocked`) |
| Roles | `make psql`, then `\du` shows `app_api` and `app_worker`; `\dt lg.*` shows checkpoint tables owned by `app_worker` |
| Ports | `ss -tlnp \| grep -E ':443\|:5432\|:6379'` shows only `127.0.0.1:443` (no 5432/6379) |
| Non-root | `docker compose exec api id` → `uid=10001` |
| Read-only | `docker compose exec api touch /app/x` → `Read-only file system` |

Now try the least-privilege role from inside the database:

```sql
-- make psql
SET ROLE app_api;
SELECT count(*) FROM jobs;       -- works
SELECT count(*) FROM chunks;     -- ERROR: permission denied
DROP TABLE jobs;                 -- ERROR: must be owner
RESET ROLE;
```

**Troubleshooting:**

| Symptom | Fix |
|---|---|
| `role "app_api" does not exist` | The volume predates this chapter. `make reset && make up`. |
| Worker exits: `permission denied for schema public` | Same cause: `lg` isn't owned by `app_worker`. Reset. |
| `Permission denied` reading `/run/secrets/...` | Rerun `make secrets` (it resets file modes to 644 inside the 700 directory). |
| Caddy fails to bind 443 | Something else uses 443, or rootless Docker: map `127.0.0.1:8443:443` and use `BASE=https://localhost:8443`. |
| MCP rejects worker requests with a host/origin error | Your `mcp` version enforces DNS-rebinding protection; allow host `mcp:8000` in its transport security settings. |
| Worker logs `decision engine: jev:...@127.0.0.1:8080` in containers | `.env` has a host URL; containers can't reach it. Use Codiv (option B) or `make up-openjev` (step 9). |
| `make run` fails after this chapter | Run `make infra-up` first (it publishes the ports via the dev override). Don't run host workers and container workers at the same time: they use different checkpoint schemas. |

**Commit:**

```bash
git add -A && git commit -m "ch10: containers, networks, roles, caddy, isolation tests"
```

**Checkpoint questions:**

1. Which single component reads the most dangerous content, and what exactly could an attacker get by fully compromising it?
2. Why must Postgres join `devnet` in the dev override instead of just publishing a port?
3. The API can `INSERT` into `jobs` but not `UPDATE` it. Who updates job status, and why is that split useful?
