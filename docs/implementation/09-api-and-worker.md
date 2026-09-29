# 09 — API, Queue and Worker

**Goal:** turn the command-line runner into a multi-user service. An API accepts jobs and returns immediately; a queue holds them; worker processes run the graph; the API reports status and serves the PDF only to its owner.

```
POST /research ──► API ──► INSERT job (queued) ──► XADD jobs {job_id} ──► 202 {job_id}
                                                          │
                              Worker ◄── XREADGROUP ◄─────┘
                                 │ run graph (resume if checkpointed)
                                 │ UPDATE jobs SET status
                                 └─► XACK
GET /jobs/{id}       ──► status + faithfulness (owner only)
GET /jobs/{id}/pdf   ──► the PDF (owner only)
```

The full sequence diagram is `docs/images/request-sequence.svg`.

---

## Concepts first

### Why a queue instead of a long HTTP request

A run takes minutes. Holding an HTTP request open that long breaks behind proxies and load balancers, ties API capacity to LLM latency, and loses the work if the connection drops. So the API **accepts** work (`202 Accepted` + a job id) and **reports** on it; separate workers **do** it. Workers can scale independently, and the queue absorbs bursts.

### Redis Streams and consumer groups

A Redis Stream is an append-only log. With a **consumer group**:

- `XADD jobs * job_id <id>` appends a message.
- `XREADGROUP GROUP workers <consumer> ... >` gives each new message to **one** consumer in the group.
- The message stays **pending** for that consumer until it calls `XACK`.
- `XAUTOCLAIM` lets another consumer take over messages that have been pending too long, meaning their worker probably crashed.

That's **at-least-once delivery**: a job is never lost, but it may be delivered twice (a worker finishes, then crashes before `XACK`). The system must therefore be **idempotent**: running a job twice must be harmless. Ours is, because of checkpoints (a finished job is detected and not re-run), and because chunk inserts use `ON CONFLICT DO NOTHING`.

### Only the id travels on the queue

The API writes the topic to Postgres and puts only the job id on the queue. The database stays the single source of truth, and the queue never carries user data.

### Authorization, not just authentication

Authentication answers "who are you?" (a valid token). Authorization answers "may you see **this** job?". Every lookup filters by `user_id`, and a job belonging to someone else returns **404, not 403**, so job ids can't be probed. This prevents IDOR bugs (insecure direct object references), one of the most common API vulnerabilities.

---

## Step 1: The API

File: `api/main.py`

```python
"""Public-facing API: authenticates users, accepts jobs, reports status, serves PDFs.

It never calls an LLM and (from chapter 10) has no internet access: it only talks to Postgres and Redis.
"""
import hmac
import os
import time
import uuid
from pathlib import Path

import redis
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field, field_validator

load_dotenv()  # host development only; no .env in containers
SECRETS_DIR = Path(os.environ.get("SECRETS_DIR", "/run/secrets"))


def secret(name: str) -> str:
    return (SECRETS_DIR / name).read_text().strip()


REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", "./reports"))
JOBS_PER_HOUR = int(os.environ.get("JOBS_PER_HOUR", "10"))
DOCS = os.environ.get("ENABLE_DOCS") == "1"

pool = ConnectionPool(
    f"host={os.environ.get('POSTGRES_HOST', '127.0.0.1')} port={os.environ.get('POSTGRES_PORT', '5432')} "
    f"dbname={os.environ.get('POSTGRES_DB', 'agent')} user={os.environ.get('DB_USER', 'agent')} "
    f"password={secret(os.environ.get('DB_PASSWORD_SECRET', 'pg_password'))}",
    min_size=1, max_size=5, open=True,
)
queue = redis.Redis(host=os.environ.get("REDIS_HOST", "127.0.0.1"), password=secret("redis_password"),
                    decode_responses=True)

# secrets/api_tokens holds one "user_id:token" pair per line.
TOKENS: dict[str, str] = {}
for line in secret("api_tokens").splitlines():
    if ":" in line:
        user, token = line.split(":", 1)
        TOKENS[token.strip()] = user.strip()

app = FastAPI(docs_url="/docs" if DOCS else None, redoc_url=None, openapi_url="/openapi.json" if DOCS else None)


def current_user(authorization: str = Header(default="")) -> str:
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() == "bearer" and token:
        for known, user in TOKENS.items():
            if hmac.compare_digest(known.encode(), token.encode()):  # constant-time comparison
                return user
    raise HTTPException(status_code=401, detail="unauthorized")


def enforce_rate_limit(user: str) -> None:
    key = f"rl:{user}:{int(time.time() // 3600)}"
    count = queue.incr(key)
    if count == 1:
        queue.expire(key, 3600)
    if count > JOBS_PER_HOUR:
        raise HTTPException(status_code=429, detail="hourly job limit reached")


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=300)

    @field_validator("topic")
    @classmethod
    def clean(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value.isprintable():
            raise ValueError("topic contains non-printable characters")
        return value


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/research", status_code=202)
def submit(req: ResearchRequest, user: str = Depends(current_user)):
    enforce_rate_limit(user)
    job_id = uuid.uuid4()
    with pool.connection() as conn:
        conn.execute("INSERT INTO jobs (id, user_id, topic) VALUES (%s, %s, %s)", (job_id, user, req.topic))
    queue.xadd("jobs", {"job_id": str(job_id)})  # only the id goes on the queue
    return {"job_id": str(job_id), "status": "queued"}


def load_job(job_id: uuid.UUID, user: str) -> dict:
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT j.id, j.status, j.error, j.created_at, j.updated_at, e.score FROM jobs j "
            "LEFT JOIN eval_results e ON e.job_id = j.id AND e.metric = 'faithfulness' "
            "WHERE j.id = %s AND j.user_id = %s",
            (job_id, user),
        ).fetchone()
    if row is None:  # 404 (not 403) for other users' jobs, so ids can't be probed
        raise HTTPException(status_code=404, detail="job not found")
    return {"job_id": str(row[0]), "status": row[1], "error": row[2],
            "created_at": row[3].isoformat(), "updated_at": row[4].isoformat(), "faithfulness": row[5]}


@app.get("/jobs/{job_id}")
def get_job(job_id: uuid.UUID, user: str = Depends(current_user)):
    return load_job(job_id, user)


@app.get("/jobs/{job_id}/pdf")
def get_pdf(job_id: uuid.UUID, user: str = Depends(current_user)):
    job = load_job(job_id, user)
    path = REPORTS_DIR / f"{job_id}.pdf"  # job_id is a parsed UUID, so no path traversal
    if job["status"] not in ("done", "done_low_confidence") or not path.exists():
        raise HTTPException(status_code=409, detail="report not ready")
    return FileResponse(path, media_type="application/pdf", filename=f"report-{job_id}.pdf")
```

**Why:**

- **`job_id: uuid.UUID` in the route signature:** FastAPI rejects anything that isn't a UUID with a 422 before your code runs, which also rules out path traversal in `get_pdf`.
- **Insert, then enqueue.** A worker can never receive an id whose row doesn't exist yet.
- **Rate limit per user per clock hour** with a Redis counter that expires. It's simple and good enough; a sliding window is a stretch goal.
- **Input normalization:** whitespace collapsed, non-printable characters rejected, length bounded, all before anything is stored.
- **Docs off by default.** `ENABLE_DOCS=1` in your `.env` turns on `/docs` for development.

---

## Step 2: The worker

File: `worker/main.py`

```python
"""Queue consumer: pulls job ids from a Redis stream and runs the LangGraph with a Postgres checkpointer."""
import asyncio
import logging
import socket

import psycopg
import redis.asyncio as aioredis
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from redis.exceptions import ResponseError

from settings import DB_URL, JOB_TIMEOUT_S, REDIS_HOST, REDIS_PASSWORD  # loads secrets first
from graph import build_graph
from decisions import engine_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("worker")
STREAM, GROUP = "jobs", "workers"
CONSUMER = socket.gethostname()


async def set_status(job_id: str, status: str, error: str | None = None) -> None:
    async with await psycopg.AsyncConnection.connect(DB_URL) as conn:
        await conn.execute("UPDATE jobs SET status = %s, error = %s, updated_at = now() WHERE id = %s",
                           (status, error, job_id))


async def run_job(graph, engine: str, job_id: str) -> None:
    config = {"configurable": {"thread_id": job_id}, "recursion_limit": 40, "run_name": "research-job",
              "metadata": {"job_id": job_id, "decision_engine": engine}}
    snapshot = await graph.aget_state(config)
    if snapshot.values.get("final_status") and not snapshot.next:
        # Finished before a crash prevented XACK: just record the outcome (idempotency).
        await set_status(job_id, snapshot.values["final_status"], snapshot.values.get("error"))
        return
    if snapshot.next:
        log.info("resuming job %s at %s", job_id, snapshot.next)
        payload = None  # continue from the last checkpoint instead of starting over
    else:
        async with await psycopg.AsyncConnection.connect(DB_URL) as conn:
            row = await (await conn.execute("SELECT topic FROM jobs WHERE id = %s", (job_id,))).fetchone()
        if row is None:
            return
        payload = {"job_id": job_id, "raw_input": row[0], "iterations": 0}

    await set_status(job_id, "running")
    try:
        final = await asyncio.wait_for(graph.ainvoke(payload, config), timeout=JOB_TIMEOUT_S)
        await set_status(job_id, final.get("final_status", "failed"), final.get("error"))
        log.info("job %s finished: %s", job_id, final.get("final_status"))
    except Exception as exc:  # users see only the error type; details stay in logs and LangSmith
        log.exception("job %s failed", job_id)
        await set_status(job_id, "failed", type(exc).__name__)


async def main() -> None:
    r = aioredis.Redis(host=REDIS_HOST, password=REDIS_PASSWORD, decode_responses=True)
    try:
        await r.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
    except ResponseError:
        pass  # group already exists

    engine = engine_name()
    async with AsyncPostgresSaver.from_conn_string(DB_URL) as saver:
        await saver.setup()
        graph = await build_graph(saver)
        log.info("worker %s ready (decision engine: %s)", CONSUMER, engine)
        while True:
            # Reclaim jobs a crashed worker left unacknowledged for 15 minutes.
            claimed = await r.xautoclaim(STREAM, GROUP, CONSUMER, min_idle_time=900_000, start_id="0-0", count=1)
            messages = claimed[1] if claimed and claimed[1] else []
            if not messages:
                resp = await r.xreadgroup(GROUP, CONSUMER, {STREAM: ">"}, count=1, block=5000)
                messages = resp[0][1] if resp else []
            for msg_id, fields in messages:
                if fields and fields.get("job_id"):
                    await run_job(graph, engine, fields["job_id"])
                await r.xack(STREAM, GROUP, msg_id)


if __name__ == "__main__":
    asyncio.run(main())
```

**Why:**

- **`XACK` only after `run_job` returns.** If the worker dies mid-run, the message stays pending and is reclaimed.
- **Three cases in `run_job`:** already finished (record it), checkpointed mid-run (resume with `None`), or fresh (start with the topic). Together they make redelivery harmless.
- **`asyncio.wait_for`** caps the total run time independently of the graph's own caps.
- **Errors:** the full traceback goes to the log; the user sees only the exception type. Internal details (hostnames, SQL) never leak through the API.

---

## Step 3: A smoke test script

File: `scripts/smoke-test.sh`

```bash
#!/usr/bin/env bash
# Submits a topic, polls until finished, downloads the PDF.
#   BASE=http://127.0.0.1:8100 scripts/smoke-test.sh "topic"    # host development (chapter 09)
#   scripts/smoke-test.sh "topic"                                # containers behind Caddy (chapter 10)
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=${BASE:-https://localhost}
TOKEN=$(head -n1 secrets/api_tokens | cut -d: -f2)
TOPIC=${1:-}
[ -n "$TOPIC" ] || TOPIC="How does the Linux CFS scheduler work"

CURL=(curl -sS --fail-with-body -H "Authorization: Bearer $TOKEN")
if [ -f infra/caddy-root.crt ]; then CURL+=(--cacert infra/caddy-root.crt); else CURL+=(-k); fi

JOB=$("${CURL[@]}" -H "Content-Type: application/json" \
  -d "$(jq -n --arg t "$TOPIC" '{topic: $t}')" "$BASE/research" | jq -r .job_id)
echo "job: $JOB"

while :; do
  STATUS=$("${CURL[@]}" "$BASE/jobs/$JOB" | jq -r .status)
  echo "$(date +%T) status: $STATUS"
  case "$STATUS" in
    done|done_low_confidence) break ;;
    failed|rejected) "${CURL[@]}" "$BASE/jobs/$JOB" | jq .; exit 1 ;;
  esac
  sleep 5
done

"${CURL[@]}" "$BASE/jobs/$JOB" | jq .
"${CURL[@]}" -o "report-$JOB.pdf" "$BASE/jobs/$JOB/pdf"
echo "saved report-$JOB.pdf"
```

`jq -n --arg` builds the JSON body safely: a topic containing quotes can't break the request.

---

## Step 4: Run all four processes

**Do:** four terminals (or tmux panes), all from the repo root:

```bash
make infra-up        # once
make mcp             # terminal 1
make worker          # terminal 2: "worker <host> ready (decision engine: jev)"
make api             # terminal 3: Uvicorn running on http://127.0.0.1:8100
make smoke-dev TOPIC="How does Raft consensus work"     # terminal 4
```

Then exercise the API by hand in terminal 4:

```bash
TOKEN=$(head -n1 secrets/api_tokens | cut -d: -f2)
B=http://127.0.0.1:8100

curl -s -o /dev/null -w "%{http_code}\n" $B/jobs/00000000-0000-0000-0000-000000000000        # 401: no token
curl -s -H "Authorization: Bearer $TOKEN" $B/jobs/not-a-uuid | jq .detail[0].type          # uuid parsing error (422)
curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
     -d '{"topic":"a"}' $B/research | jq .detail[0].msg                                     # too short (422)
curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
     -d '{"topic":"best biryani in Bengaluru"}' $B/research                                 # 202; worker rejects it
```

Test ownership with a second user:

```bash
echo "mallory:$(openssl rand -hex 24)" >> secrets/api_tokens
# restart the API (Ctrl+C in terminal 3, then `make api`) so it reloads the tokens
M=$(tail -n1 secrets/api_tokens | cut -d: -f2)
curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $M" $B/jobs/<your-job-id>      # 404
```

Look at the queue:

```bash
make redis-cli
XINFO GROUPS jobs          # consumers, pending count (0 when idle), last-delivered-id
XRANGE jobs - + COUNT 5    # messages carry only job_id
```

---

## Step 5: Kill a worker mid-job

Start two workers (terminals 2 and 5), submit a job, and kill whichever one logs `running` for that job with Ctrl+C.

The message is now pending for the dead consumer. After 15 minutes, the other worker's `XAUTOCLAIM` takes it and logs `resuming job ... at (...)`. To see it sooner, temporarily change `min_idle_time=900_000` to `30_000` (30 seconds) in `worker/main.py`, restart the surviving worker, and watch. Put the value back afterwards.

---

## Verify

| Check | Pass condition |
|---|---|
| `make smoke-dev` | Status goes `queued` → `running` → `done`; PDF saved; JSON shows `faithfulness` |
| No token | `401` |
| Bad UUID / short topic | `422` |
| Non-technical topic | Job ends `rejected` |
| Another user's job | `404` |
| `XINFO GROUPS jobs` | `pending: 0` when idle |
| Worker killed mid-job | Another worker resumes it from its checkpoint |
| `jobs` table | Every job ends in a final status (`make psql`, `SELECT status, count(*) FROM jobs GROUP BY 1;`) |

**Troubleshooting:**

| Symptom | Fix |
|---|---|
| Job stays `queued` | No worker running, or it crashed at startup: check terminal 2. |
| `409 report not ready` for a done job | The API and worker disagree on `REPORTS_DIR`: run both from the repo root. |
| `429` | You hit `JOBS_PER_HOUR`; raise it in `.env` and restart the API. |
| Worker error `NOGROUP` | The stream was deleted (`FLUSHALL`?). Restart the worker; it recreates the group. |

**Commit:**

```bash
git add -A && git commit -m "ch09: FastAPI, Redis Streams queue, worker"
```

**Checkpoint questions:**

1. A worker finishes a job and crashes before `XACK`. Walk through what happens next, and why nothing runs twice.
2. Why does another user's job return 404 rather than 403?
3. What would go wrong if the API put the topic on the queue instead of the job id?
