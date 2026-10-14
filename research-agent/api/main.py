
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