
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