
"""Run the graph from the command line.

    python worker/run_local.py "topic"            # new run (creates a job row)
    python worker/run_local.py --resume JOB_ID    # continue an interrupted run
    python worker/run_local.py --show JOB_ID      # print the saved state
    python worker/run_local.py --graph            # print the graph as Mermaid
"""
import argparse
import asyncio
import uuid

import psycopg
from graph import build_graph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from settings import DB_URL


def config_for(job_id: str) -> dict:
    return {"configurable": {"thread_id": job_id}, "recursion_limit": 40, "run_name": "research-local"}


def summarize(values: dict) -> dict:
    return {k: (f"<{len(v)} chars>" if isinstance(v, str) and len(v) > 120 else v) for k, v in values.items()}


async def create_job(topic: str) -> str:
    job_id = str(uuid.uuid4())
    async with await psycopg.AsyncConnection.connect(DB_URL) as conn:
        await conn.execute("INSERT INTO jobs (id, user_id, topic, status) VALUES (%s, 'cli', %s, 'running')",
                           (job_id, topic))
    return job_id


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("topic", nargs="?")
    ap.add_argument("--resume")
    ap.add_argument("--show")
    ap.add_argument("--graph", action="store_true")
    args = ap.parse_args()

    async with AsyncPostgresSaver.from_conn_string(DB_URL) as saver:
        await saver.setup()
        graph = await build_graph(saver)

        if args.graph:
            print(graph.get_graph().draw_mermaid())
            return
        if args.show:
            snapshot = await graph.aget_state(config_for(args.show))
            print("next:", snapshot.next)
            print(summarize(snapshot.values))
            return
        if args.resume:
            job_id, payload = args.resume, None
        else:
            if not args.topic:
                ap.error("give a topic, --resume ID, --show ID or --graph")
            job_id = await create_job(args.topic)
            payload = {"job_id": job_id, "topic": args.topic, "iterations": 0}

        print("job:", job_id)
        async for update in graph.astream(payload, config_for(job_id), stream_mode="updates"):
            for node in update:
                print("  finished:", node)
        print(summarize((await graph.aget_state(config_for(job_id))).values))


if __name__ == "__main__":
    asyncio.run(main())