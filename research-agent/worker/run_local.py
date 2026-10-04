"""Run the graph from the command line.

    python worker/run_local.py "topic"            # new run
    python worker/run_local.py --resume THREAD_ID # continue an interrupted run
    python worker/run_local.py --show THREAD_ID   # print the saved state
    python worker/run_local.py --graph            # print the graph as Mermaid
"""

import argparse
import asyncio
import uuid

from graph import build_graph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from settings import DB_URL


def config_for(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 40, "run_name": "research-local"}

def summarize(values: dict) -> dict:
    out = {}
    for key, value in values.items():
        if isinstance(value, str) and len(value) > 120:
            out[key] = f"<{len(value)} chars>"
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            out[key] = f"<{len(value)} items>"
        else:
            out[key] = value
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("topic", nargs="?")
    ap.add_argument("--resume")
    ap.add_argument("--show")
    ap.add_argument("--graph", action="store_true")
    args = ap.parse_args()

    async with AsyncPostgresSaver.from_conn_string(DB_URL) as saver:
        await saver.setup()                     # creates the checkpoint tables on first use
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
            thread_id, payload = args.resume, None   # None = continue from the last checkpoint
        else:
            if not args.topic:
                ap.error("give a topic, --resume ID, --show ID or --graph")
            thread_id = str(uuid.uuid4())
            payload = {"thread_id": thread_id, "topic": args.topic, "iterations": 0}

        print("thread:", thread_id)
        async for update in graph.astream(payload, config_for(thread_id), stream_mode="updates"):
            for node in update:
                print("  finished:", node)
        final = (await graph.aget_state(config_for(thread_id))).values
        print(summarize(final))

if __name__ == "__main__":
    asyncio.run(main())

