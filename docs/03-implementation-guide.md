# 3. Build-it-yourself implementation guide

This guide builds the system in eight phases. Each phase produces something that runs, introduces a small set of concepts, and ends with a concrete check. Try each phase on your own first, then compare with the reference code named in the phase.

| Phase | You build | Main concepts |
|---|---|---|
| 0 | Project skeleton, Postgres + Redis in Docker | Tooling, pgvector |
| 1 | A linear script: topic → search → summary → PDF | LangChain chat models, prompts, structured output |
| 2 | The same flow as a LangGraph with a critic loop | State graphs, conditional edges, checkpointing |
| 3 | An MCP server for tools | MCP, tool design, service boundaries |
| 4 | RAG over pgvector with citations | Chunking, embeddings, retrieval, grounding |
| 5 | Guardrails, evaluation gate, LangSmith, Jev | Input safety, LLM-as-judge versus System One decisions, tracing, eval datasets |
| 6 | API, queue, worker, hardened Compose deployment | Async architecture, security, operations |
| 7 | Cloud deployment (optional) | Infrastructure as code, managed services |

Budget roughly a weekend per phase. Phases 2, 4, and 5 are where most of the learning happens, so take extra time there.

---

## Phase 0: Skeleton and infrastructure

**Goal.** A repository with a Python environment and running Postgres (with pgvector) and Redis.

**Build.** Create the folder layout (`api/`, `worker/`, `mcp_server/`, `infra/`, `docs/`, `evals/`). Install [uv](https://docs.astral.sh/uv/) and create a virtual environment with Python 3.12. Write a minimal `docker-compose.yml` containing only `postgres` (image `pgvector/pgvector:pg16`) and `redis`, with ports bound to `127.0.0.1` for now so you can connect from the host during development. Add `infra/postgres/01-schema.sql` with `CREATE EXTENSION vector;` and a `jobs` table.

**Key concepts.** Why pgvector rather than a separate vector database (see `01-architecture.md`). Why `127.0.0.1` port bindings matter: Docker's published ports bypass `ufw`.

**Reference.** `infra/postgres/01-schema.sql`, the `postgres` and `redis` services in `docker-compose.yml`.

**Done when.** `psql -h 127.0.0.1 -U agent -c "SELECT extname FROM pg_extension"` lists `vector`, and `redis-cli ping` returns `PONG`.

---

## Phase 1: A linear pipeline

**Goal.** A command-line script, `python research.py "How does TCP congestion control work"`, that produces a PDF.

**Build.**

1. Call Tavily directly with the `tavily-python` client to get 5 results with raw content.
2. Create a `ChatOpenAI` model and write a prompt that turns the search results into a Markdown explainer with `[n]` citations.
3. Use `with_structured_output` with a Pydantic model to generate the search queries first (a list of 3–5 strings), instead of searching for the raw topic.
4. Convert the Markdown to HTML with the `markdown` package and render it with WeasyPrint.

**Key concepts.** Chat model interfaces and message types (`SystemMessage`, `HumanMessage`). Structured output: the model returns a validated Pydantic object rather than text you must parse. Prompt design: telling the model to cite, to paraphrase rather than copy, and to use only the given sources.

**Reference.** The `planner` and `writer` nodes in `worker/graph.py`; `render_pdf` in `mcp_server/server.py`.

**Done when.** The script produces a readable PDF with a references section, and you can explain where each step's latency and cost come from. Enable LangSmith now (`LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`) and look at the trace; you'll use it in every later phase.

**Stretch.** Measure tokens and cost per run from the trace. Try two different models and compare the output.

---

## Phase 2: LangGraph with a critic loop

**Goal.** Rebuild phase 1 as a state graph where a critic decides whether the research is sufficient, looping back to research with new queries if it isn't.

**Build.**

1. Define a `State` `TypedDict` (start with `topic`, `queries`, `results`, `iterations`, `sufficient`, `draft`).
2. Write each step as an async node function that takes the state and returns a partial update.
3. Add a `critic` node that returns structured output `{sufficient: bool, new_queries: list[str]}`.
4. Add a routing function and a conditional edge:

```python
def route_after_critic(state: State) -> str:
    done = state["sufficient"] or not state["queries"] or state["iterations"] >= 2
    return "writer" if done else "researcher"

g.add_conditional_edges("critic", route_after_critic)
```

5. Compile with a checkpointer. Start with `InMemorySaver`, then switch to `AsyncPostgresSaver` from `langgraph-checkpoint-postgres`, and pass `{"configurable": {"thread_id": "..."}}` when invoking.
6. Set `recursion_limit` in the config as a second safety net against runaway loops.

**Key concepts.** Nodes return partial state updates, which are merged into the state. Conditional edges are how agents make decisions. Checkpoints are written after every node, which makes runs resumable and inspectable. The difference between the loop cap in your routing logic and `recursion_limit`.

**Exercises.** Print the graph as Mermaid with `print(graph.get_graph().draw_mermaid())`. Kill the process halfway through a run, then resume by calling `ainvoke(None, config)` with the same `thread_id`, and confirm it continues from the last node. Use `aget_state(config)` to inspect the saved state.

**Reference.** `worker/graph.py` (graph construction at the bottom), `run_job` in `worker/main.py` (resume logic).

**Done when.** LangSmith shows a run where the critic sent the flow back at least once, and a killed run resumes without repeating completed nodes.

**Stretch: per-section parallel research.** Have the planner return an outline of sections, then fan out with `Send`. Parallel branches writing to the same key need a reducer:

```python
import operator
from typing import Annotated
from langgraph.types import Send

class State(TypedDict, total=False):
    outline: list[dict]
    section_notes: Annotated[list[dict], operator.add]   # branches append; no overwrites

def fan_out(state: State):
    return [Send("research_section", {"job_id": state["job_id"], "section": s}) for s in state["outline"]]

g.add_conditional_edges("planner", fan_out, ["research_section"])
```

**Stretch: clarification with human in the loop.** For ambiguous topics, pause the graph and ask the user:

```python
from langgraph.types import Command, interrupt

async def clarify(state: State) -> dict:
    answer = interrupt({"question": f"Did you mean {state['candidates']}?"})
    return {"topic": answer}

# Later, when the user replies:
await graph.ainvoke(Command(resume="Rust, the programming language"), config)
```

---

## Phase 3: The MCP tool server

**Goal.** Move all external capabilities (search, fetch, PDF rendering) out of the graph into an MCP server, and have the graph use them only through MCP.

**Build.**

1. Create `mcp_server/server.py` with `FastMCP("research-tools")` and decorate async functions with `@mcp.tool()`. Type hints and docstrings become the tool schema and description the LLM sees, so write them carefully.
2. Return JSON strings for structured results, so every client receives the same text content.
3. Serve it over streamable HTTP by running `mcp.streamable_http_app()` with uvicorn.
4. Wrap the ASGI app in a small bearer-token middleware so only the worker can call it.
5. In the worker, load the tools:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "research": {"transport": "streamable_http", "url": "http://localhost:8000/mcp",
                 "headers": {"Authorization": f"Bearer {token}"}},
})
tools = {t.name: t for t in await client.get_tools()}
result = await tools["web_search"].ainvoke({"query": "raft leader election"})
```

6. Add `fetch_page` with an SSRF guard (see `04-security.md`) before you ever let it fetch model-chosen URLs.

**Key concepts.** The difference between orchestration and capabilities. Tool design: small, single-purpose tools with bounded inputs (`max_results` clamped, query length capped) and bounded outputs (text truncated). Service-to-service authentication.

**Exercises.** Connect the [MCP Inspector](https://github.com/modelcontextprotocol/inspector) (`npx @modelcontextprotocol/inspector`) to your server, add the `Authorization` header, and call each tool by hand. Connect the same server to another MCP client to see the reuse benefit.

**Reference.** `mcp_server/server.py`, `build_graph` in `worker/graph.py`.

**Done when.** The graph no longer imports Tavily, httpx, or WeasyPrint; all of that lives behind MCP. An unauthenticated request to the MCP server returns 401.

---

## Phase 4: RAG over pgvector

**Goal.** Instead of passing raw search results to the writer, store chunked, embedded content in Postgres and retrieve what's relevant, with citations traceable to specific chunks.

**Build.**

1. Add a `chunks` table: `job_id`, `url`, `content`, `content_hash`, `embedding vector(1536)`, with `UNIQUE (job_id, content_hash)` and an HNSW index (`USING hnsw (embedding vector_cosine_ops)`).
2. In the researcher, split text with `RecursiveCharacterTextSplitter` (start at 3000 characters with 300 overlap), embed with `OpenAIEmbeddings`, and insert with `ON CONFLICT DO NOTHING` to deduplicate.
3. Write a `retrieve(job_id, query, k)` helper that orders by `embedding <=> query_vector` (cosine distance).
4. Have the critic and writer use `retrieve` rather than holding results in state.
5. Label retrieved chunks as numbered sources in the prompt and build the references list from the same list, so `[n]` always maps to a real URL.

**Key concepts.** Chunk size trade-offs (small chunks retrieve precisely but lose context). Why the embedding dimension in the column must match the model. Approximate nearest-neighbour indexes. Why large data belongs in the database rather than in checkpointed graph state.

**Exercises.** Run the same topic with chunk sizes of 1000, 3000, and 6000 characters and compare the evaluation scores from phase 5. Run `EXPLAIN ANALYZE` on the retrieval query to see the index being used.

**Reference.** `researcher`, `retrieve`, and `writer` in `worker/graph.py`; `chunks` in `infra/postgres/01-schema.sql`.

**Done when.** Every `[n]` in the PDF points to a URL whose chunk actually contains the cited information (spot-check five).

**Stretch: hybrid search.** Add a generated `tsvector` column with a GIN index, then combine keyword and vector rankings with reciprocal rank fusion. Keyword search catches exact names like `SO_REUSEPORT` or version numbers that embeddings tend to blur:

```sql
WITH v AS (
  SELECT id, row_number() OVER (ORDER BY embedding <=> %(vec)s::vector) AS r
  FROM chunks WHERE job_id = %(job)s ORDER BY embedding <=> %(vec)s::vector LIMIT 40
), k AS (
  SELECT id, row_number() OVER (ORDER BY ts_rank(tsv, q) DESC) AS r
  FROM chunks, plainto_tsquery('english', %(text)s) q
  WHERE job_id = %(job)s AND tsv @@ q ORDER BY ts_rank(tsv, q) DESC LIMIT 40
)
SELECT c.url, c.content
FROM chunks c
JOIN (SELECT id, SUM(1.0 / (60 + r)) AS score
      FROM (SELECT * FROM v UNION ALL SELECT * FROM k) u GROUP BY id) s USING (id)
ORDER BY s.score DESC LIMIT 10;
```

**Stretch: semantic cache.** Add a `topics` table with an embedding of the normalized topic. Before researching, look for a topic with cosine similarity above about 0.92 researched in the last 30 days, and reuse its chunks.

**Stretch: reranking.** Retrieve 20–30 candidates and rerank to the best 6–8 with a cross-encoder or a reranking API.

---

## Phase 5: Guardrails, evaluation gate, and LangSmith

**Goal.** Reject non-technical and unsafe input, refuse to ship low-quality reports, and measure quality systematically.

**Build.**

1. **Input guardrail node.** Call OpenAI moderation, then a structured classifier returning `is_technical`, `confidence`, and `normalized_topic`. State in the system prompt that the user text is data to classify, not instructions. Route rejections to `END` with a status.
2. **Evaluation node.** Ask a judge model for the fraction of claims supported by the sources plus a list of unsupported claims. Store the score in `eval_results`.
3. **Routing on quality.** Pass → render. Middle band → render with a visible low-confidence banner. Below the minimum → fail.
4. **Offline evals.** Build a labelled guardrail dataset and a benchmark topic set in LangSmith, and run experiments with `langsmith.evaluate()`. See `05-evaluation.md` and the scripts in `evals/`.

**Key concepts.** Defence in depth for inputs. The difference between generation (System 2) and typed decisions (System 1), and why calibrated probabilities make thresholds meaningful. LLM-as-judge and its limits (judges are biased toward fluent text; calibrate them against your own spot checks). Online gates versus offline regression testing. Why every prompt change should go through the offline evals.

5. **Move decisions to Jev.** Put the four decisions (topic check, chunk screening, research sufficiency, paragraph support) behind one interface with an LLM implementation first. Then add a Jev implementation using `langchain-typesafe`'s `TypeSafeClassifier` and `Noul` questions, and select the engine with an environment variable. Change the critic so the LLM is called only when new queries are needed. See `08-jev-decision-model.md`.
6. **Compare the engines.** Run the guardrail eval with each engine and a few benchmark topics with each, and compare accuracy, latency, and cost in LangSmith.

**Reference.** `worker/decisions.py`; `guardrail`, `critic`, `evaluator`, `render` in `worker/graph.py`; `evals/guardrail_eval.py`, `evals/benchmark_eval.py`.

**Done when.** The guardrail eval reports accuracy and false-accept rate for both engines, the benchmark shows a faithfulness score per topic, and you can explain from your own measurements when Jev is worth using and when it isn't.

**Stretch: reviewer loop.** Add `reviewer` (rubric: accuracy against sources, clarity, structure, depth) and `reviser` nodes between `writer` and `evaluator`, capped at two rounds. Feed the evaluator's unsupported claims to the reviser when the gate fails the first time.

**Stretch: Jev for model routing.** Use a Jev `Choice` (or LangChain's `ModelRouterMiddleware`) to send simple topics to a cheaper writer model and complex ones to a stronger model.

**Stretch: composite quality score.** Ask Jev several `Score` questions about the draft (depth, clarity, structure), each with descriptive levels, normalize each by its top level, and combine them with weights in code.

**Stretch: more metrics.** Citation coverage (share of paragraphs with a valid `[n]`), outline completeness (every "must answer" statement addressed), copy detection (longest verbatim run shared with any chunk stays under 25 words), and RAGAS faithfulness and response relevancy.

---

## Phase 6: Services, queue, and hardened deployment

**Goal.** Turn the script into a multi-user service: an API that accepts jobs, workers that run them, and a Compose deployment with real security boundaries.

**Build.**

1. **API.** FastAPI with `POST /research`, `GET /jobs/{id}`, `GET /jobs/{id}/pdf`, `GET /healthz`. Bearer-token auth with constant-time comparison. Pydantic validation. A Redis counter for per-user hourly limits. Every lookup filtered by `user_id`, returning 404 for other users' jobs.
2. **Queue.** Redis Streams with a consumer group: `XADD` in the API; `XREADGROUP`, `XACK`, and `XAUTOCLAIM` in the worker so unacknowledged jobs from crashed workers are picked up.
3. **Worker.** A loop that consumes jobs, resumes from checkpoints when present, enforces a total timeout, and records only the error type in the user-visible status.
4. **Secrets.** Compose `secrets:` mounted as files, read at startup; nothing in images or `.env`.
5. **Networks.** `public` (Caddy only), and internal `edge`, `data`, `tools` networks, plus `egress` for the worker and MCP server only.
6. **Database roles.** A separate least-privilege role for the API and the worker.
7. **Container hardening.** Non-root user, `read_only: true` with a `/tmp` tmpfs, `cap_drop: [ALL]`, `no-new-privileges`.
8. **Reverse proxy.** Caddy with `tls internal`, security headers, and a request body limit.

**Key concepts.** Why long jobs need a queue. At-least-once delivery and idempotency (which is why checkpoint resume and `ON CONFLICT` inserts matter). Least privilege at the network, database, and container levels.

**Reference.** Everything: `docker-compose.yml`, `api/main.py`, `worker/main.py`, `infra/`, `scripts/`. Follow `02-setup-local.md` to run it.

**Done when.** `make isolation` reports all `ok`, `make smoke` produces a PDF, and a job survives a worker restart.

**Stretch.** Server-sent events for progress (`GET /jobs/{id}/events` streaming node names from the worker via Redis pub/sub). A small chat UI. Keycloak for OIDC login.

---

## Phase 7: Cloud (optional)

Follow `06-cloud-deployment.md`. The application code barely changes; the queue, file storage, secrets, and authentication swap to managed services.

---

## Learning checklist

By the end you should be able to explain, without notes, each of these: how a LangGraph conditional edge decides the next node; what a checkpoint contains and how resume works; what MCP standardizes and what it doesn't; how an HNSW index finds neighbours approximately; why hybrid search beats pure vector search for technical text; what indirect prompt injection is and how this design limits it; what LLM-as-judge can and can't tell you; and why the system uses a queue instead of a long HTTP request.
