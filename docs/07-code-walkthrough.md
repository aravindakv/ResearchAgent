# 7. Code walkthrough

A file-by-file tour of the reference implementation, in the order a request touches it.

## `docker-compose.yml`

Defines seven services and five networks. The `x-hardening` anchor applies the same container hardening (read-only root, `/tmp` tmpfs, no capabilities, no privilege escalation, `.env` loading) to the API, worker, and MCP server. Read the `networks:` line of each service alongside the table in `04-security.md`; together they are the security design. `deploy.replicas: 2` on the worker demonstrates competing consumers. The `secrets:` block at the bottom maps files in `./secrets/` to `/run/secrets/<name>` inside only the services that list them.

## `infra/caddy/Caddyfile`

`tls internal` makes Caddy issue a certificate for `localhost` from its own local CA. The `header` block adds HSTS, `nosniff`, framing protection, and a strict referrer policy, and removes the `Server` header. `request_body max_size 64KB` stops oversized payloads before they reach Python.

## `infra/postgres/01-schema.sql` and `02-roles.sh`

The schema creates `jobs`, `chunks` (with an HNSW vector index and a generated `tsvector` with a GIN index), `eval_results`, and an empty `lg` schema. The roles script runs as the superuser at first start, reads passwords from secrets, creates `app_api` and `app_worker`, and grants each only what it needs. Making `app_worker` own `lg` and putting `lg` first in its `search_path` means LangGraph's `setup()` creates its checkpoint tables there.

## `api/main.py`

At import time the module opens a connection pool as `app_api`, connects to Redis, and loads the token file. `current_user` is a FastAPI dependency that every protected route declares; it parses `Authorization: Bearer ...` and compares against each known token with `hmac.compare_digest`. `enforce_rate_limit` uses a Redis key per user per clock hour. `ResearchRequest` normalizes whitespace and rejects non-printable characters. `submit` writes the job row before adding to the stream, so a worker never receives an ID that doesn't exist. `load_job` joins the faithfulness score and filters by owner. `get_pdf` builds the file path from a parsed UUID, which rules out path traversal.

## `worker/settings.py`

Reads secrets, loads the OpenAI and LangSmith keys into the process environment (the SDKs look for them there), turns tracing off if there is no LangSmith key, and builds the database connection string for `app_worker`.

## `worker/main.py`

`main` creates the consumer group, opens an `AsyncPostgresSaver`, runs `setup()`, and builds the graph once. The loop first tries `XAUTOCLAIM` to take over messages another consumer left unacknowledged for 15 minutes, then blocks on `XREADGROUP` for new ones. A message is acknowledged only after `run_job` returns, which gives at-least-once delivery.

`run_job` inspects the saved state for the thread. If the run already finished (a crash between finishing and acknowledging), it just records the status. If there is a pending next node, it resumes with `ainvoke(None, config)`. Otherwise it starts fresh. `asyncio.wait_for` enforces the total timeout, and any exception is logged in full but stored as only its type name.

## `worker/decisions.py`

All System 1 decisions live here behind four async methods: `check_topic`, `screen_chunks`, `research_sufficient`, and `paragraph_support`. `JevDecisions` implements them with `TypeSafeClassifier` and `Noul` questions; `LLMDecisions` implements them with structured output. `make_decisions` picks Jev when `TYPESAFE_API_KEY` is set. The module reads no secrets, so the eval scripts can import it on the host and test the same code the worker runs. The question wording and thresholds are module constants, so tuning happens in one place. Note that the LLM engine's `screen_chunks` keeps everything: one LLM call per chunk would be too slow and expensive, which is the clearest illustration of why a System One model is useful here.

## `worker/graph.py`

The module-level Pydantic models (`Plan`, `NewQueries`) are the structured-output schemas for the LLM's generation tasks. They avoid defaults and numeric constraints because some OpenAI structured-output modes reject them; clamping happens in code instead. The critic calls the LLM only when Jev says the sources are insufficient. `as_untrusted` wraps chunks in `<source>` tags, and `SOURCE_RULES` is appended to every prompt that sees web content.

`build_graph` connects to the MCP server once, indexes tools by name, and creates the decision engine; it returns the compiled graph together with the engine's name, which the worker adds to LangSmith metadata. `retrieve` embeds the query and orders chunks by cosine distance. The node functions follow the table in `01-architecture.md`. Note three details. The researcher hashes chunk text for deduplication. The writer builds the reference list from the same retrieved list it showed the model, so citation numbers line up. The evaluator retrieves with the same query and `k` as the writer so it judges against the same evidence.

The graph wiring at the bottom uses lambdas for simple routing and `route_after_critic` for the loop decision. `quality_fail` exists so the failing branch can set a status, since routing functions cannot modify state.

## `mcp_server/server.py`

`web_search` clamps its inputs, calls Tavily in a thread (the client is synchronous), keeps only http(s) results, truncates text, and returns JSON. `fetch_page` applies the SSRF guard through `safe_get`, which disables automatic redirects, re-validates each hop, limits size and content type, and extracts the main text with `trafilatura`. `render_pdf` converts Markdown, sanitizes with `nh3`, and renders with WeasyPrint using `deny_fetch`, all off the event loop. `BearerAuth` is a plain ASGI middleware: `/healthz` is open for Docker health checks, everything else needs the token, and lifespan events pass through so the MCP session manager starts correctly.

## `scripts/`

`init-secrets.sh` is idempotent: it never overwrites an existing secret. `smoke-test.sh` is a complete API client in about 30 lines of Bash and a good template for your own tests. `check-isolation.sh` turns the network design into assertions by attempting TCP connections from inside each container.

## `evals/`

`guardrail_eval.py` (with `--engine jev|llm`) and `benchmark_eval.py` both follow the LangSmith pattern: ensure a dataset exists, define a `target` function, define evaluators, call `evaluate`. See `05-evaluation.md`.
