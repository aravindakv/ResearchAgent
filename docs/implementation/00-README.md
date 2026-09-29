# Research Agent: Build-It-Yourself Guide

This guide takes you from an empty folder to a working AI research agent. You type a technical topic, the agent checks that it is technical, researches it on the web in several rounds until a critic is satisfied, stores what it found in a vector database, writes a cited report, checks its own quality, and hands you a PDF.

You write every file yourself, chapter by chapter. Each chapter adds one idea on top of a system that already runs, so you always have something working to test. Every chapter ends with a **Verify** section. Don't move on until it passes.

---

## What you will have at the end

```
 you ──HTTPS──► Caddy ──► API (FastAPI) ──► Postgres + pgvector  (jobs, chunks, scores, checkpoints)
                            │
                            └──► Redis Stream "jobs"
                                      │
                                      ▼
                        Worker (LangGraph) ──► MCP server ──► web search, page fetch, PDF render
                          │    │
                          │    ├──► OpenAI (planning, writing, embeddings)
                          │    ├──► Jev / OpenJev (typed decisions: guardrail, screening, critic, quality gate)
                          │    └──► LangSmith (traces, datasets, experiments)
                          └──► reports volume ◄── API serves the PDF after an ownership check
```

**The flow you will test at the end:**

1. You submit "How does Raft consensus work?" with a bearer token. The API validates it, rate-limits you, stores a job, and queues its id.
2. A worker picks up the job and runs the LangGraph. It saves a checkpoint after every node, so a crashed run resumes where it stopped.
3. The guardrail rejects non-technical topics and prompt-injection attempts.
4. The planner writes search queries. The researcher searches through the MCP server, screens every chunk, embeds it, and stores it in pgvector.
5. The critic decides whether there is enough material and loops back with new queries if not.
6. The writer drafts a cited report from retrieved chunks. The evaluator checks every paragraph against the sources.
7. The MCP server renders the PDF, and you download it.

---

## Chapters (do them in order)

| # | File | What you build | Main concepts |
|---|---|---|---|
| 00 | `00-README.md` | This roadmap | – |
| 01 | `01-environment-and-repo.md` | Tools, repo skeleton, secrets, Python environment | Secrets as files, reproducible setup |
| 02 | `02-postgres-and-redis.md` | Postgres + pgvector and Redis in Docker, the schema | pgvector, HNSW, generated columns |
| 03 | `03-first-pipeline.md` | A linear script: topic → queries → search → report → PDF | Chat models, prompts, structured output, tracing |
| 04 | `04-langgraph-and-critic-loop.md` | The pipeline as a LangGraph with a critic loop and checkpoints | State graphs, conditional edges, resume after crash |
| 05 | `05-mcp-tool-server.md` | An MCP server for search, fetch and PDF; the graph uses it | MCP, tool design, service auth, SSRF |
| 06 | `06-rag-with-pgvector.md` | Chunking, embeddings, retrieval, citations | RAG, why big data stays out of graph state |
| 07 | `07-guardrails-and-quality-gate.md` | Input guardrail, per-paragraph evaluation, quality routing | Defense in depth, LLM-as-judge |
| 08 | `08-jev-decisions.md` | A Jev-compatible decision engine (TypeSafe Jev or open-source OpenJev), chunk screening | System 1 vs System 2, calibrated probabilities |
| 09 | `09-api-and-worker.md` | FastAPI + Redis Streams queue + worker process | Async architecture, at-least-once delivery |
| 10 | `10-containers-and-hardening.md` | Dockerfiles, Compose, networks, roles, Caddy, isolation tests | Least privilege, trust zones |
| 11 | `11-evaluation-with-langsmith.md` | Guardrail and benchmark evals, engine comparison, CI | Offline evals, regression testing |
| 12 | `12-operations-and-next-steps.md` | Resilience drills, backups, troubleshooting, stretch goals | Operating what you built |

**Suggested pace (part-time):** chapters 01–03 in week 1, 04–06 in weeks 2–3, 07–08 in week 4, 09–10 in week 5, 11–12 in week 6.

---

## How each chapter works

Every step has three parts:

- **Do:** the commands or the code.
- **Why:** the concept behind it. This is the part you are learning.
- **Verify:** a concrete check with the expected result.

### File captions and the extractor

Every file is introduced by a caption line that starts with `File:` followed by the path in backticks, directly above the code block. For example (shown quoted here so the extractor ignores it):

> File: `worker/example.py`
>
> ```python
> print("this block is the complete content of worker/example.py")
> ```

A `File:` caption always means **create or completely replace** that file. There are no partial snippets to merge by hand: when a chapter changes a file you wrote earlier, it gives you the whole new version, and explains what changed and why.

You can type the files yourself (recommended for the core logic, since you learn more), or write them with the extractor that ships with this guide:

```bash
cd ~/Documents/Learning/AI/GitHub/ResearchAgent/research-agent        # your repo root
GUIDE=../research-agent-guide                                          # where you unpacked this guide

python3 $GUIDE/extract_code.py $GUIDE/03-first-pipeline.md             # dry run: lists NEW / SAME / CHANGED files
python3 $GUIDE/extract_code.py $GUIDE/03-first-pipeline.md --write     # creates NEW files
python3 $GUIDE/extract_code.py $GUIDE/04-*.md --write --force          # also replaces CHANGED files
```

Always run the dry run first. `CHANGED` means the chapter replaces a file you already have. That's expected for files that evolve, like `worker/graph.py`, but if you have made your own edits, `--force` overwrites them. Commit before extracting, so `git diff` shows exactly what a chapter changed.

A few steps ask you to delete a file or run a command. The extractor never does those; the chapter tells you explicitly.

### Paths and commands

All paths are relative to the repo root, `research-agent/`. **Run every command from the repo root** unless a step says otherwise. Several settings (the secrets folder, the reports folder) are relative paths.

---

## Ports used on your machine

| Port | What | Chapters |
|---|---|---|
| 5432 | Postgres (localhost only) | 02–09, and later for host development |
| 6379 | Redis (localhost only) | 02–09, and later for host development |
| 8200 | MCP server while running on the host | 05–09 |
| 8100 | API while running on the host | 09 |
| 443 / 80 | Caddy (HTTPS), localhost only | 10+ |

If one of these is already taken on your machine (a local Postgres, another project's cluster), chapter 02 shows how to move it.

## Version baseline (September 2026)

Python 3.12 · LangGraph 0.6+ · LangChain core 0.3+ · MCP Python SDK 1.10+ · FastAPI 0.115+ · PostgreSQL 16 with pgvector · Redis 7 · Caddy 2 · Docker Compose v2.

These libraries move quickly. The requirements files use minimum versions so they install today. When something doesn't match (a renamed parameter, a moved import), the error message usually names it. Check the library's changelog, and note what you changed in your commit message.

## Accounts and keys you'll need

| Service | Needed from | Cost |
|---|---|---|
| OpenAI API | chapter 03 | Pay per use; set a monthly limit on the project |
| Tavily (web search) | chapter 03 | Free tier is enough |
| LangSmith (tracing, evals) | chapter 03 (optional but strongly recommended) | Free tier |
| A Jev-compatible server: TypeSafe Jev (early access) **or** OpenJev on Codiv (free tier) **or** self-hosted OpenJev | chapter 08 (optional) | Free options available |

## Relationship to the design docs

The earlier `docs/` package (architecture, security, evaluation, Jev, diagrams) explains **what** the finished system looks like and why. This guide is **how to build it yourself**. The chapters point to the relevant design sections and diagrams when a concept first appears. Keep `docs/images/` handy: the agent graph and topology diagrams are useful while you build.
