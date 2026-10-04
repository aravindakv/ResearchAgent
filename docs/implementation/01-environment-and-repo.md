# 01 — Environment and Repository

**Goal:** a machine with every tool at a known version, an empty repository with the right layout, generated secrets, and a Python environment with all the libraries you'll use.

---

## Step 1: Install the system tools

**Do:**

```bash
sudo apt-get update
sudo apt-get install -y git curl jq make openssl ca-certificates \
     libpango-1.0-0 libpangoft2-1.0-0 fonts-dejavu-core
```

Install **uv**, a fast Python package and version manager, and use it to install Python 3.12:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc          # or open a new terminal so `uv` is on PATH
uv python install 3.12
```

Install **Docker Engine** with the Compose plugin from Docker's repository (the Ubuntu `docker.io` package lags behind):

```bash
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
 | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

If you already have Docker working (for example from another project), skip this.

**Why:**

- `libpango` and the DejaVu fonts are what WeasyPrint needs to lay out PDFs. Without them, the PDF step in chapter 03 fails with an `OSError` about a missing library.
- `uv` replaces `pip` + `venv` + `pyenv`, and it's much faster. More importantly, it makes the Python version explicit, so "works on my machine" problems from a different system Python disappear.
- Docker from Docker's own repository gets you Compose v2 (`docker compose`, with a space).

---

## Step 2: Get your API keys

**Do:** create these now, so later chapters don't stall:

- **OpenAI:** create a project API key at platform.openai.com. Set a **monthly spend limit** on the project.
- **Tavily:** sign up at tavily.com and copy the API key.
- **LangSmith:** sign up at smith.langchain.com and create an API key. Optional, but tracing is half the learning value.
- **A Jev-compatible server:** only needed in chapter 08, and optional. Either request TypeSafe early access, or create a free Codiv account for the open-source OpenJev (no card needed). Chapter 08 compares the options.

**Why:** you'll store them as files in step 5, never in code or in `.env`.

---

## Step 3: Create the repository

**Do:**

```bash
mkdir -p ~/Documents/Learning/AI/GitHub/ResearchAgent
cd ~/Documents/Learning/AI/GitHub/ResearchAgent
unzip ~/Downloads/research-agent-guide.zip          # the guide, next to your repo
mkdir research-agent && cd research-agent
git init
mkdir -p worker api mcp_server infra/postgres infra/caddy scripts evals/datasets reports
```

Use any location you like; the guide only assumes that the guide folder sits next to the repo, so `../research-agent-guide` works.

**Final layout** (built up over the chapters):

```
research-agent/
├── worker/          # LangGraph agent: settings, graph, decisions, queue consumer (ch 03-09)
├── mcp_server/      # MCP tool server: search, fetch, PDF (ch 05)
├── api/             # FastAPI service (ch 09)
├── infra/
│   ├── postgres/    # schema and roles (ch 02, 10)
│   └── caddy/       # reverse proxy config (ch 10)
├── evals/           # LangSmith evaluation scripts and datasets (ch 11)
├── scripts/         # secrets, smoke test, isolation checks
├── secrets/         # generated secrets and API keys (never committed)
├── reports/         # generated PDFs during host development (never committed)
├── docker-compose.yml
├── Makefile
└── .env             # non-secret configuration (never committed)
```

---

## Step 4: Base files

File: `.gitignore`

```gitignore
# Local configuration and secrets: never commit these
.env
secrets/
backups/

# Generated output
reports/
report-*.pdf
infra/caddy-root.crt

# Python
.venv/
.venv-mcp/
__pycache__/
*.pyc
```

File: `.env.example`

```bash
# Non-secret configuration for HOST development (chapters 03-09).
# Copy to .env and edit. Secrets never go here: they live as files in ./secrets/.
# In chapter 10 the containers read this file too, but docker-compose.yml overrides the
# host-specific values (hosts, paths, bind addresses).

SECRETS_DIR=./secrets

POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5432
POSTGRES_DB=agent

REDIS_HOST=127.0.0.1

MCP_BIND=127.0.0.1
MCP_PORT=8200
MCP_URL=http://127.0.0.1:8200/mcp

REPORTS_DIR=./reports

# Set to a current OpenAI chat model available on your account.
OPENAI_CHAT_MODEL=CHANGE_ME
OPENAI_EMBED_MODEL=text-embedding-3-small

LANGSMITH_TRACING=true
LANGSMITH_PROJECT=research-agent-dev

JOBS_PER_HOUR=10
JOB_TIMEOUT_S=900
ENABLE_DOCS=1

# Decision engine (chapter 08). Leave these commented out for TypeSafe's hosted Jev,
# or uncomment for OpenJev (see chapter 08 for all options).
# TYPESAFE_BASE_URL=https://api.codiv.ai
# TYPESAFE_MODEL=openjev-latest
# TYPESAFE_BASE_URL_LONG=
# TYPESAFE_MODEL_LONG=
```

File: `requirements-dev.txt`

```text
# Everything needed to run all components on the host during development.
# Each service gets its own smaller requirements.txt in chapter 10.

# Agent
langgraph>=0.6
langgraph-checkpoint-postgres>=2.0
langchain-core>=0.3
langchain-openai>=0.3
langchain-mcp-adapters>=0.1
langchain-text-splitters>=0.3
langsmith>=0.3
openai>=1.40
tavily-python>=0.5

# Data
psycopg[binary]>=3.2
psycopg-pool>=3.2
redis>=5.0
pydantic>=2.7
python-dotenv>=1.0

# API
fastapi>=0.115
uvicorn[standard]>=0.30

# MCP server
mcp>=1.10,<2                # client side: langchain-mcp-adapters requires mcp 1.x (server: chapter 05)
starlette>=0.37
httpx>=0.27
trafilatura>=1.12
markdown>=3.6
nh3>=0.2
weasyprint>=62
```

**Why a single dev requirements file now:** during chapters 03–09 you run every component on your machine from one virtual environment, which keeps the feedback loop short. In chapter 10 each service gets its own minimal `requirements.txt` for its container image. That split matters for security (less code in each image) and image size.

---

## Step 5: Secrets as files

**Concept:** there are two kinds of configuration.

- **Settings** (hosts, ports, model names) are not secret. They go in `.env`.
- **Secrets** (API keys, passwords, tokens) go in individual files under `secrets/`, one value per file.

Why files instead of environment variables? Environment variables leak easily: they show up in `docker inspect`, in crash reports, in child processes, and in `/proc/<pid>/environ`. Files can be permission-restricted, and Docker mounts them at `/run/secrets/<name>` inside containers. By using files from day one, your code reads secrets the same way on your laptop (`./secrets/`) and in containers (`/run/secrets/`); only `SECRETS_DIR` changes.

File: `scripts/init-secrets.sh`

```bash
#!/usr/bin/env bash
# Generates internal secrets and prompts for third-party API keys.
# Safe to re-run: it never overwrites a secret that already exists.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p secrets
chmod 700 secrets

gen() { [ -s "secrets/$1" ] || openssl rand -hex 32 > "secrets/$1"; }
for name in pg_password pg_api_password pg_worker_password redis_password mcp_token; do gen "$name"; done
[ -s secrets/api_tokens ] || echo "dev:$(openssl rand -hex 24)" > secrets/api_tokens

for name in openai_api_key tavily_api_key langsmith_api_key typesafe_api_key; do
  if [ ! -e "secrets/$name" ]; then
    read -rsp "Enter $name (LangSmith and TypeSafe/Codiv may be left empty): " value; echo
    printf '%s' "$value" > "secrets/$name"
  fi
done

# The directory is owner-only. Files are readable so non-root container users can read their
# bind mounts in chapter 10; the 700 directory still keeps other host users out.
chmod 644 secrets/*
echo "Secrets ready in ./secrets (directory mode 700)."
```

**Why each generated secret exists** (you'll use them in later chapters):

| File | Used for | From chapter |
|---|---|---|
| `pg_password` | Postgres superuser | 02 |
| `redis_password` | Redis authentication | 02 |
| `mcp_token` | Worker → MCP server authentication | 05 |
| `api_tokens` | User tokens for the API (`user:token` per line) | 09 |
| `pg_api_password`, `pg_worker_password` | Least-privilege database roles | 10 |

---

## Step 6: Makefile and README

File: `Makefile`

```makefile
.RECIPEPREFIX = >
PY := .venv/bin/python
.PHONY: venv secrets

venv:
> uv venv .venv --python 3.12
> uv pip install --python $(PY) -r requirements-dev.txt

secrets:
> ./scripts/init-secrets.sh
```

`.RECIPEPREFIX = >` lets recipe lines start with `>` instead of a tab. That avoids the classic "missing separator" error when an editor converts tabs to spaces.

File: `README.md`

```markdown
# Research Agent

A learning project: an AI agent that researches a technical topic and returns a cited PDF.
Built step by step following `research-agent-guide/`.

## Quick reference

    make venv        # Python environment
    make secrets     # generate secrets, enter API keys
```

---

## Step 7: Create everything

**Do:**

```bash
chmod +x scripts/init-secrets.sh
make secrets                  # paste your keys when prompted (input is hidden)
cp .env.example .env
nano .env                     # set OPENAI_CHAT_MODEL to a current model on your account
make venv                     # takes a minute or two
```

---

## Verify

```bash
ls -la secrets/                                   # 10 files; directory shows drwx------
wc -c secrets/openai_api_key secrets/tavily_api_key   # both non-zero
grep OPENAI_CHAT_MODEL .env                       # not CHANGE_ME
.venv/bin/python --version                        # Python 3.12.x
.venv/bin/python -c "import langgraph, langchain_openai, mcp, fastapi, weasyprint; print('imports ok')"
docker compose version                            # v2.x
git status --short                                # secrets/, .env and .venv/ must NOT appear
```

| Check | Pass condition |
|---|---|
| `ls -la secrets/` | 10 files, directory mode `drwx------` |
| imports | prints `imports ok` |
| `git status` | shows `.gitignore`, `.env.example`, `Makefile`, `README.md`, `requirements-dev.txt`, `scripts/`, but never `secrets/` or `.env` |

**If `import weasyprint` fails** with an error about `libpango` or `libgobject`, the system libraries from step 1 are missing. Install them and retry.

**Commit:**

```bash
git add -A && git commit -m "ch01: repo skeleton, secrets, dev environment"
```

**Checkpoint questions:**

1. Name two ways a secret in an environment variable can leak that a file inside a mode-700 directory avoids.
2. Why does `init-secrets.sh` refuse to overwrite existing secrets?
