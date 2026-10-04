.RECIPEPREFIX = >
PY := .venv/bin/python
PY_MCP := .venv-mcp/bin/python
.PHONY: venv venv-mcp secrets infra-up infra-down psql redis-cli mcp run resume show graph api worker smoke-dev

# ---- setup ----
venv:
> uv venv .venv --python 3.12 --clear
> uv pip install --python $(PY) -r requirements-dev.txt

venv-mcp:
> uv venv .venv-mcp --python 3.12 --clear
> uv pip install --python $(PY_MCP) -r mcp_server/requirements.txt

secrets:
> ./scripts/init-secrets.sh

# ---- infrastructure (chapter 02) ----
infra-up:
> docker compose up -d postgres redis

infra-down:
> docker compose down

psql:
> docker compose exec postgres psql -U agent -d agent

redis-cli:
> docker compose exec redis sh -c 'redis-cli -a "$$(cat /run/secrets/redis_password)" --no-auth-warning'

# ---- host development (chapters 04-09) ----
mcp:
> $(PY_MCP) mcp_server/server.py

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