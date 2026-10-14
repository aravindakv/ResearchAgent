.RECIPEPREFIX = >
PY := .venv/bin/python
DEV := docker compose -f docker-compose.yml -f docker-compose.dev.yml
ENGINE ?= jev
.PHONY: venv secrets infra-up infra-down psql redis-cli mcp run resume show graph api worker smoke-dev \
        up up-openjev down ps logs smoke isolation trust-ca backup reset eval-guardrail eval-benchmark

# ---- setup (chapter 01) ----
venv:
> uv venv .venv --python 3.12
> uv pip install --python $(PY) -r requirements-dev.txt

secrets:
> ./scripts/init-secrets.sh

# ---- host development (chapters 02-09) ----
infra-up:
> $(DEV) up -d postgres redis

infra-down:
> docker compose stop postgres redis

psql:
> docker compose exec postgres psql -U agent -d agent

redis-cli:
> docker compose exec redis sh -c 'redis-cli -a "$$(cat /run/secrets/redis_password)" --no-auth-warning'

mcp:
> $(PY) mcp_server/server.py

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

# ---- containers (chapter 10) ----
up:
> docker compose up -d --build

down:
> docker compose down

ps:
> docker compose ps

logs:
> docker compose logs -f --tail=100 worker mcp api

up-openjev:
> docker compose -f docker-compose.yml -f docker-compose.openjev.yml up -d --build

smoke:
> ./scripts/smoke-test.sh "$(TOPIC)"

isolation:
> ./scripts/check-isolation.sh

trust-ca:
> docker compose cp caddy:/data/caddy/pki/authorities/local/root.crt infra/caddy-root.crt
> @echo "Saved infra/caddy-root.crt"

backup:
> mkdir -p backups
> docker compose exec -T postgres pg_dump -U agent agent | gzip > backups/agent-$$(date +%F-%H%M).sql.gz
> @ls -lh backups | tail -n 3

reset:
> docker compose down -v

# ---- evaluation (chapter 11) ----
eval-guardrail:
> $(PY) evals/guardrail_eval.py --engine $(ENGINE)

eval-benchmark:
> $(PY) evals/benchmark_eval.py