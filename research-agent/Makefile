.RECIPEPREFIX = >
PY := .venv/bin/python
.PHONY: venv secrets

venv:
> uv venv .venv --python 3.12
> uv pip install --python $(PY) -r requirements-dev.txt

secrets:
> ./scripts/init-secrets.sh