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
    read -rsp "Enter $name (LangSmith and TypeSafe may be left empty): " value; echo
    printf '%s' "$value" > "secrets/$name"
  fi
done

# The directory is owner-only. Files are readable so non-root container users can read their
# bind mounts in chapter 10; the 700 directory still keeps other host users out.
chmod 644 secrets/*
echo "Secrets ready in ./secrets (directory mode 700)."