"""Worker configuration.

Works in two places without changes:
* on your machine: reads .env (via python-dotenv) and secrets from ./secrets
* in a container (chapter 10): Compose sets the environment and mounts secrets at /run/secrets
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

SECRETS_DIR = os.path.abspath(os.getenv("SECRETS_DIR", "./secrets"))

def secret(name: str, required: bool = True) -> str:
    """Read one secret file. Secrets are never taken from environment variables."""
    path = os.path.join(SECRETS_DIR, name)
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read().strip()
    if required:
        raise RuntimeError(f"missing secret {path}: run `make secrets`")
    return ""

# SDKs (OpenAI, Tavily, LangSmith, TypeSafe) read their keys from the environment.
# Load them into THIS process only; nothing else ever sees them.
for _file, _env in {"openai_api_key": "OPENAI_API_KEY", "tavily_api_key": "TAVILY_API_KEY",
                    "langsmith_api_key": "LANGSMITH_API_KEY", "typesafe_api_key": "TYPESAFE_API_KEY"}.items():
    _value = secret(_file, required=False)
    if _value:
        os.environ[_env] = _value
if not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "false"

CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "CHANGE_ME")
if CHAT_MODEL == "CHANGE_ME":
    raise RuntimeError("set OPENAI_CHAT_MODEL in .env to a chat model available on your account")
EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")

DB_URL = (f"host={os.environ.get('POSTGRES_HOST', '127.0.0.1')} "
          f"port={os.environ.get('POSTGRES_PORT', '5432')} "
          f"dbname={os.environ.get('POSTGRES_DB', 'agent')} "
          f"user={os.environ.get('DB_USER', 'agent')} "
          f"password={secret(os.environ.get('DB_PASSWORD_SECRET', 'pg_password'))}")

REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PASSWORD = secret("redis_password")

MCP_URL = os.environ.get("MCP_URL", "http://127.0.0.1:8200/mcp")
MCP_TOKEN = secret("mcp_token")

REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", "./reports"))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

JOB_TIMEOUT_S = int(os.environ.get("JOB_TIMEOUT_S", "900"))