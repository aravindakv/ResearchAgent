"""MCP tool server: the only component that reads the open web.

Tools: web_search, fetch_page, render_pdf. Every HTTP request except /healthz needs the service token.
Runs on the host during development (127.0.0.1:8200) and in a container from chapter 10 (0.0.0.0:8000).
"""
import asyncio
import base64
import hmac
import ipaddress
import json
import os
import socket
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
import markdown as md_lib
import nh3
import trafilatura
import uvicorn
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from starlette.responses import JSONResponse, PlainTextResponse
from tavily import TavilyClient
from weasyprint import HTML

load_dotenv()
SECRETS_DIR = Path(os.environ.get("SECRETS_DIR", "/run/secrets"))
BIND = os.environ.get("MCP_BIND", "127.0.0.1")
PORT = int(os.environ.get("MCP_PORT", "8200"))

def secret(name: str) -> str:
    return (SECRETS_DIR / name).read_text().strip()

MAX_BYTES = 2_000_000
MAX_TEXT = 20_000
USER_AGENT = "research-agent/0.1 (+local learning project)"
ALLOWED_TAGS = {"h1", "h2", "h3", "h4", "p", "ul", "ol", "li", "strong", "em", "code", "pre", "blockquote",
                "table", "thead", "tbody", "tr", "th", "td", "a", "br", "hr"}
PDF_CSS = """
@page { size: A4; margin: 2cm; @bottom-center { content: counter(page); font-size: 9pt; } }
body { font-family: 'DejaVu Sans', sans-serif; font-size: 10.5pt; line-height: 1.5; }
h1 { font-size: 20pt; } h2 { font-size: 14pt; margin-top: 1.4em; }
pre, code { font-family: 'DejaVu Sans Mono', monospace; font-size: 9pt; }
pre { background: #f4f4f4; padding: 8px; white-space: pre-wrap; }
blockquote { border-left: 3px solid #c77; margin: 0; padding-left: 10px; color: #733; }
table { border-collapse: collapse; } td, th { border: 1px solid #ccc; padding: 4px; }
"""

tavily = TavilyClient(api_key=secret("tavily_api_key"))
mcp = MCPServer("research-tools")


# ---------------------------------------------------------------- SSRF guard
def assert_public_url(url: str) -> None:
    """Allow only http(s) URLs whose host resolves exclusively to public IP addresses.
    Raises ToolError: an anticipated refusal whose message the client is allowed to see. Any other
    exception is reported to the client as a generic error, with details only in the server log.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ToolError("only http(s) URLs are allowed")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    infos = socket.getaddrinfo(parsed.hostname, port, proto=socket.IPPROTO_TCP)
    if not infos:
        raise ToolError("host does not resolve")
    for info in infos:
        if not ipaddress.ip_address(info[4][0]).is_global:
            raise ToolError("non-public address blocked")

async def safe_get(url: str) -> str:
    async with httpx.AsyncClient(follow_redirects=False, timeout=10, headers={"User-Agent": USER_AGENT}) as client:
        for _ in range(4):                               # re-validate every redirect hop
            assert_public_url(url)
            async with client.stream("GET", url) as resp:
                if resp.is_redirect:
                    url = urljoin(url, resp.headers.get("location", ""))
                    continue
                resp.raise_for_status()
                if not resp.headers.get("content-type", "").startswith(("text/html", "text/plain")):
                    raise ToolError("unsupported content type")
                body = bytearray()
                async for chunk in resp.aiter_bytes():
                    body += chunk
                    if len(body) > MAX_BYTES:
                        break
                return body.decode(resp.encoding or "utf-8", errors="replace")
    raise ToolError("too many redirects")

# ---------------------------------------------------------------- tools
@mcp.tool()
async def web_search(query: str, max_results: int = 4) -> str:
    """Search the web. Returns a JSON list of {url, title, content}."""
    query = " ".join(query.split())[:300]
    max_results = max(1, min(int(max_results), 8))
    result = await asyncio.to_thread(tavily.search, query=query, max_results=max_results, include_raw_content=True)
    hits = []
    for item in result.get("results", []):
        url = item.get("url", "")
        text = (item.get("raw_content") or item.get("content") or "")[:MAX_TEXT]
        if text and urlparse(url).scheme in ("http", "https"):
            hits.append({"url": url, "title": item.get("title", ""), "content": text})
    return json.dumps(hits)

@mcp.tool()
async def fetch_page(url: str) -> str:
    """Fetch a public web page and return its main text."""
    html = await safe_get(url)
    return (trafilatura.extract(html) or "")[:MAX_TEXT]

def deny_fetch(url, *args, **kwargs):
    raise ValueError(f"external resource blocked: {url}")

@mcp.tool()
async def render_pdf(markdown: str) -> str:
    """Render Markdown to a PDF and return it base64-encoded."""
    body = nh3.clean(md_lib.markdown(markdown, extensions=["tables", "fenced_code"]), tags=ALLOWED_TAGS)
    document = f"<html><head><meta charset='utf-8'><style>{PDF_CSS}</style></head><body>{body}</body></html>"
    pdf = await asyncio.to_thread(lambda: HTML(string=document, url_fetcher=deny_fetch).write_pdf())
    return base64.b64encode(pdf).decode()

# ---------------------------------------------------------------- service auth
class BearerAuth:
    """ASGI middleware: every HTTP request except /healthz needs the shared service token."""

    def __init__(self, app, token: str):
        self.app, self.expected = app, f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            if scope["path"] == "/healthz":
                await JSONResponse({"ok": True})(scope, receive, send)
                return
            got = dict(scope["headers"]).get(b"authorization", b"")
            if not hmac.compare_digest(got, self.expected):     # constant-time comparison
                await PlainTextResponse("unauthorized", status_code=401)(scope, receive, send)
                return
        await self.app(scope, receive, send)   # lifespan events pass through untouched


if __name__ == "__main__":
    uvicorn.run(BearerAuth(mcp.streamable_http_app(), secret("mcp_token")), host=BIND, port=PORT)