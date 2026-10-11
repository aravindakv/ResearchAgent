"""A minimal MCP client for the graph, built on the MCP Python SDK 2.x.

Replaces langchain-mcp-adapters, which (as of October 2026) only supports SDK 1.x. The graph needs
very little from MCP: list the server's tools once, then call them by name with a dict of arguments
and get text back. That's what this module provides:

    tools = await load_mcp_tools(MCP_URL, MCP_TOKEN)
    text = await tools["web_search"].ainvoke({"query": "raft leader election"})

Each call opens a short-lived connection. The server is stateless (chapter 05), so there is no
session to keep alive, and a restarted MCP server is picked up on the next call.
"""
from dataclasses import dataclass

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

TIMEOUT_S = 120  # render_pdf and search on slow pages can take a while


class MCPToolError(RuntimeError):
    """The server ran the tool and reported an error (isError: true), e.g. 'non-public address blocked'."""


def _client(url: str, token: str) -> Client:
    http = httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"}, timeout=TIMEOUT_S)
    return Client(streamable_http_client(url, http_client=http))


@dataclass
class MCPTool:
    name: str
    description: str
    url: str
    token: str

    async def ainvoke(self, arguments: dict) -> str:
        """Call the tool and return its text content. Raises MCPToolError if the tool reports an error."""
        async with _client(self.url, self.token) as client:
            result = await client.call_tool(self.name, arguments)
        text = "".join(getattr(block, "text", "") for block in result.content)
        if getattr(result, "is_error", None) or getattr(result, "isError", None):
            raise MCPToolError(f"{self.name}: {text}")
        return text


async def load_mcp_tools(url: str, token: str) -> dict[str, MCPTool]:
    """Connect once, list the server's tools, and return them by name."""
    async with _client(url, token) as client:
        listed = await client.list_tools()
    return {t.name: MCPTool(t.name, t.description or "", url, token) for t in listed.tools}
