"""MCP-backed belief store for reasons-service-edge.

Provides the same interface as the local BeliefStore but routes calls
through MCP to the edge service. The model never sees MCP — it uses
the same search_beliefs, show_belief, etc. tools it already knows.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

_REQ_COUNTER = 0


def _next_id() -> str:
    global _REQ_COUNTER
    _REQ_COUNTER += 1
    return str(_REQ_COUNTER)


class McpBeliefStore:
    """Belief store backed by a reasons-service-edge MCP server."""

    def __init__(self, mcp_url: str, token: str | None = None, domain_id: str | None = None):
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for MCP belief store. "
                "Install with: pip install worker-bee[mcp]"
            )
        self.mcp_url = mcp_url.rstrip("/")
        self.token = token
        self.domain_id = domain_id
        self._session_id: str | None = None
        self._client = httpx.Client(timeout=30)

    def _call(self, tool_name: str, arguments: dict) -> dict | list | str:
        headers = {
            "Content-Type": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        # Initialize session if needed
        if not self._session_id:
            init_resp = self._client.post(
                self.mcp_url,
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": _next_id(),
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "worker-bee", "version": "0.1.0"},
                    },
                },
            )
            if init_resp.status_code == 200:
                init_data = init_resp.json()
                sid = init_resp.headers.get("Mcp-Session-Id")
                if sid:
                    self._session_id = sid
                    headers["Mcp-Session-Id"] = sid
                logger.debug("MCP session initialized: %s", init_data)

                # Send initialized notification
                self._client.post(
                    self.mcp_url,
                    headers=headers,
                    json={
                        "jsonrpc": "2.0",
                        "method": "notifications/initialized",
                    },
                )

        resp = self._client.post(
            self.mcp_url,
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": _next_id(),
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            },
        )
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            raise RuntimeError(f"MCP error: {data['error']}")

        result = data.get("result", {})
        content = result.get("content", [])
        texts = [c.get("text", "") for c in content if c.get("type") == "text"]
        return "\n".join(texts)

    def search(self, query: str, limit: int = 20) -> str:
        args = {"query": query}
        if self.domain_id:
            args["domain_id"] = self.domain_id
        return self._call("search", args)

    def show(self, belief_id: str) -> str:
        args = {"belief_id": belief_id}
        if self.domain_id:
            args["domain_id"] = self.domain_id
        return self._call("get_belief", args)

    def explain(self, belief_id: str) -> str:
        args = {"belief_id": belief_id}
        if self.domain_id:
            args["domain_id"] = self.domain_id
        return self._call("explain_belief", args)

    def list_beliefs(self, truth_value: str = "IN", limit: int = 50) -> str:
        args = {"truth_value": truth_value, "limit": limit}
        if self.domain_id:
            args["domain_id"] = self.domain_id
        return self._call("list_beliefs", args)

    def list_domains(self) -> str:
        return self._call("list_domains", {})

    def propose_belief(self, belief_id: str, text: str, source: str = "") -> str:
        args = {"belief_id": belief_id, "text": text}
        if source:
            args["source"] = source
        if self.domain_id:
            args["domain_id"] = self.domain_id
        return self._call("propose_belief", args)

    def propose_retraction(self, belief_id: str, reason: str = "") -> str:
        args = {"belief_id": belief_id}
        if reason:
            args["reason"] = reason
        if self.domain_id:
            args["domain_id"] = self.domain_id
        return self._call("propose_retraction", args)

    @property
    def has_any(self) -> bool:
        return True

    def close(self) -> None:
        self._client.close()


_mcp_store: McpBeliefStore | None = None


def set_mcp_beliefs(
    mcp_url: str, token: str, domain_id: str | None = None
) -> McpBeliefStore:
    global _mcp_store
    if _mcp_store:
        _mcp_store.close()
    _mcp_store = McpBeliefStore(mcp_url, token, domain_id)
    logger.info("MCP belief store configured: %s (domain=%s)", mcp_url, domain_id)
    return _mcp_store


def get_mcp_store() -> McpBeliefStore | None:
    return _mcp_store


def clear_mcp_beliefs() -> None:
    global _mcp_store
    if _mcp_store:
        _mcp_store.close()
    _mcp_store = None
