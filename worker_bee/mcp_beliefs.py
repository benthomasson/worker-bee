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

    @staticmethod
    def _parse_sse(text: str) -> dict | None:
        """Extract the JSON-RPC response from an SSE stream."""
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("data: ") and line != "data: ":
                try:
                    return json.loads(line[6:])
                except (json.JSONDecodeError, ValueError):
                    continue
        return None

    def _post(self, headers: dict, body: dict) -> tuple[dict, dict]:
        """Post a JSON-RPC request; return (parsed_data, response_headers)."""
        resp = self._client.post(self.mcp_url, headers=headers, json=body)
        resp.raise_for_status()
        resp_headers = dict(resp.headers)
        ct = resp.headers.get("content-type", "")
        if "text/event-stream" in ct:
            data = self._parse_sse(resp.text)
            if data is None:
                raise RuntimeError(f"No JSON-RPC response in SSE stream")
            return data, resp_headers
        return resp.json(), resp_headers

    def _call(self, tool_name: str, arguments: dict) -> dict | list | str:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        if not self._session_id:
            init_data, init_headers = self._post(
                headers,
                {
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
            sid = init_headers.get("mcp-session-id") or init_headers.get("Mcp-Session-Id")
            if sid:
                self._session_id = sid
                headers["Mcp-Session-Id"] = sid
            logger.debug("MCP session initialized: %s", init_data)

            self._client.post(
                self.mcp_url,
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                },
            )

        data, _ = self._post(
            headers,
            {
                "jsonrpc": "2.0",
                "id": _next_id(),
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            },
        )

        if "error" in data:
            raise RuntimeError(f"MCP error: {data['error']}")

        result = data.get("result", {})
        content = result.get("content", [])
        texts = [c.get("text", "") for c in content if c.get("type") == "text"]
        return "\n".join(texts)

    def search(self, query: str, limit: int = 20) -> str:
        args: dict = {"query": query}
        if self.domain_id:
            args["domain"] = self.domain_id
        return self._call("search", args)

    def show(self, node_id: str) -> str:
        args: dict = {"node_id": node_id}
        if self.domain_id:
            args["domain"] = self.domain_id
        return self._call("show", args)

    def explain(self, node_id: str) -> str:
        args: dict = {"node_id": node_id}
        if self.domain_id:
            args["domain"] = self.domain_id
        return self._call("explain", args)

    def list_beliefs(self, truth_value: str = "IN", limit: int = 50) -> str:
        args: dict = {}
        if truth_value:
            args["status"] = truth_value
        if self.domain_id:
            args["domain"] = self.domain_id
        return self._call("list", args)

    def list_domains(self) -> str:
        return self._call("domains", {})

    def propose_belief(self, node_id: str, text: str, source: str = "") -> str:
        args: dict = {"node_id": node_id, "text": text}
        if source:
            args["source"] = source
        if self.domain_id:
            args["domain"] = self.domain_id
        return self._call("add", args)

    def propose_retraction(self, node_id: str, reason: str = "") -> str:
        args: dict = {"node_id": node_id}
        if reason:
            args["reason"] = reason
        if self.domain_id:
            args["domain"] = self.domain_id
        return self._call("retract", args)

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
