"""Private memory service for Byte control-plane integration."""

from __future__ import annotations

import argparse
import os
import time
from threading import Lock
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from byte import INTERNAL_AUTH_HEADER, PRODUCT_NAME, __version__
from byte.processor._reasoning_store import ReasoningMemoryStore
from byte.processor.ai_memory import AIMemoryStore
from byte.processor.intent import IntentGraph, extract_request_intent
from byte.processor.tool_result import ToolResultStore
from byte_server._control_plane import request_text, response_text
from byte_server.models import MemoryRememberRequest, MemoryResolveRequest


class MemoryServiceRuntime:
    def __init__(self, *, max_entries: int = 2000) -> None:
        self.tool_results = ToolResultStore()
        self.reasoning = ReasoningMemoryStore(max_entries=max_entries)
        self.ai_memory = AIMemoryStore(max_entries=max_entries)
        self.intent_graph = IntentGraph(window_size=3)
        self._session_summaries: dict[tuple[str, str], str] = {}
        self._retrieval_contexts: dict[tuple[str, str], str] = {}
        self._lock = Lock()
        self._writes = 0

    def resolve(self, payload: MemoryResolveRequest) -> dict[str, Any]:
        scope = payload.scope
        request_payload = dict(payload.request or {})
        provider_mode = str(payload.provider_mode or "hosted").strip().lower()
        intent = extract_request_intent(request_payload)
        self.intent_graph.record(request_payload, session_id=scope.session)

        context: dict[str, Any] = {}
        hits: dict[str, Any] = {}

        tool_name = str(request_payload.get("byte_tool_name", "") or "").strip()
        tool_args = request_payload.get("byte_tool_args")
        if tool_name:
            tool_entry = self.tool_results.get(tool_name, tool_args, scope=scope.scope_key)
            if tool_entry is not None:
                context["byte_tool_result_context"] = (
                    f"Reusable tool result from {tool_name}: {tool_entry['result']}"
                )
                hits["tool_result"] = True

        query = request_text(request_payload)
        reasoning_hits = self.reasoning.lookup_related(
            query_text=query,
            verified_only=True,
            min_score=0.6,
            top_k=1,
        )
        if reasoning_hits:
            hit = reasoning_hits[0]
            metadata = dict(hit.get("metadata", {}) or {})
            if provider_mode == "local" and metadata.get("full_prefix"):
                context["byte_retrieval_context"] = metadata.get("full_prefix")
                hits["reasoning_mode"] = "full_prefix"
            else:
                summary = str(metadata.get("summary") or hit.get("answer") or "").strip()
                if summary:
                    context["byte_retrieval_context"] = summary
                    hits["reasoning_mode"] = "summary"

        session_summary = self._session_summaries.get((scope.tenant, scope.session), "")
        if session_summary:
            context["byte_repo_summary"] = session_summary
            hits["session_summary"] = True

        retrieval_context = self._retrieval_contexts.get((scope.tenant, intent.route_key), "")
        if retrieval_context:
            existing = str(context.get("byte_retrieval_context", "") or "").strip()
            context["byte_retrieval_context"] = (
                f"{existing}\n{retrieval_context}".strip() if existing else retrieval_context
            )
            hits["retrieval_context"] = True

        return {
            "scope": scope.model_dump(),
            "intent": intent.route_key,
            "provider_mode": provider_mode,
            "hits": hits,
            "context": context,
        }

    def remember(self, payload: MemoryRememberRequest) -> dict[str, Any]:
        scope = payload.scope
        request_payload = dict(payload.request or {})
        provider_mode = str(payload.provider_mode or "hosted").strip().lower()
        intent = extract_request_intent(request_payload)
        answer_text = response_text(payload.response)
        self.intent_graph.record(request_payload, session_id=scope.session)

        remembered = {
            "intent": intent.route_key,
            "provider_mode": provider_mode,
            "stored": [],
        }

        tool_name = str(request_payload.get("byte_tool_name", "") or "").strip()
        tool_args = request_payload.get("byte_tool_args")
        tool_result = request_payload.get("byte_tool_result")
        if tool_name and tool_result not in (None, ""):
            self.tool_results.put(
                tool_name,
                tool_args,
                tool_result,
                ttl=float(request_payload.get("byte_tool_ttl", 300) or 300),
                scope=scope.scope_key,
                metadata={"tenant": scope.tenant, "session": scope.session},
            )
            remembered["stored"].append("tool_result")

        if answer_text:
            metadata = {
                "tenant": scope.tenant,
                "session": scope.session,
                "workflow": scope.workflow,
                "summary": answer_text[:1200],
            }
            if provider_mode == "local":
                metadata["full_prefix"] = answer_text[:4000]
            self.reasoning.remember(
                kind="local_full" if provider_mode == "local" else "hosted_summary",
                key=f"{scope.tenant}:{intent.route_key}",
                answer=answer_text,
                verified=True,
                metadata=metadata,
                source="memory_service",
            )
            self.ai_memory.remember(
                request_payload,
                answer=answer_text,
                model=str(request_payload.get("model", "") or ""),
                provider=provider_mode,
                scope=scope.scope_key,
                metadata={"worker_id": payload.worker_id, "intent": intent.route_key},
                source="memory_service",
            )
            with self._lock:
                self._session_summaries[(scope.tenant, scope.session)] = answer_text[:1200]
                self._retrieval_contexts[(scope.tenant, intent.route_key)] = answer_text[:1600]
                self._writes += 1
            remembered["stored"].extend(["reasoning", "session_summary", "retrieval_context"])

        return remembered

    def stats(self) -> dict[str, Any]:
        return {
            "writes": self._writes,
            "tool_results": self.tool_results.stats(),
            "reasoning": self.reasoning.stats(),
            "ai_memory": self.ai_memory.stats(),
            "intent_graph": self.intent_graph.stats(),
            "session_summaries": len(self._session_summaries),
            "retrieval_contexts": len(self._retrieval_contexts),
            "timestamp": time.time(),
        }


def _normalize_internal_token(value: str | None) -> str:
    return str(value or "").strip()


def _require_internal_auth(request: Request) -> None:
    if not _internal_auth_token:
        return
    if request.headers.get(INTERNAL_AUTH_HEADER) == _internal_auth_token:
        return
    raise HTTPException(status_code=403, detail="Byte internal authentication failed.")


runtime = MemoryServiceRuntime()
_internal_auth_token = _normalize_internal_token(os.getenv("BYTE_INTERNAL_TOKEN", ""))
app = FastAPI(title=f"{PRODUCT_NAME} Memory Service", version=__version__)


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"status": "ok", "service": "byte_memory"}


@app.get("/internal/v1/runtime")
async def runtime_snapshot(request: Request) -> dict[str, Any]:
    _require_internal_auth(request)
    return runtime.stats()


@app.get("/internal/v1/intent-graph")
async def intent_graph(request: Request, limit: int = 25) -> dict[str, Any]:
    _require_internal_auth(request)
    return runtime.intent_graph.stats(top_n=limit)


@app.post("/internal/v1/resolve")
async def resolve_memory(request: Request, payload: MemoryResolveRequest) -> JSONResponse:
    _require_internal_auth(request)
    return JSONResponse(content=runtime.resolve(payload))


@app.post("/internal/v1/remember")
async def remember_memory(request: Request, payload: MemoryRememberRequest) -> JSONResponse:
    _require_internal_auth(request)
    return JSONResponse(content=runtime.remember(payload))


def create_app() -> FastAPI:
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--max-entries", type=int, default=2000)
    parser.add_argument(
        "--internal-auth-token",
        default=os.getenv("BYTE_INTERNAL_TOKEN", ""),
    )
    args = parser.parse_args()

    global runtime, _internal_auth_token
    runtime = MemoryServiceRuntime(max_entries=max(1, int(args.max_entries or 1)))
    _internal_auth_token = _normalize_internal_token(args.internal_auth_token)
    uvicorn.run(app, host=args.host, port=args.port)


__all__ = [
    "MemoryServiceRuntime",
    "_internal_auth_token",
    "app",
    "create_app",
    "main",
    "runtime",
]
