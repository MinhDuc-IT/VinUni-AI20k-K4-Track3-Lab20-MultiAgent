"""Tracing hooks for local JSON traces and optional LangSmith."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal local span context used across agents."""

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started


def configure_tracing(settings: Settings | None = None) -> dict[str, Any]:
    """Enable LangSmith tracing when credentials are present.

    Without LANGSMITH_API_KEY, the lab still records local JSON traces on ResearchState.
    """

    settings = settings or get_settings()
    status: dict[str, Any] = {
        "provider": "local",
        "langsmith_enabled": False,
        "project": settings.langsmith_project,
    }
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
        status["provider"] = "langsmith+local"
        status["langsmith_enabled"] = True
    return status


def export_trace(
    state: ResearchState,
    *,
    run_name: str,
    output_dir: Path = Path("reports/traces"),
) -> Path:
    """Write a local JSON trace artifact for screenshots / submission."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"{run_name}_{stamp}.json"
    payload = {
        "run_name": run_name,
        "exported_at": stamp,
        "query": state.request.query,
        "route_history": state.route_history,
        "iteration": state.iteration,
        "num_sources": len(state.sources),
        "errors": state.errors,
        "trace": state.trace,
        "agent_results": [item.model_dump() for item in state.agent_results],
        "final_answer_preview": (state.final_answer or "")[:500],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
