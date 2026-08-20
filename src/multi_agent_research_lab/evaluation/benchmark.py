"""Benchmark helpers for single-agent vs multi-agent."""

from __future__ import annotations

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]

_CITATION_RE = re.compile(r"\[(\d+)\]")

# Rough OpenRouter gpt-4o-mini style rates (USD / 1M tokens). Good enough for lab comparison.
_INPUT_COST_PER_M = 0.15
_OUTPUT_COST_PER_M = 0.60


def _sum_tokens(state: ResearchState) -> tuple[int, int]:
    input_tokens = 0
    output_tokens = 0
    for result in state.agent_results:
        meta = result.metadata or {}
        if isinstance(meta.get("input_tokens"), int):
            input_tokens += meta["input_tokens"]
        if isinstance(meta.get("output_tokens"), int):
            output_tokens += meta["output_tokens"]
    if input_tokens or output_tokens:
        return input_tokens, output_tokens
    # Baseline path records tokens on trace events only.
    for event in state.trace:
        payload = event.get("payload") or {}
        if isinstance(payload.get("input_tokens"), int):
            input_tokens += payload["input_tokens"]
        if isinstance(payload.get("output_tokens"), int):
            output_tokens += payload["output_tokens"]
    return input_tokens, output_tokens


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000) * _INPUT_COST_PER_M + (
        output_tokens / 1_000_000
    ) * _OUTPUT_COST_PER_M


def citation_coverage(state: ResearchState) -> float:
    """Fraction of available sources referenced by inline [n] citations."""

    answer = state.final_answer or ""
    if not state.sources:
        return 0.0
    cited = {int(match) for match in _CITATION_RE.findall(answer)}
    valid = {idx for idx in cited if 1 <= idx <= len(state.sources)}
    return len(valid) / len(state.sources)


def heuristic_quality_score(state: ResearchState) -> float:
    """Lightweight 0-10 quality proxy for lab demos (not a substitute for peer review)."""

    answer = state.final_answer or ""
    if not answer.strip():
        return 0.0

    score = 3.0
    words = len(answer.split())
    if words >= 120:
        score += 2.0
    elif words >= 60:
        score += 1.0

    coverage = citation_coverage(state)
    score += 3.0 * coverage

    if "Sources" in answer or "sources" in answer.lower():
        score += 1.0
    if state.sources:
        score += 1.0
    if state.errors:
        score -= min(2.0, 0.5 * len(state.errors))

    return max(0.0, min(10.0, round(score, 1)))


def failure_rate(state: ResearchState) -> float:
    answer = state.final_answer or ""
    if not answer.strip():
        return 1.0
    if "Unable to produce" in answer:
        return 1.0
    return 0.0


def build_metrics(run_name: str, state: ResearchState, latency: float) -> BenchmarkMetrics:
    input_tokens, output_tokens = _sum_tokens(state)
    notes_parts = [
        f"tokens_in={input_tokens}",
        f"tokens_out={output_tokens}",
        f"sources={len(state.sources)}",
        f"routes={','.join(state.route_history) or 'n/a'}",
    ]
    if state.errors:
        notes_parts.append(f"errors={len(state.errors)}")

    return BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=estimate_cost_usd(input_tokens, output_tokens),
        quality_score=heuristic_quality_score(state),
        citation_coverage=citation_coverage(state),
        failure_rate=failure_rate(state),
        notes="; ".join(notes_parts),
    )


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run one configuration and collect latency/cost/quality proxies."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    metrics = build_metrics(run_name, state, latency)
    return state, metrics
