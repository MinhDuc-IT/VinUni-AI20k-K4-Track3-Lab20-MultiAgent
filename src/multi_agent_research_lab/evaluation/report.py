"""Benchmark report rendering."""

from __future__ import annotations

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(
    metrics: list[BenchmarkMetrics],
    *,
    query: str | None = None,
    trace_paths: list[str] | None = None,
    failure_mode_notes: str | None = None,
) -> str:
    """Render benchmark metrics to markdown suitable for lab submission."""

    lines = [
        "# Benchmark Report",
        "",
        "Comparison of **single-agent baseline** vs **multi-agent workflow** "
        "(Supervisor -> Researcher -> Analyst -> Writer).",
        "",
    ]
    if query:
        lines.extend(["## Query", "", f"> {query}", ""])

    lines.extend(
        [
            "## Metrics",
            "",
            "| Run | Latency (s) | Cost (USD) | Quality* | Citation cov. | Failure rate | Notes |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.6f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )

    lines.extend(
        [
            "",
            "*Quality is a lab heuristic (length + citations + structure), not peer-review score.",
            "",
            "## Analysis",
            "",
        ]
    )

    by_name = {item.run_name: item for item in metrics}
    baseline = by_name.get("single-agent") or by_name.get("baseline")
    multi = by_name.get("multi-agent")
    if baseline and multi:
        latency_delta = multi.latency_seconds - baseline.latency_seconds
        cost_delta = (multi.estimated_cost_usd or 0) - (baseline.estimated_cost_usd or 0)
        quality_delta = (multi.quality_score or 0) - (baseline.quality_score or 0)
        citation_delta = (multi.citation_coverage or 0) - (baseline.citation_coverage or 0)
        lines.extend(
            [
                f"- Latency delta (multi - single): **{latency_delta:+.2f}s**",
                f"- Estimated cost delta: **{cost_delta:+.6f} USD**",
                f"- Quality heuristic delta: **{quality_delta:+.1f}**",
                f"- Citation coverage delta: **{citation_delta:+.0%}**",
                "",
                "### When multi-agent helped",
                "",
                "- Clear role separation (search -> analysis -> writing) improved citation discipline "
                "and structured evidence handling.",
                "- Shared state + route history made failures easier to debug.",
                "",
                "### When single-agent was preferable",
                "",
                "- Lower latency and fewer tokens for short factual questions.",
                "- Less coordination overhead when the task does not need independent verification.",
                "",
            ]
        )
    else:
        lines.append("Insufficient runs to compare single vs multi-agent.")
        lines.append("")

    lines.extend(
        [
            "## Failure modes observed / expected",
            "",
            failure_mode_notes
            or (
                "1. **Empty / weak retrieval** - local corpus may miss niche queries; "
                "Researcher falls back to raw snippets or errors.\n"
                "2. **Coordination overhead** - multi-agent makes 3 LLM calls, so latency/cost rise.\n"
                "3. **Citation drift** - Writer may cite indices not grounded in Analyst notes; "
                "Critic/heuristic coverage catches some of this.\n"
                "4. **Max-iteration stop** - Supervisor ends the graph if workers fail to fill "
                "required fields before `MAX_ITERATIONS`.\n"
                "\n"
                "**Fix approach:** tighten Researcher filtering, add Critic before done, "
                "cache search results, and keep a strong single-agent baseline for simple queries."
            ),
            "",
            "## Traces",
            "",
        ]
    )
    if trace_paths:
        for path in trace_paths:
            lines.append(f"- `{path}`")
    else:
        lines.append(
            "- Local JSON traces are written under `reports/traces/`. "
            "Attach a screenshot or LangSmith link if enabled."
        )
    lines.extend(
        [
            "",
            "## Exit ticket notes",
            "",
            "- **Use multi-agent** when the task needs retrieval + analysis + cited synthesis, "
            "or independent verification.",
            "- **Avoid multi-agent** for short, single-hop answers where latency/cost dominate.",
            "",
        ]
    )
    return "\n".join(lines)
