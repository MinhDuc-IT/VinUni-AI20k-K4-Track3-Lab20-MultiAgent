from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    build_metrics,
    citation_coverage,
    heuristic_quality_score,
)
from multi_agent_research_lab.evaluation.report import render_markdown_report


def test_report_renders_markdown() -> None:
    report = render_markdown_report(
        [BenchmarkMetrics(run_name="baseline", latency_seconds=1.23)],
        query="demo query",
        trace_paths=["reports/traces/demo.json"],
    )
    assert "Benchmark Report" in report
    assert "baseline" in report
    assert "demo query" in report
    assert "Failure modes" in report


def test_citation_coverage_and_quality() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.sources = [
        SourceDocument(title="A", url="https://a.example", snippet="a"),
        SourceDocument(title="B", url="https://b.example", snippet="b"),
    ]
    state.final_answer = (
        "Multi-agent helps on complex tasks [1]. Single-agent is enough for simple ones [2]. "
        "Sources\n[1] A\n[2] B\n" + (" word" * 80)
    )
    assert citation_coverage(state) == 1.0
    assert heuristic_quality_score(state) >= 7.0
    metrics = build_metrics("multi-agent", state, latency=2.5)
    assert metrics.failure_rate == 0.0
    assert metrics.citation_coverage == 1.0
