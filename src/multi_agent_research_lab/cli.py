"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import LabError, StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import configure_tracing, export_trace
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore
from multi_agent_research_lab.utils.timer import elapsed_timer

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    tracing = configure_tracing(settings)
    if tracing["langsmith_enabled"]:
        console.print(f"[dim]LangSmith tracing enabled -> project={tracing['project']}[/dim]")


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline via OpenRouter."""

    _init()
    request = _parse_query(query)
    state = ResearchState(request=request)

    try:
        llm = LLMClient()
        with elapsed_timer() as elapsed:
            response = llm.complete(
                system_prompt=(
                    "You are a careful research assistant. Produce a clear, structured "
                    "answer for technical learners. Prefer concrete claims over fluff."
                ),
                user_prompt=request.query,
            )
        latency = elapsed()
    except LabError as exc:
        console.print(Panel.fit(str(exc), title="LLM Error", style="red"))
        raise typer.Exit(code=1) from exc

    state.final_answer = response.content
    state.add_trace_event(
        "baseline",
        {
            "latency_seconds": round(latency, 3),
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "model": get_settings().openrouter_model,
        },
    )
    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline"))
    console.print(
        Panel.fit(
            (
                f"latency={latency:.2f}s | "
                f"in={response.input_tokens} | "
                f"out={response.output_tokens}"
            ),
            title="Baseline Metrics",
        )
    )
    console.print(f"[dim]Trace exported -> {export_trace(state, run_name='single-agent')}[/dim]")


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent LangGraph workflow."""

    _init()
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow()
    try:
        with elapsed_timer() as elapsed:
            result = workflow.run(state)
        latency = elapsed()
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    except LabError as exc:
        console.print(Panel.fit(str(exc), title="Workflow Error", style="red"))
        raise typer.Exit(code=1) from exc

    console.print(Panel.fit(result.final_answer or "(empty)", title="Multi-Agent Final Answer"))
    console.print(
        Panel.fit(
            (
                f"routes={result.route_history} | "
                f"sources={len(result.sources)} | "
                f"errors={len(result.errors)} | "
                f"latency={latency:.2f}s"
            ),
            title="Multi-Agent Metrics",
        )
    )
    trace_path = export_trace(result, run_name="multi-agent")
    console.print(f"[dim]Trace exported -> {trace_path}[/dim]")


@app.command()
def benchmark(
    query: Annotated[
        str,
        typer.Option(
            "--query",
            "-q",
            help="Shared research query for both runs",
        ),
    ] = "Compare single-agent and multi-agent workflows for complex research tasks",
) -> None:
    """Benchmark single-agent vs multi-agent and write reports/benchmark_report.md."""

    _init()
    request = _parse_query(query)

    def run_single(q: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=q))
        llm = LLMClient()
        response = llm.complete(
            system_prompt=(
                "You are a careful research assistant. Produce a clear, structured "
                "answer for technical learners. Prefer concrete claims over fluff."
            ),
            user_prompt=q,
        )
        state.final_answer = response.content
        state.add_trace_event(
            "baseline",
            {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "model": get_settings().openrouter_model,
            },
        )
        return state

    def run_multi(q: str) -> ResearchState:
        return MultiAgentWorkflow().run(ResearchState(request=ResearchQuery(query=q)))

    try:
        single_state, single_metrics = run_benchmark("single-agent", request.query, run_single)
        multi_state, multi_metrics = run_benchmark("multi-agent", request.query, run_multi)
    except LabError as exc:
        console.print(Panel.fit(str(exc), title="Benchmark Error", style="red"))
        raise typer.Exit(code=1) from exc

    single_trace = export_trace(single_state, run_name="single-agent")
    multi_trace = export_trace(multi_state, run_name="multi-agent")
    report = render_markdown_report(
        [single_metrics, multi_metrics],
        query=request.query,
        trace_paths=[str(single_trace), str(multi_trace)],
    )
    report_path = LocalArtifactStore().write_text("benchmark_report.md", report)

    console.print(
        Panel.fit(
            (
                f"single: {single_metrics.latency_seconds:.2f}s | "
                f"q={single_metrics.quality_score} | cite={single_metrics.citation_coverage}\n"
                f"multi:  {multi_metrics.latency_seconds:.2f}s | "
                f"q={multi_metrics.quality_score} | cite={multi_metrics.citation_coverage}"
            ),
            title="Benchmark Summary",
        )
    )
    console.print(f"[green]Wrote[/green] {report_path}")
    console.print(f"[dim]Traces -> {single_trace} ; {multi_trace}[/dim]")


if __name__ == "__main__":
    app()
