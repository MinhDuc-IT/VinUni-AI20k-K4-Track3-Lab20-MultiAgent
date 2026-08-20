from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse
from multi_agent_research_lab.services.search_client import SearchClient
from multi_agent_research_lab.core.schemas import SourceDocument


class _FakeLLM(LLMClient):
    def __init__(self) -> None:  # noqa: D107
        pass

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        return LLMResponse(content=f"notes for: {user_prompt[:40]}", input_tokens=1, output_tokens=1)


class _FakeSearch(SearchClient):
    def __init__(self) -> None:  # noqa: D107
        pass

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        return [
            SourceDocument(
                title="Fake Source",
                url="https://example.com",
                snippet="Multi-agent systems coordinate specialized roles.",
            )
        ][:max_results]


def test_workflow_build_compiles() -> None:
    workflow = MultiAgentWorkflow(llm_client=_FakeLLM(), search_client=_FakeSearch())
    app = workflow.build()
    assert app is not None


def test_workflow_run_with_fakes() -> None:
    workflow = MultiAgentWorkflow(llm_client=_FakeLLM(), search_client=_FakeSearch())
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    result = workflow.run(state)
    assert result.route_history[0] == "researcher"
    assert result.route_history[-1] == "done"
    assert result.final_answer
    assert len(result.sources) == 1


def test_route_after_supervisor_reads_history() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.record_route("analyst")
    route = MultiAgentWorkflow._route_after_supervisor({"payload": state})
    assert route == "analyst"
