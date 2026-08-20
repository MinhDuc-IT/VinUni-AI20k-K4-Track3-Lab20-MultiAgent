"""Supervisor routing unit tests."""

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def _state(query: str = "Explain multi-agent systems") -> ResearchState:
    return ResearchState(request=ResearchQuery(query=query))


def test_supervisor_routes_to_researcher_first() -> None:
    supervisor = SupervisorAgent(settings=Settings(max_iterations=6))
    state = supervisor.run(_state())
    assert state.route_history[-1] == "researcher"


def test_supervisor_routes_researcher_analyst_writer_done() -> None:
    supervisor = SupervisorAgent(settings=Settings(max_iterations=6))
    state = _state()

    state = supervisor.run(state)
    assert state.route_history[-1] == "researcher"

    state.sources = [
        SourceDocument(title="Doc", url="https://example.com", snippet="Evidence about agents.")
    ]
    state.research_notes = "Notes with [1]"
    state = supervisor.run(state)
    assert state.route_history[-1] == "analyst"

    state.analysis_notes = "Claims and tradeoffs"
    state = supervisor.run(state)
    assert state.route_history[-1] == "writer"

    state.final_answer = "Final answer with [1]"
    state = supervisor.run(state)
    assert state.route_history[-1] == "done"


def test_supervisor_stops_at_max_iterations() -> None:
    supervisor = SupervisorAgent(settings=Settings(max_iterations=2))
    state = _state()
    state.iteration = 2
    assert supervisor.decide(state) == "done"
