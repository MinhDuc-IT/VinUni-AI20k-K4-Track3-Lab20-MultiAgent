"""Supervisor / router."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def decide(self, state: ResearchState) -> str:
        """Return one of: researcher | analyst | writer | done."""

        if state.iteration >= self.settings.max_iterations:
            return "done"
        if state.final_answer:
            return "done"

        # Fallback: if research failed repeatedly, still try to write from what we have.
        if state.errors and not state.sources and "researcher" in state.route_history:
            if not state.final_answer:
                return "writer"
            return "done"

        if not state.sources or not state.research_notes:
            return "researcher"
        if not state.analysis_notes:
            return "analyst"
        if not state.final_answer:
            return "writer"
        return "done"

    def run(self, state: ResearchState) -> ResearchState:
        """Record the next route on shared state."""

        route = self.decide(state)
        state.record_route(route)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.SUPERVISOR,
                content=route,
                metadata={"iteration": state.iteration, "route": route},
            )
        )
        state.add_trace_event(
            "supervisor.route",
            {"route": route, "iteration": state.iteration, "errors": list(state.errors)},
        )
        return state
