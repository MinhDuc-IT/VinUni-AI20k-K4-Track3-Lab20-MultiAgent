"""LangGraph multi-agent workflow."""

from __future__ import annotations

from typing import TypedDict, cast

from langgraph.graph import END, START, StateGraph

from multi_agent_research_lab.agents import (
    AnalystAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class WorkflowState(TypedDict):
    """LangGraph wrapper around the lab's shared ResearchState."""

    payload: ResearchState


def _as_research_state(payload: ResearchState | dict[str, object]) -> ResearchState:
    if isinstance(payload, ResearchState):
        return payload
    return ResearchState.model_validate(payload)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        search_client: SearchClient | None = None,
    ) -> None:
        self.llm_client = llm_client or LLMClient()
        self.search_client = search_client or SearchClient()
        self.supervisor = SupervisorAgent()
        self.researcher = ResearcherAgent(self.search_client, self.llm_client)
        self.analyst = AnalystAgent(self.llm_client)
        self.writer = WriterAgent(self.llm_client)
        self._app: object | None = None

    def build(self) -> object:
        """Create and compile the LangGraph graph."""

        graph: StateGraph[WorkflowState] = StateGraph(WorkflowState)

        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("researcher", self._researcher_node)
        graph.add_node("analyst", self._analyst_node)
        graph.add_node("writer", self._writer_node)

        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._route_after_supervisor,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "done": END,
            },
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")

        self._app = graph.compile()
        return self._app

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final ResearchState."""

        app = self._app or self.build()
        result = cast(WorkflowState, app.invoke({"payload": state}))  # type: ignore[union-attr]
        final_state = _as_research_state(result["payload"])
        final_state.add_trace_event(
            "workflow.done",
            {
                "route_history": list(final_state.route_history),
                "num_sources": len(final_state.sources),
                "error_count": len(final_state.errors),
            },
        )
        return final_state

    def _supervisor_node(self, state: WorkflowState) -> WorkflowState:
        payload = self.supervisor.run(_as_research_state(state["payload"]))
        return {"payload": payload}

    def _researcher_node(self, state: WorkflowState) -> WorkflowState:
        payload = self.researcher.run(_as_research_state(state["payload"]))
        return {"payload": payload}

    def _analyst_node(self, state: WorkflowState) -> WorkflowState:
        payload = self.analyst.run(_as_research_state(state["payload"]))
        return {"payload": payload}

    def _writer_node(self, state: WorkflowState) -> WorkflowState:
        payload = self.writer.run(_as_research_state(state["payload"]))
        return {"payload": payload}

    @staticmethod
    def _route_after_supervisor(state: WorkflowState) -> str:
        payload = _as_research_state(state["payload"])
        if not payload.route_history:
            return "done"
        return payload.route_history[-1]
