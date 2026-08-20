"""Analyst agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""

        if not state.sources and not state.research_notes:
            state.errors.append("analyst: missing research notes/sources")
            state.add_trace_event("analyst.skipped", {"reason": "no research input"})
            return state

        tokens: dict[str, object] = {}
        with trace_span("analyst.run") as span:
            source_titles = "\n".join(
                f"[{idx}] {doc.title}" for idx, doc in enumerate(state.sources, start=1)
            )
            user_prompt = (
                f"Audience: {state.request.audience}\n"
                f"Query: {state.request.query}\n\n"
                f"Research notes:\n{state.research_notes or '(none)'}\n\n"
                f"Sources:\n{source_titles or '(none)'}\n\n"
                "Produce structured analysis with:\n"
                "1) Key claims\n"
                "2) Supporting evidence with citations [n]\n"
                "3) Conflicting / weak evidence\n"
                "4) Open questions\n"
            )
            try:
                response = self.llm_client.complete(
                    system_prompt=(
                        "You are an analyst. Extract claims, compare viewpoints, and flag "
                        "weak or conflicting evidence. Be concise and explicit about uncertainty."
                    ),
                    user_prompt=user_prompt,
                )
                state.analysis_notes = response.content
                tokens = {
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                }
            except LabError as exc:
                state.errors.append(f"analyst.llm: {exc}")
                state.analysis_notes = state.research_notes
                tokens = {"fallback": True}
                span["attributes"]["error"] = str(exc)

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.ANALYST,
                    content=state.analysis_notes or "",
                    metadata=tokens,
                )
            )

        state.add_trace_event(
            "analyst.done",
            {**tokens, "duration_seconds": span["duration_seconds"]},
        )
        return state
