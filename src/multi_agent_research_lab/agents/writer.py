"""Writer agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""

        context = state.analysis_notes or state.research_notes
        if not context:
            state.errors.append("writer: missing analysis/research notes")
            state.final_answer = (
                "Unable to produce a researched answer because upstream notes are missing."
            )
            state.add_trace_event("writer.skipped", {"reason": "no context"})
            return state

        citations = "\n".join(
            f"[{idx}] {doc.title} ({doc.url or 'local/offline'})"
            for idx, doc in enumerate(state.sources, start=1)
        )

        tokens: dict[str, object] = {}
        with trace_span("writer.run") as span:
            user_prompt = (
                f"Audience: {state.request.audience}\n"
                f"Query: {state.request.query}\n\n"
                f"Analysis / research context:\n{context}\n\n"
                f"Available citations:\n{citations or '(none)'}\n\n"
                "Write a clear final answer. Use inline citations like [1], [2] where claims "
                "rely on sources. End with a Sources section listing the citations used."
            )
            try:
                response = self.llm_client.complete(
                    system_prompt=(
                        "You are a technical writer. Synthesize a grounded answer for the "
                        "stated audience. Prefer evidence over speculation. Include citations."
                    ),
                    user_prompt=user_prompt,
                )
                state.final_answer = response.content
                tokens = {
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                }
            except LabError as exc:
                state.errors.append(f"writer.llm: {exc}")
                state.final_answer = (
                    f"{context}\n\nSources:\n{citations}" if citations else context
                )
                tokens = {"fallback": True}
                span["attributes"]["error"] = str(exc)

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
                    content=state.final_answer or "",
                    metadata=tokens,
                )
            )

        state.add_trace_event(
            "writer.done",
            {**tokens, "duration_seconds": span["duration_seconds"]},
        )
        return state
