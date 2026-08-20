"""Researcher agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        search_client: SearchClient | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.search_client = search_client or SearchClient()
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""

        tokens: dict[str, int | None] = {}
        with trace_span("researcher.run", {"query": state.request.query}) as span:
            try:
                docs = self.search_client.search(
                    state.request.query,
                    max_results=state.request.max_sources,
                )
            except LabError as exc:
                state.errors.append(f"researcher.search: {exc}")
                state.add_trace_event("researcher.error", {"error": str(exc)})
                span["attributes"]["error"] = str(exc)
                return state

            state.sources = docs
            if not docs:
                state.errors.append("researcher: no sources found")
                state.research_notes = "No sources found for the query."
                state.add_trace_event("researcher.done", {"num_sources": 0})
                return state

            source_block = "\n".join(
                f"[{idx}] {doc.title}\nURL: {doc.url or 'n/a'}\n{doc.snippet}"
                for idx, doc in enumerate(docs, start=1)
            )
            try:
                response = self.llm_client.complete(
                    system_prompt=(
                        "You are a research assistant. Summarize the provided sources into "
                        "concise research notes. Preserve important claims and cite sources "
                        "as [1], [2], etc. Do not invent facts beyond the sources."
                    ),
                    user_prompt=(
                        f"Query: {state.request.query}\n\nSources:\n{source_block}\n\n"
                        "Write research notes with citations."
                    ),
                )
                state.research_notes = response.content
                tokens = {
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                }
            except LabError as exc:
                state.errors.append(f"researcher.llm: {exc}")
                state.research_notes = source_block

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=state.research_notes or "",
                    metadata={"num_sources": len(docs), **tokens},
                )
            )
            span["attributes"]["num_sources"] = len(docs)

        state.add_trace_event(
            "researcher.done",
            {
                "num_sources": len(state.sources),
                **tokens,
                "duration_seconds": span["duration_seconds"],
            },
        )
        return state
