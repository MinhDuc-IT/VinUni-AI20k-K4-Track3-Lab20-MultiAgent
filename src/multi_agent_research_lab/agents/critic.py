"""Optional critic agent for lightweight answer validation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class CriticAgent(BaseAgent):
    """Optional fact-checking and citation-coverage checks."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings."""

        findings: list[str] = []
        answer = state.final_answer or ""

        if not answer.strip():
            findings.append("Final answer is empty.")
        if state.sources and "[" not in answer:
            findings.append("Answer has sources available but no inline citations like [1].")
        if state.sources and "Sources" not in answer and "sources" not in answer.lower():
            findings.append("Answer is missing an explicit Sources section.")
        if len(answer.split()) < 80:
            findings.append("Answer looks short for a research summary; may lack depth.")

        content = (
            "Critic findings:\n- " + "\n- ".join(findings)
            if findings
            else "Critic findings: no major issues detected."
        )
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=content,
                metadata={"issue_count": len(findings)},
            )
        )
        state.add_trace_event("critic.done", {"issue_count": len(findings), "findings": findings})
        if findings:
            state.errors.extend(f"critic: {item}" for item in findings)
        return state
