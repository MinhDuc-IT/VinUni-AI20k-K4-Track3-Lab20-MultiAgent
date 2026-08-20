"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import LabError


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """OpenRouter-backed LLM client (OpenAI-compatible API)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.openrouter_api_key:
            raise LabError(
                "OPENROUTER_API_KEY is not set. Add it to your .env file before calling the LLM."
            )
        self._client = OpenAI(
            api_key=self.settings.openrouter_api_key,
            base_url=self.settings.openrouter_base_url,
            timeout=float(self.settings.timeout_seconds),
            default_headers={
                "HTTP-Referer": "https://github.com/vinuni-ai20k/multi-agent-research-lab",
                "X-Title": "Multi-Agent Research Lab",
            },
        )

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with retry and token usage."""

        return self._complete_with_retry(system_prompt, user_prompt)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
    )
    def _complete_with_retry(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        completion = self._client.chat.completions.create(
            model=self.settings.openrouter_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        choice = completion.choices[0].message.content
        if not choice:
            raise LabError("LLM returned an empty completion")

        usage = completion.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None

        return LLMResponse(
            content=choice.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=None,
        )
