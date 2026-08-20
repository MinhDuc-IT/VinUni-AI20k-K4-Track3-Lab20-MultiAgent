"""Search client abstraction for ResearcherAgent."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.core.schemas import SourceDocument

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _repo_root() -> Path:
    # services/ -> package/ -> src/ -> repo root
    return Path(__file__).resolve().parents[3]


def _default_corpus_dir() -> Path:
    return _repo_root() / "ai_agent_offline_research_corpus_v2" / "topics"


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text) if len(token) > 2}


class SearchClient:
    """Search via Tavily when configured, otherwise local offline corpus."""

    def __init__(
        self,
        settings: Settings | None = None,
        corpus_dir: Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.corpus_dir = corpus_dir or _default_corpus_dir()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""

        if max_results < 1:
            return []
        if self.settings.tavily_api_key:
            return self._search_tavily(query, max_results)
        return self._search_local_corpus(query, max_results)

    def _search_tavily(self, query: str, max_results: int) -> list[SourceDocument]:
        payload = json.dumps(
            {
                "api_key": self.settings.tavily_api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": False,
            }
        ).encode("utf-8")
        request = Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LabError(f"Tavily search failed: {exc}") from exc

        documents: list[SourceDocument] = []
        for item in body.get("results", [])[:max_results]:
            documents.append(
                SourceDocument(
                    title=str(item.get("title") or "Untitled"),
                    url=item.get("url"),
                    snippet=str(item.get("content") or item.get("snippet") or "")[:500],
                    metadata={"provider": "tavily", "score": item.get("score")},
                )
            )
        return documents

    def _search_local_corpus(self, query: str, max_results: int) -> list[SourceDocument]:
        if not self.corpus_dir.exists():
            raise LabError(
                f"Offline corpus not found at {self.corpus_dir}. "
                "Set TAVILY_API_KEY or keep ai_agent_offline_research_corpus_v2 in the repo."
            )

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored: list[tuple[float, SourceDocument]] = []
        for path in sorted(self.corpus_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue

            topic = data.get("topic") or {}
            knowledge = data.get("knowledge_base") or {}
            topic_name = str(topic.get("name") or path.stem)
            topic_tags = " ".join(str(tag) for tag in topic.get("tags") or [])

            for article in knowledge.get("knowledge_articles") or []:
                title = str(article.get("title") or "Untitled article")
                content = str(article.get("content") or "")
                haystack = f"{topic_name} {topic_tags} {title} {content}"
                score = self._score(query_tokens, haystack)
                if score <= 0:
                    continue
                scored.append(
                    (
                        score,
                        SourceDocument(
                            title=f"{topic_name}: {title}",
                            url=None,
                            snippet=content[:500],
                            metadata={
                                "provider": "local_corpus",
                                "article_id": article.get("article_id"),
                                "topic_file": path.name,
                                "score": score,
                            },
                        ),
                    )
                )

            for document in knowledge.get("source_documents") or []:
                title = str(document.get("title") or "Untitled source")
                content = str(document.get("full_text") or "")
                takeaways = " ".join(str(item) for item in document.get("key_takeaways") or [])
                haystack = f"{topic_name} {topic_tags} {title} {content} {takeaways}"
                score = self._score(query_tokens, haystack)
                if score <= 0:
                    continue
                scored.append(
                    (
                        score,
                        SourceDocument(
                            title=title,
                            url=document.get("provenance_url"),
                            snippet=(content or takeaways)[:500],
                            metadata={
                                "provider": "local_corpus",
                                "document_id": document.get("document_id"),
                                "citation_label": document.get("citation_label"),
                                "is_synthetic": document.get("is_synthetic"),
                                "topic_file": path.name,
                                "score": score,
                            },
                        ),
                    )
                )

        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:max_results]]

    @staticmethod
    def _score(query_tokens: set[str], text: str) -> float:
        text_tokens = _tokenize(text)
        if not text_tokens:
            return 0.0
        overlap = query_tokens & text_tokens
        if not overlap:
            return 0.0
        # Prefer denser overlap; slight boost for more matched terms.
        return len(overlap) / len(query_tokens) + 0.05 * len(overlap)
