from multi_agent_research_lab.services.search_client import SearchClient


def test_local_corpus_search_returns_sources() -> None:
    client = SearchClient()
    docs = client.search(
        "single agent vs multi-agent architectures research",
        max_results=3,
    )
    assert len(docs) >= 1
    assert docs[0].title
    assert docs[0].snippet
    assert docs[0].metadata.get("provider") == "local_corpus"
