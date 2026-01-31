from pathlib import Path


def test_readme_contains_reranker_documentation():
    """Verify README.md documents reranker feature."""
    readme_path = Path(__file__).parent.parent / "README.md"
    content = readme_path.read_text()

    # Check for required sections/keywords
    assert "--rerank" in content, "README missing --rerank flag documentation"
    assert "reranker" in content.lower(), "README missing reranker configuration section"
    assert "cohere" in content.lower(), "README missing Cohere provider documentation"
    assert "jina" in content.lower(), "README missing Jina provider documentation"
    assert "cross-encoder" in content.lower() or "local" in content.lower(), "README missing local reranker documentation"
    assert "COHERE_API_KEY" in content, "README missing API key setup instructions"
    assert "JINA_API_KEY" in content, "README missing Jina API key setup"


def test_readme_reranker_config_example():
    """Verify README contains reranker configuration example."""
    readme_path = Path(__file__).parent.parent / "README.md"
    content = readme_path.read_text()

    # Check for config structure
    assert '"provider"' in content, "README missing reranker provider field"
    assert '"model"' in content, "README missing reranker model field"
    assert '"candidates"' in content or "candidates" in content, "README missing candidates explanation"
    assert '"top_k"' in content or "top_k" in content, "README missing top_k explanation"
