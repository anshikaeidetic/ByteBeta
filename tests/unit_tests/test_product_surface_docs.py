from pathlib import Path


def test_readme_leads_with_proxy_and_byte_client_story() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "OpenAI-compatible gateway" in readme
    assert "from byte import ByteClient" in readme
    assert "python benchmark.py --provider openai --compare-baseline" in readme
    assert 'response["choices"][0]["message"]["content"]' in readme
    assert "[Architecture overview](docs/architecture.md)" in readme
    assert "[Route and auth matrix](docs/route-auth-matrix.md)" in readme


def test_usage_guide_defaults_to_byte_client() -> None:
    usage = Path("docs/usage.md").read_text(encoding="utf-8")

    assert 'ByteClient(mode="safe"' in usage
    assert "/v1/chat/completions" in usage
    assert "Client.aput" in usage
    assert "Advanced: low-level init helpers" in usage


def test_docs_toc_and_env_example_cover_release_surface() -> None:
    toc = Path("docs/toc.rst").read_text(encoding="utf-8")
    env_example = Path(".env.example").read_text(encoding="utf-8")

    for page in (
        "architecture.md",
        "service-topology.md",
        "control-plane.md",
        "inference-worker.md",
        "memory-service.md",
        "mcp-gateway.md",
        "route-auth-matrix.md",
        "environment-reference.md",
        "operator-runbook.md",
    ):
        assert page in toc

    assert "# Byte gateway basics" in env_example
    assert "BYTE_INTERNAL_TOKEN=" in env_example
    assert "BYTE_ADMIN_TOKEN=" in env_example
