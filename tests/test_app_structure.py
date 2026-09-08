"""
Tests for project layout and path integrity.
Uses absolute paths resolved from the project root for reliability.
"""

import os
from pathlib import Path


# Resolve project root relative to this test file
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_main_file_exists():
    assert (_PROJECT_ROOT / "app" / "main.py").exists()


def test_requirements_exists():
    assert (_PROJECT_ROOT / "requirements.txt").exists()


def test_app_folder_exists():
    assert (_PROJECT_ROOT / "app").exists()


def test_dockerfile_exists():
    assert (_PROJECT_ROOT / "dockerfile").exists()


def test_docker_compose_exists():
    assert (_PROJECT_ROOT / "docker-compose.yml").exists()


def test_ci_workflow_exists():
    assert (_PROJECT_ROOT / ".github" / "workflows" / "ci.yml").exists()


def test_env_example_exists():
    assert (_PROJECT_ROOT / ".env.example").exists()


def test_core_modules_exist():
    """Verify all documented source modules exist."""
    expected_modules = [
        "app/chains/rag_chain.py",
        "app/chains/router.py",
        "app/config/settings.py",
        "app/ingestion/loader.py",
        "app/ingestion/splitter.py",
        "app/ingestion/embedder.py",
        "app/retrieval/retriever.py",
        "app/llm/mistral_client.py",
        "app/llm/fallback.py",
        "app/utils/confidence.py",
        "app/utils/mlflow_logger.py",
        "app/experiment/manager.py",
        "app/experiment/tracker.py",
        "app/experiment/evaluator.py",
        "app/experiment/artifacts.py",
    ]
    for module in expected_modules:
        assert (_PROJECT_ROOT / module).exists(), f"Missing module: {module}"