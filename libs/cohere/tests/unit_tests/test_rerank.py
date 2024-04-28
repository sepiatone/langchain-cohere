"""Test chat model integration."""


from typing import cast

from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain_cohere import CohereRerank


def test_initialization() -> None:
    """Test chat model initialization."""
    CohereRerank(cohere_api_key="test")


def test_api_key_is_secret_string() -> None:
    """Test that the API key is stored as a SecretStr."""
    reranker = CohereRerank(
        cohere_api_key="secret-api-key",
        model="foo",
    )
    assert isinstance(reranker.cohere_api_key, SecretStr)


def test_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test that the API key is masked when passed from an environment variable."""
    monkeypatch.setenv("COHERE_API_KEY", "secret-api-key")
    reranker = CohereRerank(
        model="foo",
    )
    print(reranker.cohere_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test that the API key is masked when passed via the constructor."""
    reranker = CohereRerank(
        cohere_api_key="secret-api-key",
        model="foo",
    )
    print(reranker.cohere_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_uses_actual_secret_value_from_secretstr() -> None:
    """Test that the actual secret value is correctly retrieved."""
    reranker = CohereRerank(
        cohere_api_key="secret-api-key",
        model="foo",
    )
    assert (
        cast(SecretStr, reranker.cohere_api_key).get_secret_value() == "secret-api-key"
    )
