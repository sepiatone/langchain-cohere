"""Test embedding model integration."""

from typing import cast

from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain_cohere.embeddings import CohereEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    CohereEmbeddings(cohere_api_key="test")


def test_api_key_is_secret_string() -> None:
    """Test that the API key is stored as a SecretStr."""
    embeddings_model = CohereEmbeddings(
        cohere_api_key="secret-api-key",
        model="foo",
    )
    assert isinstance(embeddings_model.cohere_api_key, SecretStr)


def test_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test that the API key is masked when passed from an environment variable."""
    monkeypatch.setenv("COHERE_API_KEY", "secret-api-key")
    embeddings_model = CohereEmbeddings(
        model="foo",
    )
    print(embeddings_model.cohere_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test that the API key is masked when passed via the constructor."""
    embeddings_model = CohereEmbeddings(
        cohere_api_key="secret-api-key",
        model="foo",
    )
    print(embeddings_model.cohere_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_uses_actual_secret_value_from_secretstr() -> None:
    """Test that the actual secret value is correctly retrieved."""
    embeddings_model = CohereEmbeddings(
        cohere_api_key="secret-api-key",
        model="foo",
    )
    assert (
        cast(SecretStr, embeddings_model.cohere_api_key).get_secret_value()
        == "secret-api-key"
    )
