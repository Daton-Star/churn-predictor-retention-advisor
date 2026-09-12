"""Unit tests for the Gemini guardrail: the response is only ever allowed
to reach a caller with a validated action or an explicit manual_review
flag. No test here calls the real Gemini API (no network, no API key
needed) -- google.genai is faked via sys.modules injection.
"""

import sys
import types

import pytest

import genai_advisor
from config import ALLOWED_ACTIONS


def install_fake_genai(response_text=None, exception=None):
    """Register a fake google.genai module whose Client().models.generate_content()
    either returns `response_text` or raises `exception`, so
    generate_customer_explanation can run its real parsing/guardrail logic
    against a controlled response.
    """

    class FakeResponse:
        text = response_text

    class FakeModels:
        def generate_content(self, **kwargs):
            if exception is not None:
                raise exception
            return FakeResponse()

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = FakeModels()

    fake_module = types.ModuleType("google.genai")
    fake_module.Client = FakeClient
    fake_module.types = types.SimpleNamespace(GenerateContentConfig=lambda **kwargs: kwargs)
    sys.modules["google.genai"] = fake_module


@pytest.fixture(autouse=True)
def cleanup_fake_module():
    yield
    sys.modules.pop("google.genai", None)


def test_extract_json_plain():
    result = genai_advisor._extract_json('{"a": 1, "b": "two"}')
    assert result == {"a": 1, "b": "two"}


def test_extract_json_fenced_code_block():
    text = '```json\n{"a": 1}\n```'
    assert genai_advisor._extract_json(text) == {"a": 1}


def test_extract_json_garbage_raises():
    with pytest.raises(Exception):
        genai_advisor._extract_json("not json at all")


def test_no_api_key_flags_manual_review(monkeypatch):
    monkeypatch.setattr(genai_advisor, "get_api_key", lambda: None)
    result = genai_advisor.generate_customer_explanation(
        customer_id=1, churn_probability=0.8, top_factors=[]
    )
    assert result["needs_manual_review"] is True
    assert result["recommended_action"] is None
    assert "API key" in result["error"]


def test_valid_action_passes_guardrail(monkeypatch):
    monkeypatch.setattr(genai_advisor, "get_api_key", lambda: "fake-key")
    monkeypatch.setattr(genai_advisor, "get_model_name", lambda: "gemini-2.5-flash")
    install_fake_genai(
        response_text='{"explanation": "Low order frequency and no recent purchases.", '
        '"recommended_action": "send discount offer"}'
    )

    result = genai_advisor.generate_customer_explanation(
        customer_id=1, churn_probability=0.8, top_factors=[]
    )
    assert result["needs_manual_review"] is False
    assert result["recommended_action"] == "send discount offer"
    assert result["error"] is None


def test_action_matching_is_case_insensitive(monkeypatch):
    monkeypatch.setattr(genai_advisor, "get_api_key", lambda: "fake-key")
    monkeypatch.setattr(genai_advisor, "get_model_name", lambda: "gemini-2.5-flash")
    install_fake_genai(
        response_text='{"explanation": "They have gone quiet.", "recommended_action": "Send Discount Offer"}'
    )

    result = genai_advisor.generate_customer_explanation(
        customer_id=1, churn_probability=0.8, top_factors=[]
    )
    assert result["needs_manual_review"] is False
    # Matched action is normalized to the canonical casing from ALLOWED_ACTIONS.
    assert result["recommended_action"] in ALLOWED_ACTIONS


def test_invalid_action_flagged_for_manual_review(monkeypatch):
    monkeypatch.setattr(genai_advisor, "get_api_key", lambda: "fake-key")
    monkeypatch.setattr(genai_advisor, "get_model_name", lambda: "gemini-2.5-flash")
    install_fake_genai(
        response_text='{"explanation": "They seem unhappy.", "recommended_action": "offer a free unicorn"}'
    )

    result = genai_advisor.generate_customer_explanation(
        customer_id=1, churn_probability=0.8, top_factors=[]
    )
    assert result["needs_manual_review"] is True
    assert "outside the allowed list" in result["error"]


def test_api_exception_flagged_not_raised(monkeypatch):
    monkeypatch.setattr(genai_advisor, "get_api_key", lambda: "fake-key")
    monkeypatch.setattr(genai_advisor, "get_model_name", lambda: "gemini-2.5-flash")
    install_fake_genai(exception=RuntimeError("network exploded"))

    result = genai_advisor.generate_customer_explanation(
        customer_id=1, churn_probability=0.8, top_factors=[]
    )
    assert result["needs_manual_review"] is True
    assert "Gemini call failed" in result["error"]
