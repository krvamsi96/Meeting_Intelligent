"""Meeting-derived text baked into a prompt TEMPLATE must not be parsed as template variables."""

from types import SimpleNamespace

import pytest
from langchain_core.runnables import RunnableLambda

from backend.services import llm_workflow as wf

TITLES = ["Weekly sync", "Q3 {budget} review", "{}", "Plan {a} and {b}", "{{already}} doubled", "}"]


@pytest.fixture()
def fake_llm(monkeypatch):
    seen = {}

    def _respond(prompt_value):
        seen["prompt"] = prompt_value.to_string() if hasattr(prompt_value, "to_string") else str(prompt_value)
        return SimpleNamespace(content="the answer")

    monkeypatch.setattr(wf, "get_llm", lambda *a, **k: RunnableLambda(_respond))
    return seen


@pytest.mark.parametrize("title", TITLES)
def test_chatbot_answers_for_any_meeting_title(title, fake_llm):
    """Regression: a title with braces made every chat reply 'I encountered an error…'."""
    answer = wf.generate_chatbot_response("what was decided?", ["chunk one"], [], title, "general")
    assert answer == "the answer"


@pytest.mark.parametrize("title", ["Q3 {budget} review", "Plan {a} and {b}"])
def test_the_title_reaches_the_model_verbatim(title, fake_llm):
    wf.generate_chatbot_response("what was decided?", ["chunk one"], [], title, "general")
    assert title in fake_llm["prompt"], "braces must be escaped for the template, then rendered back literally"


def test_query_and_context_with_braces_are_passed_through_as_variables(fake_llm):
    wf.generate_chatbot_response("what is {x}?", ["ctx with {y} and }"], [{"role": "user", "content": "hist {z}"}], "T", "general")
    prompt = fake_llm["prompt"]
    assert "what is {x}?" in prompt and "ctx with {y} and }" in prompt and "hist {z}" in prompt


def test_escape_helper_is_inverse_of_template_rendering():
    from langchain_core.prompts import ChatPromptTemplate

    raw = "a {b} c {{d}}"
    rendered = ChatPromptTemplate.from_messages([("system", wf.escape_template_braces(raw))]).format_messages()[0].content
    assert rendered == raw
