"""AIRAWAT metadata (sentiment / risks / topics) JSON: parse, validate, repair, fail honestly.

Root cause pinned here (measured live on meeting 1647): the model answered ``"action_items": [\\n    * Review ...``
- Markdown bullets inside a JSON array - inside a ```json fence. ``json.loads`` raised
``Expecting value: line 24 column 5 (char 660)``; at temperature 0.1 / top_p 0.1 the same prompt gave the same bytes
on every attempt, so the pipeline's identical retry could never succeed and the whole meeting failed.

Everything is mocked (no network, no database).
"""

import json
from types import SimpleNamespace

import pytest
from langchain_core.runnables import RunnableLambda

from backend.services import llm_workflow as wf
from backend.utils.exceptions import AirawatError
from backend.utils.helpers import LLMJsonError, parse_llm_json_object

TX = "Speaker_1: We will review the release plan and finish the API testing by Friday.\n" * 6

VALID = {
    "risks": "Vendor contract is not signed yet.",
    "project_name": "Release Plan",
    "sentiment": "Positive",
    "emotions": "Confidence: 70%, Focus: 30%",
    "questions_raised": ["Who owns the API testing?"],
    "keyword_frequency": {"release": 5, "testing": 3},
    "action_items": ["Review the release plan", "Finish the API testing by Friday"],
    "decisions": ["Deploy on Friday"],
}

# The exact shape AIRAWAT returned (content shortened): valid JSON except for the bullets in two arrays.
BULLETS_IN_ARRAYS = (
    "```json\r\n{\r\n"
    '  "risks": "No major risks or blockers identified during this session.",\r\n'
    '  "project_name": "AI Meeting",\r\n'
    '  "sentiment": "Positive",\r\n'
    '  "emotions": "Confidence: 80%, Focus: 20%",\r\n'
    '  "questions_raised": [\r\n    "What is the role of the LLM engine?"\r\n  ],\r\n'
    '  "keyword_frequency": {\r\n    "AI": 10,\r\n    "Meeting": 15\r\n  },\r\n'
    '  "action_items": [\r\n    * Review the release plan,\r\n    * Deploy on Friday\r\n  ],\r\n'
    '  "decisions": [\r\n    * The team will deploy on Friday\r\n  ]\r\n}\r\n```'
)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(wf.time, "sleep", lambda s: None)


class ScriptedLLM:
    """An LLM whose replies are scripted; records every message list it received."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def as_runnable(self):
        def _answer(messages):
            self.calls.append(messages)
            reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
            if isinstance(reply, Exception):
                raise reply
            return SimpleNamespace(content=reply if isinstance(reply, str) else json.dumps(reply),
                                   response_metadata={"finish_reason": "stop"})
        return RunnableLambda(_answer)


def _run(*replies):
    llm = ScriptedLLM(*replies)
    return wf._extract_metadata_json(llm.as_runnable(), TX), llm


# ── the parser ─────────────────────────────────────────────────────────────────────────────────────

def test_a_bare_json_object_parses():
    assert parse_llm_json_object(json.dumps(VALID))["sentiment"] == "Positive"


def test_a_markdown_fenced_object_parses():
    assert parse_llm_json_object("```json\n" + json.dumps(VALID) + "\n```")["project_name"] == "Release Plan"


def test_prose_around_the_object_and_think_blocks_are_ignored():
    raw = "<think>hmm {not json}</think>Here is the JSON:\n" + json.dumps(VALID) + "\nHope that helps! {thanks}"
    assert parse_llm_json_object(raw)["risks"].startswith("Vendor")


def test_braces_inside_strings_do_not_end_the_object_early():
    obj = {"risks": "uses {curly} braces and a \"quote\"", "a": 1}
    assert parse_llm_json_object("text " + json.dumps(obj) + " tail")["a"] == 1


def test_the_real_bullets_in_array_reply_is_rejected_with_a_precise_content_free_reason():
    with pytest.raises(LLMJsonError) as err:
        parse_llm_json_object(BULLETS_IN_ARRAYS)
    assert err.value.kind == "malformed"
    assert "Expecting value" in err.value.reason and "line 14 column 5" in err.value.reason
    assert "list bullet" in err.value.reason
    assert "Review the release plan" not in err.value.reason            # no reply / transcript content in the reason


@pytest.mark.parametrize(
    "raw,kind",
    [("", "empty"), ("   \n ", "empty"), (None, "empty"),
     ('{"risks": "cut off in the mid', "truncated"),
     ('{"risks": "x", "sentiment": "Positive",', "truncated"),
     ("no json here at all", "malformed"),
     ("[1, 2, 3]", "malformed"),                     # an array is not the object that was asked for
     ('{"a": 1,}', "malformed"),                     # trailing comma
     ("{'a': 1}", "malformed")],                     # single quotes
)
def test_unusable_replies_are_classified(raw, kind):
    with pytest.raises(LLMJsonError) as err:
        parse_llm_json_object(raw)
    assert err.value.kind == kind


def test_a_json_array_reply_is_not_an_object():
    with pytest.raises(LLMJsonError):
        parse_llm_json_object('  ["a", "b"]  ')


# ── validation: field types, no fabricated defaults ─────────────────────────────────────────────────

def test_a_valid_reply_is_returned_in_the_pipeline_shape_on_the_first_call():
    out, llm = _run(VALID)
    assert len(llm.calls) == 1
    assert out["sentiment"] == "Positive" and out["project_name"] == "Release Plan"
    assert json.loads(out["keyword_frequency"]) == {"release": 5, "testing": 3}
    assert json.loads(out["questions_raised"]) == ["Who owns the API testing?"]
    assert "Review the release plan" in out["action_items"] and "Deploy on Friday" in out["decisions"]
    assert out["agenda_key_decisions"] == []                  # agenda/decisions come only from the dedicated call


def test_a_fenced_valid_reply_needs_no_retry():
    out, llm = _run("```json\n" + json.dumps(VALID) + "\n```")
    assert len(llm.calls) == 1 and out["sentiment"] == "Positive"


def test_sentiment_is_normalised_and_emotions_dict_becomes_text():
    reply = dict(VALID, sentiment="  positive.", emotions={"Confidence": 60, "Focus": 40.0})
    out, _ = _run(reply)
    assert out["sentiment"] == "Positive" and out["emotions"] == "Confidence: 60%, Focus: 40%"


def test_an_empty_action_item_array_is_the_models_explicit_none():
    out, _ = _run(dict(VALID, action_items=[], decisions=[]))
    assert out["action_items"] == "" and out["decisions"] == ""


def test_topics_are_capped_and_numeric_strings_coerced():
    topics = {f"term{i}": i for i in range(1, 16)}
    topics["five"] = "5"
    out, _ = _run(dict(VALID, keyword_frequency=topics))
    parsed = json.loads(out["keyword_frequency"])
    assert len(parsed) == 10 and parsed["term15"] == 15 and "term1" not in parsed


def test_a_missing_field_is_never_replaced_by_a_default():
    broken = {k: v for k, v in VALID.items() if k != "sentiment"}
    with pytest.raises(AirawatError) as err:
        _run(broken)
    assert "missing field(s)" in str(err.value) and '"sentiment"' in str(err.value)
    assert "Neutral" not in str(err.value)


@pytest.mark.parametrize(
    "patch,fragment",
    [({"sentiment": "Mostly positive"}, '"sentiment" must be exactly one of'),
     ({"questions_raised": "none"}, '"questions_raised" must be an array of strings'),
     ({"keyword_frequency": ["a", "b"]}, '"keyword_frequency" must be an object'),
     ({"keyword_frequency": {"a": "many"}}, '"keyword_frequency" must be an object'),
     ({"risks": 5}, '"risks" must be a string'),
     ({"project_name": ""}, '"project_name" is an empty string'),
     ({"action_items": [1, 2]}, '"action_items" must contain only non-empty strings')],
)
def test_wrong_types_are_reported_by_field_name(patch, fragment):
    with pytest.raises(AirawatError) as err:
        _run(dict(VALID, **patch))
    assert fragment in str(err.value)


# ── repair loop: bounded, with the validation error, never the same prompt blindly ───────────────────

def test_bullets_in_arrays_are_repaired_by_asking_again_with_the_validation_error():
    out, llm = _run(BULLETS_IN_ARRAYS, VALID)
    assert len(llm.calls) == 2 and out["sentiment"] == "Positive"
    repair_messages = llm.calls[1].to_messages()
    roles = [m.type for m in repair_messages]
    assert roles == ["system", "human", "ai", "human"]
    assert "could not be used" in repair_messages[-1].content
    assert "line 14 column 5" in repair_messages[-1].content and "double-quoted string" in repair_messages[-1].content
    assert "* Review the release plan" in repair_messages[2].content        # the model sees what it wrote


def test_the_first_call_is_just_the_prompt_and_the_transcript():
    _, llm = _run(VALID)
    assert [m.type for m in llm.calls[0].to_messages()] == ["system", "human"]


def test_retry_success_on_the_last_allowed_attempt():
    out, llm = _run(BULLETS_IN_ARRAYS, "not json", VALID)
    assert len(llm.calls) == 3 and out["project_name"] == "Release Plan"


def test_retry_exhaustion_is_bounded_and_fails_with_every_reason_kept():
    with pytest.raises(AirawatError) as err:
        _run(BULLETS_IN_ARRAYS)                       # the model keeps saying the same thing
    assert err.value.details["attempts"] == 1 + wf._METADATA_FORMAT_REPAIRS == 3
    assert len(err.value.details["reasons"]) == 3 and err.value.details["stage"] == "metadata"
    assert "no valid JSON after 3 attempt(s)" in str(err.value)


def test_exhaustion_makes_exactly_three_calls_never_more():
    llm = ScriptedLLM(BULLETS_IN_ARRAYS)
    with pytest.raises(AirawatError):
        wf._extract_metadata_json(llm.as_runnable(), TX)
    assert len(llm.calls) == 3


def test_an_empty_reply_is_retried_then_reported_as_empty():
    with pytest.raises(AirawatError) as err:
        _run("")
    assert "the reply was empty" in str(err.value)


def test_a_truncated_reply_is_reported_as_cut_off():
    with pytest.raises(AirawatError) as err:
        _run('{"risks": "cut off in the mid')
    assert "not closed" in str(err.value)


def test_an_http_failure_is_not_a_format_error_and_is_not_repaired():
    llm = ScriptedLLM(ConnectionError("HTTP 503 from AIRAWAT"))
    with pytest.raises(AirawatError) as err:
        wf._extract_metadata_json(llm.as_runnable(), TX)
    assert len(llm.calls) == 1                         # transport retries live in the client, not here
    assert "ConnectionError" in str(err.value) and "meeting metadata" in str(err.value)


def test_the_reply_content_is_never_written_to_the_log(caplog):
    secret = "ZEBRA-QUOKKA-CONFIDENTIAL"
    reply = BULLETS_IN_ARRAYS.replace("Review the release plan", secret)
    with caplog.at_level("INFO", logger=wf.logger.name):
        with pytest.raises(AirawatError):
            _run(reply)
    assert secret not in caplog.text
    assert "[metadata] attempt=1/3" in caplog.text and "validation=failed" in caplog.text


# ── process_meeting_report: a metadata failure names its sections, other sections are not corrupted ────

@pytest.fixture()
def report_env(monkeypatch):
    summary = SimpleNamespace(text="The team reviewed the release plan and agreed to deploy on Friday.", method="llm",
                              llm_calls=1, violations=[])
    monkeypatch.setattr(wf, "generate_meeting_summary", lambda *a, **k: summary)
    monkeypatch.setattr(wf.model_router, "get_llm", lambda **kw: object())
    monkeypatch.setattr(wf, "get_llm", lambda *a, **k: object())
    monkeypatch.setattr(wf, "_generate_section",
                        lambda llm, prompt, transcript, max_chars=None: "### Minutes of the meeting\n- release reviewed\n- deploy on Friday")
    monkeypatch.setattr(wf, "_extract_agenda_key_decisions", lambda llm, t, user_agenda="": [])
    monkeypatch.setattr(wf, "_extract_action_items", lambda llm, t, reference_dt=None: [])


def test_a_failed_metadata_section_is_named_and_the_others_stay_cached(report_env, monkeypatch):
    def metadata_fails(llm, t):
        raise AirawatError("no valid JSON after 3 attempt(s)", details={"stage": "metadata"})

    monkeypatch.setattr(wf, "_extract_metadata_json", metadata_fails)
    cache = {}
    with pytest.raises(AirawatError) as err:
        wf.process_meeting_report(TX, "2026-10-12", "10:00", "00:30:00", section_cache=cache)
    assert err.value.details["failed_sections"] == ["metadata"]
    # the sections that DID succeed are kept for the retry and are not replaced by anything
    assert {"summary", "mom", "agenda_decisions", "action_items"} <= set(cache)
    assert "decisions_risks_sentiment" not in cache


def test_a_report_never_contains_a_default_sentiment_when_metadata_failed(report_env, monkeypatch):
    monkeypatch.setattr(wf, "_extract_metadata_json", lambda llm, t: (_ for _ in ()).throw(AirawatError("bad")))
    with pytest.raises(AirawatError):
        wf.process_meeting_report(TX, "2026-10-12", "10:00", "00:30:00")
