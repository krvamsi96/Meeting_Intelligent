"""Agenda -> decision mapping is built from what was said, never from filler.

Nothing here needs AIRAWAT or a database: the model's answer is given as data and the validation / mapping / API layers
are exercised directly. Each test names the behaviour it protects.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.services import final_aggregation_service as fas
from backend.services import llm_workflow as wf

REPO_ROOT = Path(__file__).resolve().parents[2]

TRANSCRIPT = (
    "Speaker_1: Good morning everyone. Let's get started.\n"
    "Speaker_2: Morning. I have the status of the release ready.\n"
    "Speaker_1: Great. Today we discussed the migration timeline and decided to move production deployment to October 15.\n"
    "Speaker_2: Understood. That gives the testing team two more weeks.\n"
    "Speaker_1: We will discuss the vendor contract later. Also, we will make one task to prepare a full feature list.\n"
)


@pytest.fixture(autouse=True)
def _no_embedding_model(monkeypatch):
    """Scores / semantic de-duplication are not what is under test: keep them deterministic and model-free."""
    from backend.services import agenda_service

    monkeypatch.setattr(wf, "compute_batch_match_scores",
                        lambda pairs: [{"match_score": 50.0, "confidence_level": "Fair Match", "model_name": "stub"} for _ in pairs])
    monkeypatch.setattr(agenda_service, "deduplicate_semantic", lambda texts, threshold=None: list(range(len(texts))))


def _validate(items, transcript=TRANSCRIPT, source="llm_detected"):
    return wf.validate_and_format_agenda_key_decisions(items, default_source=source, transcript=transcript)


MIGRATION = {
    "agenda": "Migration timeline",
    "key_decision": "Production deployment moved to October 15",
    "evidence": "Today we discussed the migration timeline and decided to move production deployment to October 15.",
}


# ── no hardcoded / generic agenda or decision text ─────────────────────────────────────────────────

@pytest.mark.parametrize("title", ["Review", "Discussion", "Review and Discussion", "General Discussion", "Team Meeting Update",
                                   "Discussion and review", "Meeting", "Overview"])
def test_a_generic_agenda_title_is_never_kept(title):
    assert _validate([{"agenda": title, "key_decision": ""}]) == []


@pytest.mark.parametrize("title", ["Migration timeline", "Vendor contract", "Budget review", "Q3 review", "Security audit findings"])
def test_a_title_that_names_a_subject_is_kept(title):
    # A generated topic must be about something that was SAID: the transcript has to contain the title's words.
    spoken = TRANSCRIPT + f"Speaker_1: Next we covered the {title.lower()} in detail.\n"
    out = _validate([{"agenda": title, "key_decision": ""}], transcript=spoken)
    assert [i["agenda"] for i in out] == [title]


@pytest.mark.parametrize("title", ["Zeppelin procurement", "Underwater basket weaving"])
def test_a_generated_topic_that_was_never_discussed_is_dropped(title):
    assert _validate([{"agenda": title, "key_decision": ""}]) == []


def test_a_user_typed_agenda_is_kept_even_when_the_transcript_never_mentions_it():
    out = _validate([{"agenda": "Zeppelin procurement", "key_decision": "", "source": "user"}])
    assert [i["agenda"] for i in out] == ["Zeppelin procurement"] and out[0]["agenda_type"] == "explicit"


def test_an_agenda_the_user_typed_is_kept_even_if_it_is_generic():
    out = _validate([{"agenda": "Review", "key_decision": "", "source": "user"}], source="user")
    assert [i["agenda"] for i in out] == ["Review"]


@pytest.mark.parametrize("decision", ["Review", "Discussion", "Review and Discussion", "Discussed the topic", "No formal decision recorded."])
def test_a_generic_decision_is_not_a_decision(decision):
    (item,) = _validate([{"agenda": "Migration timeline", "key_decision": decision, "evidence": "decided to move production deployment"}])
    assert item["decision"] == "" and item["decision_grounded"] is False


def test_no_agenda_at_all_is_an_empty_list_not_a_made_up_one():
    assert _validate([]) == []
    assert _validate([{"agenda": "", "key_decision": "x"}]) == []


# ── the user's own example ────────────────────────────────────────────────────────────────────────

def test_a_clearly_stated_decision_is_mapped_to_its_agenda():
    (item,) = _validate([MIGRATION])
    assert item["agenda"] == "Migration timeline"
    assert item["decision"] == "Production deployment moved to October 15"
    assert item["decision_grounded"] is True


def test_the_mapping_is_not_vetoed_by_a_low_embedding_similarity():
    """'Migration timeline' vs 'Production deployment moved to October 15' scores ~36 % in embedding space although the
    pairing is exactly right. The final mapping used to drop it (has_decision False) because 36 < 45."""
    rows = [SimpleNamespace(agenda_id_str="A1", agenda_order=1, agenda="Migration timeline",
                            key_decision="Production deployment moved to October 15", decision_grounded=True, match_score=35.9)]
    (mapped,) = fas._build_agenda_decision_mapping(rows)
    assert mapped["has_decision"] is True and mapped["decision"] == "Production deployment moved to October 15"
    assert mapped["match_score"] == 35.9
    assert "confidence_level" not in mapped               # no "Good/Fair Match" label is claimed for a low similarity


def test_an_agenda_without_a_grounded_decision_maps_to_nothing():
    rows = [SimpleNamespace(agenda_id_str="A1", agenda_order=1, agenda="Vendor contract",
                            key_decision="No formal decision recorded.", decision_grounded=False, match_score=None),
            SimpleNamespace(agenda_id_str="A2", agenda_order=2, agenda="Office move", key_decision="Move to floor three",
                            decision_grounded=False, match_score=80.0)]
    mapped = fas._build_agenda_decision_mapping(rows)
    assert [m["has_decision"] for m in mapped] == [False, False]
    assert all("decision" not in m for m in mapped)


# ── nothing invented: a decision needs the transcript behind it ─────────────────────────────────

def test_a_quote_that_is_not_in_the_transcript_removes_the_decision_but_keeps_the_agenda():
    made_up = dict(MIGRATION, evidence="Everyone unanimously approved moving deployment to November 30.")
    (item,) = _validate([made_up])
    assert item["agenda"] == "Migration timeline"
    assert item["decision"] == "" and item["decision_grounded"] is False and item["evidence"] is None


def test_a_decision_without_any_quote_is_not_kept_when_the_transcript_is_known():
    (item,) = _validate([dict(MIGRATION, evidence="")])
    assert item["decision"] == "" and item["decision_grounded"] is False


@pytest.mark.parametrize("quote", [
    "We will discuss the vendor contract later.",                       # a deferral
    "Also, we will make one task to prepare a full feature list.",      # a task
])
def test_a_task_or_a_deferral_is_not_presented_as_a_decision(quote):
    (item,) = _validate([{"agenda": "Vendor contract", "key_decision": "Prepare a full feature list", "evidence": quote}])
    assert item["decision"] == "" and item["decision_grounded"] is False


def test_paraphrased_punctuation_and_case_in_a_real_quote_still_verify():
    quote = "today we discussed the MIGRATION timeline, and decided to move production deployment to october 15"
    (item,) = _validate([dict(MIGRATION, evidence=quote)])
    assert item["decision_grounded"] is True


def test_without_the_transcript_the_older_evidence_or_decisive_verb_rule_still_applies():
    (item,) = _validate([dict(MIGRATION)], transcript="")
    assert item["decision_grounded"] is True


def test_a_non_latin_transcript_is_not_judged_by_a_translated_quote():
    hindi = "वक्ता: हमने माइग्रेशन की समय सीमा पर चर्चा की और तय किया कि तैनाती पंद्रह अक्टूबर को होगी। " * 3
    (item,) = _validate([MIGRATION], transcript=hindi)
    assert item["decision_grounded"] is True


# ── the API and the other readers say "no decision" as empty, never as filler text ────────────────

def _akd(**over):
    base = dict(id=1, agenda_id_str="A1", agenda_order=1, agenda="Vendor contract", agenda_type="inferred", agenda_source="llm_detected",
                key_decision="No formal decision recorded.", decision_grounded=False, match_score=None, embedding_model="stub",
                speaker=None, evidence=None, reason=None, timestamp=None, confidence=None)
    base.update(over)
    return SimpleNamespace(**base)


def test_the_agenda_api_returns_an_empty_decision_not_the_stored_marker():
    from backend.api.meetings import _build_agenda_item

    item = _build_agenda_item(_akd())
    assert item.key_decision == "" and item.decision_text == "" and item.decision == ""
    assert item.decision_grounded is False and item.match_score is None


def test_the_agenda_api_marks_a_grounded_decision_grounded_whatever_its_similarity_score():
    from backend.api.meetings import _build_agenda_item

    item = _build_agenda_item(_akd(agenda="Migration timeline", key_decision="Production deployment moved to October 15",
                                   decision_grounded=True, match_score=35.9))
    assert item.decision == "Production deployment moved to October 15"
    assert item.decision_grounded is True and item.match_score == 35.9 and item.confidence_level is None


def test_exports_show_not_specified_instead_of_a_made_up_agenda():
    from backend.services import export_service as ex

    assert ex._meeting_agenda_text(SimpleNamespace(agenda=None, agenda_key_decisions=[])) == "Not specified"
    assert ex._meeting_agenda_text(SimpleNamespace(agenda="Board review", agenda_key_decisions=[])) == "Board review"
    topics = [SimpleNamespace(agenda="Migration timeline"), SimpleNamespace(agenda="Vendor contract")]
    assert ex._meeting_agenda_text(SimpleNamespace(agenda="", agenda_key_decisions=topics)) == "Migration timeline; Vendor contract"
    assert ex._real_decision("No formal decision recorded.") == "" and ex._real_decision("Move to floor three") == "Move to floor three"


@pytest.mark.parametrize("path, forbidden", [
    ("backend/workers/pipelines.py", "Discussion recording"),
    ("backend/services/export_service.py", "Discussion and review of session recording"),
    ("backend/services/export_service.py", "Discussion Topic {i+1}"),
    ("backend/rag/agent_router.py", '"General Discussion"'),
    ("backend/rag/agent_router.py", '"Discussion concluded"'),
    ("backend/services/document_service.py", "Document analysis and review"),
    ("backend/services/document_service.py", "Document content review and analysis"),
])
def test_the_hardcoded_agenda_and_decision_filler_is_gone_from_the_source(path, forbidden):
    assert forbidden not in (REPO_ROOT / path).read_text(encoding="utf-8")


def test_the_extraction_prompt_forbids_generic_titles_and_allows_an_empty_agenda():
    from backend.prompts.meeting_prompts import agenda_key_decisions_prompt

    prompt = agenda_key_decisions_prompt()
    assert "empty list" in prompt and "NEVER use a generic title" in prompt and "'Review and Discussion'" in prompt


# ── meetings stored by the old code keep no generated agenda either ────────────────────────────────

@pytest.mark.parametrize("stored", ["Discussion recording: After 0011 1", "Discussion and review of session recording",
                                    "Document analysis and review", "Document content review and analysis", "  ", None])
def test_a_placeholder_the_platform_used_to_generate_reads_as_no_agenda(stored):
    from backend.utils.helpers import clean_agenda_text

    assert clean_agenda_text(stored) is None


@pytest.mark.parametrize("real", ["Board review of Q3 budget", "Migration timeline; Vendor contract", "Discussion of the migration plan"])
def test_an_agenda_somebody_actually_supplied_is_untouched(real):
    from backend.utils.helpers import clean_agenda_text

    assert clean_agenda_text(real) == real


def test_the_meeting_api_schema_hides_a_stored_placeholder_agenda():
    """Rows written before this fix still hold 'Discussion and review of session recording' / 'Discussion recording: ...'."""
    from datetime import datetime

    from backend.schemas import schemas

    def meeting(agenda):
        return schemas.Meeting(id=1, filename="a.mp4", upload_time=datetime(2026, 9, 29), status="completed", agenda=agenda)

    assert meeting("Discussion and review of session recording").agenda is None
    assert meeting("Discussion recording: 8").agenda is None
    assert meeting("Budget approval").agenda == "Budget approval"


# ── the decision wording just outside the quoted words still counts ──────────────────────────────────

SPLIT_TRANSCRIPT = (
    "Speaker_1: Great. Today we discuss the migration timeline and decided to move production.\n"
    "Speaker_2: Deployment to October 15th. Understood. That gives the testing team.\n"
)


def test_a_quote_that_starts_after_the_decision_words_is_still_a_decision():
    """A sentence split across two speaker turns, and the model quoted only its tail: 'decided to move production' is
    a few words before the quote. This exact case (measured end to end) lost a real decision."""
    (item,) = _validate([{"agenda": "Migration timeline", "key_decision": "Move production deployment to October 15th",
                          "evidence": "Deployment to October 15th. Understood."}], transcript=SPLIT_TRANSCRIPT)
    assert item["decision_grounded"] is True and item["decision"].startswith("Move production deployment")


def test_a_decision_far_away_from_a_task_does_not_turn_the_task_into_a_decision():
    filler = " ".join(["and then we talked about many other unrelated things during the long meeting"] * 4)
    transcript = f"Speaker_1: Last quarter we decided to rename the team. {filler}. Then we will make one task to prepare the feature list."
    (item,) = _validate([{"agenda": "Feature list", "key_decision": "Prepare the feature list",
                          "evidence": "we will make one task to prepare the feature list"}], transcript=transcript)
    assert item["decision"] == "" and item["decision_grounded"] is False
