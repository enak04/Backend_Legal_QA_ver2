"""
tests/test_conversational.py - Tests for the human-like conversation layer.

Covers:
- Shared voice inlined into prompts (and part of the prompt version hash)
- Talker receives progress + recent openers
- Whole-message affirmation / "done" detection (no substring false positives)
- CONFIRMING: confirmation by intent or short affirmation; "not right" and mixed fixes -> COLLECTING
- Mid-conversation questions: answered now via the legal service, or deferred naturally
- Presenter: quick mode guarded, disclaimer only once, follow-up guarded with honest fallback,
  code guard allows the person's own facts
- Safety: reviewed lead-in placed before the verbatim helpline message
"""

import pytest

from src.db import create_session, get_session
from src.llm import FakeLLM
from src.messages import DISCLAIMER_SHORT, FOLLOWUP_FALLBACK_REPLY
from src.pipeline import handle_turn, is_done_signal, is_short_affirmation
from src.presenter import check_code_guard, present
from src.prompts import load_prompt
from src.safety import get_helplines_config, get_helpline_reply
from src.schemas import (
    AffectReading,
    CaseFile,
    CasePatch,
    Fact,
    FaithfulnessReport,
    Gap,
    GapAnalysis,
    HardCore,
    Intent,
    ServiceQuery,
    ServiceResponse,
    Source,
    Status,
    UnderstandOutput,
)
from src.service_client import ServiceClient
from src.talker import build_talker_reply, extract_recent_openers


class CountingServiceClient(ServiceClient):
    """Mock service client that records queries and can be forced to fail."""

    def __init__(self, fail: bool = False):
        super().__init__(use_mock=True)
        self.fail = fail
        self.queries: list[ServiceQuery] = []

    async def ask(self, query, session_id=None):
        self.queries.append(query)
        if self.fail:
            return ServiceResponse(status="error")
        return await super().ask(query, session_id=session_id)


NOTICE_GAP = Gap(
    topic="Notice",
    question_hint="Did he give you any warning before changing the locks?",
    why_it_matters="It changes what you can do.",
    impact="high",
)


# ---------------------------------------------------------------------------
# Voice and talker inputs
# ---------------------------------------------------------------------------

def test_voice_is_inlined_into_conversational_prompts():
    for name in ("talker", "confirm_summary", "presenter"):
        text, _ = load_prompt(name)
        assert "{{voice}}" not in text
        assert "WHO YOU ARE" in text
        assert "PHRASES YOU NEVER USE" in text


def test_voice_change_changes_prompt_version(tmp_path):
    (tmp_path / "demo.txt").write_text("{{voice}}\nHello {{name}}", encoding="utf-8")
    (tmp_path / "voice.txt").write_text("Be kind.", encoding="utf-8")
    text_1, version_1 = load_prompt("demo", prompts_dir=tmp_path)
    (tmp_path / "voice.txt").write_text("Be very kind.", encoding="utf-8")
    text_2, version_2 = load_prompt("demo", prompts_dir=tmp_path)
    assert text_1.startswith("Be kind.")
    assert text_2.startswith("Be very kind.")
    assert version_1 != version_2


def test_extract_recent_openers():
    messages = [
        {"role": "assistant", "text": "Got it. Where is the flat?"},
        {"role": "user", "text": "Pune"},
        {"role": "assistant", "text": "That helps a lot! Did he give any notice?"},
    ]
    openers = extract_recent_openers(messages)
    assert "- Got it." in openers
    assert "- That helps a lot!" in openers
    assert "Pune" not in openers
    assert extract_recent_openers([]) == "(none yet)"


@pytest.mark.asyncio
async def test_talker_receives_progress_and_openers():
    fake = FakeLLM(responses={"talker": "Ok."})
    session = {"_id": "s1", "status": "COLLECTING", "language": "en"}
    messages = [{"role": "assistant", "text": "Locked out overnight, that's awful. Where is this?"}]
    await build_talker_reply(
        session=session,
        messages=messages,
        llm=fake,
        stream=False,
        gaps=[NOTICE_GAP],
        progress="Close: about one or two important things left.",
    )
    system = fake.call_history[0]["system"]
    assert "Close: about one or two important things left." in system
    assert "- Locked out overnight, that's awful." in system
    assert "What to find out: Did he give you any warning before changing the locks?" in system


# ---------------------------------------------------------------------------
# Short-reply detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("msg", ["yes", "Yes!", "yep, thanks", "That's right.", "go ahead", "haan ji", "sí", "retry"])
def test_short_affirmations(msg):
    assert is_short_affirmation(msg)


@pytest.mark.parametrize("msg", [
    "no, that's not right",
    "not correct",
    "the date is wrong",
    "I took the book back",
    "ok but the landlord's name is Alice",
    "I know my rights",
])
def test_not_affirmations(msg):
    assert not is_short_affirmation(msg)


def test_done_signal_whole_message_only():
    assert is_done_signal("That's all, thanks", 2)
    assert is_done_signal("nothing else to add", 2)
    assert is_done_signal("just give me the answer please", 1)
    assert not is_done_signal("He abandoned the flat", 4)
    assert not is_done_signal("I have no more money for rent", 4)
    assert not is_done_signal("no", 2)
    assert is_done_signal("no", 3)


# ---------------------------------------------------------------------------
# CONFIRMING
# ---------------------------------------------------------------------------

def _confirming_case() -> dict:
    return CaseFile(
        hard_core=HardCore(jurisdiction="Ontario", user_goal="Stop eviction"),
        facts=[Fact(id="f1", text="Landlord served a notice", source=Source.user_said, status=Status.active)],
    ).model_dump()


@pytest.mark.asyncio
async def test_not_right_is_not_a_confirmation():
    session = await create_session()
    session["status"] = "CONFIRMING"
    session["case_file"] = _confirming_case()
    srv = CountingServiceClient()
    fake = FakeLLM(responses={
        # Even if the extractor misreads the intent, "not right" must never send the query
        "extractor": UnderstandOutput(intent=Intent.provides_info),
        "affect": AffectReading(),
        "gap_analyzer": GapAnalysis(gaps=[NOTICE_GAP]),
        "talker": "Ah, sorry about that. What did I get wrong?",
    })
    await handle_turn(session, "no, that's not right", llm=fake, service_client=srv, stream=False)
    assert session["status"] == "COLLECTING"
    assert srv.queries == []


@pytest.mark.asyncio
async def test_mixed_agree_and_fix_goes_back_to_collecting():
    session = await create_session()
    session["status"] = "CONFIRMING"
    session["case_file"] = _confirming_case()
    srv = CountingServiceClient()
    fake = FakeLLM(responses={
        "extractor": UnderstandOutput(intent=Intent.corrects),
        "affect": AffectReading(),
        "gap_analyzer": GapAnalysis(gaps=[NOTICE_GAP]),
        "talker": "Ah, got it. So when did it actually arrive?",
    })
    await handle_turn(session, "yeah mostly, but the date is wrong", llm=fake, service_client=srv, stream=False)
    assert session["status"] == "COLLECTING"
    assert srv.queries == []


@pytest.mark.asyncio
async def test_short_yes_confirms_even_if_intent_missed():
    session = await create_session()
    session["status"] = "CONFIRMING"
    session["case_file"] = _confirming_case()
    srv = CountingServiceClient()
    fake = FakeLLM(responses={
        "extractor": UnderstandOutput(intent=Intent.other),
        "affect": AffectReading(),
        "rewriter": ServiceQuery(query_text="Tenant in Ontario", documents=[{"name": "Lease", "state": "has"}]),
        "presenter": "Short version: he can't evict you without a tribunal order.",
        "faithfulness": FaithfulnessReport(ok=True),
    })
    await handle_turn(session, "Yes", llm=fake, service_client=srv, stream=False)
    assert session["status"] == "ANSWERED"
    assert len(srv.queries) == 1


# ---------------------------------------------------------------------------
# Mid-conversation questions
# ---------------------------------------------------------------------------

def _question_fake(jurisdiction: str | None, needs_facts: bool = True) -> FakeLLM:
    hard_core = {"user_goal": "Get back into the flat"}
    if jurisdiction:
        hard_core["jurisdiction"] = jurisdiction
    return FakeLLM(responses={
        "extractor": UnderstandOutput(
            intent=Intent.asks_legal_question,
            question="Can a landlord change the locks without a court order?",
            question_needs_case_facts=needs_facts,
            patch=CasePatch(set_hard_core=hard_core),
        ),
        "affect": AffectReading(distress=2),
        "gap_analyzer": GapAnalysis(gaps=[NOTICE_GAP]),
        "presenter": (
            "Generally, no: he has to go through the legal eviction process first. "
            "Did he give you any warning before changing the locks?"
        ),
        "faithfulness": FaithfulnessReport(ok=True),
        "talker": "Good question, and I'll get you a proper answer on it. Where is the flat?",
    })


@pytest.mark.asyncio
async def test_question_answered_now_when_location_known():
    session = await create_session()
    srv = CountingServiceClient()
    fake = _question_fake("Pune")
    reply, _ = await handle_turn(session, "He changed the locks in Pune. Can he even do that?", llm=fake, service_client=srv, stream=False)

    assert session["status"] == "COLLECTING"
    assert len(srv.queries) == 1
    assert srv.queries[0].kind == "quick_question"
    assert "Can a landlord change the locks without a court order?" in srv.queries[0].query_text
    assert "legal eviction process" in reply

    presenter_calls = [c for c in fake.call_history if c["prompt_name"] == "presenter"]
    assert presenter_calls and "Mode: quick" in presenter_calls[0]["system"]
    assert NOTICE_GAP.question_hint in presenter_calls[0]["system"]
    assert not [c for c in fake.call_history if c["prompt_name"] == "talker"]

    updated = await get_session(session["_id"])
    assert updated["asked_counts"]["Notice"] == 1
    assert updated["quick_answers"][0]["question"].startswith("Can a landlord")
    assert updated["disclaimer_shown"] is True


@pytest.mark.asyncio
async def test_general_question_answered_without_location():
    session = await create_session()
    srv = CountingServiceClient()
    await handle_turn(session, "What is a leave and licence agreement?", llm=_question_fake(None, needs_facts=False), service_client=srv, stream=False)
    assert len(srv.queries) == 1


@pytest.mark.asyncio
async def test_question_deferred_when_location_unknown():
    session = await create_session()
    srv = CountingServiceClient()
    fake = _question_fake(None, needs_facts=True)
    reply, _ = await handle_turn(session, "Can he even do that?", llm=fake, service_client=srv, stream=False)

    assert srv.queries == []
    talker_calls = [c for c in fake.call_history if c["prompt_name"] == "talker"]
    assert len(talker_calls) == 1
    system = talker_calls[0]["system"]
    assert 'They asked: "Can a landlord change the locks without a court order?"' in system
    assert "do not answer it yourself" in system


@pytest.mark.asyncio
async def test_question_deferred_when_service_fails():
    session = await create_session()
    srv = CountingServiceClient(fail=True)
    fake = _question_fake("Pune")
    await handle_turn(session, "Can he even do that?", llm=fake, service_client=srv, stream=False)

    assert len(srv.queries) == 1
    talker_calls = [c for c in fake.call_history if c["prompt_name"] == "talker"]
    assert len(talker_calls) == 1
    assert "They asked:" in talker_calls[0]["system"]


# ---------------------------------------------------------------------------
# Presenter
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_disclaimer_only_with_first_legal_answer():
    fake = FakeLLM(responses={
        "presenter": "Short version: you have options.",
        "faithfulness": FaithfulnessReport(ok=True),
    })
    session = {"_id": "s_disc", "case_file": {}}
    resp = ServiceResponse(status="answered", answer="You have options.")

    await present(session, resp, mode="quick", llm=fake)
    await present(session, resp, mode="initial", llm=fake)

    presenter_systems = [c["system"] for c in fake.call_history if c["prompt_name"] == "presenter"]
    assert DISCLAIMER_SHORT in presenter_systems[0]
    assert DISCLAIMER_SHORT not in presenter_systems[1]
    assert session["disclaimer_shown"] is True


@pytest.mark.asyncio
async def test_followup_is_guarded_with_honest_fallback():
    fake = FakeLLM(responses={
        "presenter": "Under Section 99 you can also sue for damages.",
        "faithfulness": FaithfulnessReport(ok=False, unsupported_claims=["sue for damages"]),
    })
    session = {"_id": "s_follow", "case_file": {}}
    resp = ServiceResponse(status="answered", answer="You can dispute the notice at the tribunal.")
    result = await present(session, resp, mode="followup", new_message="Can I sue him?", llm=fake)
    assert result == FOLLOWUP_FALLBACK_REPLY
    assert session["last_presenter_path"] == "fallback"
    assert "disclaimer_shown" not in session


def test_code_guard_allows_persons_own_facts():
    text = "You said the deposit was ₹50,000, so it's worth asking for it back in writing."
    source = "Ask for the deposit back in writing."
    assert check_code_guard(text, source)  # amount not in the source -> flagged
    assert check_code_guard(text, source, allowed_context="Facts:\n- Deposit paid was ₹50,000") == []


# ---------------------------------------------------------------------------
# Safety lead-in
# ---------------------------------------------------------------------------

def test_helpline_reply_has_lead_in_and_verbatim_message():
    cat = get_helplines_config()["categories"]["self_harm"]
    for lang_name, key in (("English", "en"), ("Hindi", "hi"), ("Hinglish", "hinglish")):
        reply = get_helpline_reply("self_harm", lang_name)
        assert reply.startswith(cat["lead_in"][key])
        assert reply.endswith(cat["messages"][key])


def test_unsupported_language_falls_back_to_english_helpline():
    cat = get_helplines_config()["categories"]["self_harm"]
    assert get_helpline_reply("self_harm", "Spanish").endswith(cat["messages"]["en"])
    assert "es" not in cat["messages"]
