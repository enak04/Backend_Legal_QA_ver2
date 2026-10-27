"""
tests/test_confirm_service_presenter.py - Integration and unit tests for the second half of conversation.

Verifies:
1. PII Redaction in redact.py (phone, email, long digit sequences).
2. Presenter code guard catches fake section numbers and regenerates.
3. Presenter fallback template path when regeneration fails.
4. Correction during CONFIRMING moves back to COLLECTING.
5. needs_more_info loop sets pending_service_gaps and returns to COLLECTING.
6. ANSWERED follow-up answers from stored answer only.
7. ANSWERED with new facts returns to CONFIRMING with updated recap.
8. Full start-to-finish conversation: questions -> recap -> confirmation -> answer -> follow-up -> correction -> updated recap.
"""

import pytest

from src.db import create_session, get_session
from src.llm import FakeLLM
from src.pipeline import handle_turn
from src.presenter import check_code_guard, fallback_template, present
from src.redact import redact_service_query, redact_text
from src.schemas import (
    AffectReading,
    CaseFile,
    CasePatch,
    Document,
    DocumentState,
    Fact,
    FaithfulnessReport,
    Gap,
    GapAnalysis,
    HardCore,
    Intent,
    NewDocument,
    NewFact,
    ServiceQuery,
    ServiceQueryFact,
    ServiceResponse,
    Source,
    Status,
    UnderstandOutput,
)

from src.service_client import ServiceClient


def test_pii_redaction():
    """Verify redact_text and redact_service_query strip phone numbers, emails, and digit sequences."""
    text = "Call me at +1 416-555-0199 or email user.test@example.com, my SSN is 123456789."
    cleaned = redact_text(text)
    assert "416-555-0199" not in cleaned
    assert "user.test@example.com" not in cleaned
    assert "123456789" not in cleaned
    assert "[REDACTED_PHONE]" in cleaned
    assert "[REDACTED_EMAIL]" in cleaned
    assert "[REDACTED_ID]" in cleaned

    query = ServiceQuery(
        query_text="Tenant at 416-555-0199 with email test@domain.com account 987654321.",
        facts=[ServiceQueryFact(text="Phone 416-555-0199", certainty="stated")],
        documents=[{"name": "Lease 123456789", "state": "has"}],
        questions_to_answer=["Can I contact 416-555-0199?"],
    )
    redacted_query = redact_service_query(query)
    assert "416-555-0199" not in redacted_query.query_text
    assert "416-555-0199" not in redacted_query.facts[0].text
    assert "123456789" not in redacted_query.documents[0]["name"]
    assert "416-555-0199" not in redacted_query.questions_to_answer[0]


@pytest.mark.asyncio
async def test_presenter_catches_fake_section_and_regenerates():
    """
    Code guard detects fake section number in Draft 1; regenerates; Draft 2 passes.
    """
    source_answer = "Tenants have 14 days to dispute an eviction notice with the tribunal."
    service_resp = ServiceResponse(
        status="answered",
        answer=source_answer,
        sections={"summary": source_answer},
    )

    # Draft 1 hallucinates Section 48(1)
    # Draft 2 uses only source content
    draft_1 = "Under Section 48(1) of the Act, tenants have 14 days to dispute."
    draft_2 = "As noted, you have 14 days to file your dispute with the tribunal."

    fake_llm = FakeLLM(responses={
        "presenter": draft_1,  # First call returns draft_1
        "faithfulness": FaithfulnessReport(ok=True),
    })

    # Override responses dynamically for sequential calls to presenter
    presenter_calls = 0

    async def custom_text(prompt_name, system, messages, tier, session_id=None, temperature=0.3):
        nonlocal presenter_calls
        if prompt_name == "presenter":
            presenter_calls += 1
            if presenter_calls == 1:
                return draft_1
            return draft_2
        return "text"

    fake_llm.text = custom_text

    session = {"_id": "test_sess", "case_file": {}}
    result = await present(session, service_resp, mode="initial", llm=fake_llm)

    assert result == draft_2
    assert session["last_presenter_path"] == "regenerated"
    assert presenter_calls == 2


@pytest.mark.asyncio
async def test_presenter_fallback_template_path():
    """
    When regeneration still contains ungrounded legal elements, falls back to plain template.
    """
    source_answer = "The notice period must be at least 60 days."
    service_resp = ServiceResponse(
        status="answered",
        answer=source_answer,
        sections={"summary": source_answer, "next_steps": "File dispute.", "caveats": "Strict deadline."},
    )

    # Both drafts contain hallucinated Section 99
    bad_draft = "Under Section 99 and Roe v. Wade, notice must be 60 days."
    fake_llm = FakeLLM(responses={
        "presenter": bad_draft,
        "faithfulness": FaithfulnessReport(ok=False, unsupported_claims=["Section 99"]),
    })

    session = {"_id": "test_sess", "case_file": {}}
    result = await present(session, service_resp, mode="initial", llm=fake_llm)

    assert session["last_presenter_path"] == "fallback"
    assert "Summary:" in result
    assert "File dispute." in result
    assert "Strict deadline." in result
    assert "IMPORTANT NOTICE" in result or "disclaimer" in result.lower()


@pytest.mark.asyncio
async def test_confirming_correction_moves_to_collecting():
    """
    When in CONFIRMING state, if user provides a correction (Intent.corrects),
    the session applies the patch and moves back to COLLECTING.
    """
    session = await create_session()
    session["status"] = "CONFIRMING"
    session["case_file"] = CaseFile(
        hard_core=HardCore(jurisdiction="Ontario", user_goal="Prevent eviction"),
        facts=[Fact(id="f1", text="Landlord is Bob", source=Source.user_said, confidence=1.0, status=Status.active)],
    ).model_dump()

    fake_llm = FakeLLM(responses={
        "extractor": UnderstandOutput(
            intent=Intent.corrects,
            patch=CasePatch(add_facts=[NewFact(text="Landlord is Alice", source="user_said", evidence="Alice")]),
        ),
        "affect": AffectReading(),
        "gap_analyzer": GapAnalysis(
            gaps=[Gap(topic="Notice Date", question_hint="When served", why_it_matters="Deadlines", impact="high", can_skip=False)],
        ),
        "talker": "Got it, I updated the landlord to Alice. When did you receive the notice?",
    })

    reply, decision = await handle_turn(
        session=session,
        user_message="Actually my landlord is Alice, not Bob.",
        llm=fake_llm,
        stream=False,
    )

    assert session["status"] == "COLLECTING"
    assert "Alice" in reply


@pytest.mark.asyncio
async def test_needs_more_info_loop():
    """
    In CONFIRMING, user confirms. Service has mock mode and query has no documents.
    Service returns needs_more_info. Chatbot transitions back to COLLECTING
    with pending_service_gaps and talker asks for them naturally.
    """
    session = await create_session()
    session["status"] = "CONFIRMING"
    # Case with NO documents
    session["case_file"] = CaseFile(
        hard_core=HardCore(jurisdiction="Ontario", user_goal="Stop eviction"),
        facts=[Fact(id="f1", text="User was told to leave", source=Source.user_said, confidence=1.0, status=Status.active)],
    ).model_dump()

    fake_llm = FakeLLM(responses={
        "extractor": UnderstandOutput(intent=Intent.confirms),
        "affect": AffectReading(),
        "rewriter": ServiceQuery(
            query_text="Tenant in Ontario faces eviction.",
            jurisdiction="Ontario",
            user_goal="Stop eviction",
            documents=[],  # Empty documents triggers mock needs_more_info
        ),
        "gap_analyzer": GapAnalysis(gaps=[]),
        "talker": "Before our legal team can give advice, do you have a written rental agreement?",
    })

    srv_client = ServiceClient(use_mock=True)

    reply, decision = await handle_turn(
        session=session,
        user_message="Yes, that summary is completely correct.",
        llm=fake_llm,
        service_client=srv_client,
        stream=False,
    )

    assert session["status"] == "COLLECTING"
    assert session["pending_service_gaps"] == ["Do you have a written rental agreement?"]
    assert "written rental agreement" in reply


@pytest.mark.asyncio
async def test_answered_followup_question():
    """
    In ANSWERED state, user asks a follow-up question.
    Presenter is called in followup mode using stored answer only.
    Session status remains ANSWERED.
    """
    session = await create_session()
    session["status"] = "ANSWERED"
    session["last_service_answer"] = ServiceResponse(
        status="answered",
        answer="The tenancy tribunal is the Landlord and Tenant Board (LTB).",
    ).model_dump()
    session["case_file"] = CaseFile(
        hard_core=HardCore(jurisdiction="Ontario", user_goal="Dispute eviction"),
    ).model_dump()

    fake_llm = FakeLLM(responses={
        "extractor": UnderstandOutput(intent=Intent.asks_legal_question, patch=CasePatch()),
        "affect": AffectReading(),
        "presenter": "The tribunal referenced in the advice is the Landlord and Tenant Board.",
        "faithfulness": FaithfulnessReport(ok=True),
    })

    reply, decision = await handle_turn(
        session=session,
        user_message="What tribunal is that?",
        llm=fake_llm,
        stream=False,
    )

    assert session["status"] == "ANSWERED"
    assert "Landlord and Tenant Board" in reply


@pytest.mark.asyncio
async def test_answered_with_new_facts_returns_to_confirming():
    """
    In ANSWERED state, if user provides new case facts, the patch is applied,
    status moves back to CONFIRMING, and an updated recap is shown.
    """
    session = await create_session()
    session["status"] = "ANSWERED"
    session["last_service_answer"] = ServiceResponse(
        status="answered",
        answer="You can dispute the N12 notice.",
    ).model_dump()
    session["case_file"] = CaseFile(
        hard_core=HardCore(jurisdiction="Ontario", user_goal="Dispute eviction"),
        documents=[Document(id="d1", name="N12 Notice", state=DocumentState.has, status=Status.active)],
    ).model_dump()


    fake_llm = FakeLLM(responses={
        "extractor": UnderstandOutput(
            intent=Intent.provides_info,
            patch=CasePatch(add_documents=[NewDocument(name="Text messages from landlord", state="has")]),
        ),
        "affect": AffectReading(),
        "confirm_summary": "Here is your updated summary with your text messages included. Does this look accurate?",
    })

    reply, decision = await handle_turn(
        session=session,
        user_message="I also have text messages where the landlord admitted they want higher rent.",
        llm=fake_llm,
        stream=False,
    )

    assert session["status"] == "CONFIRMING"
    assert "updated" in reply or "summary" in reply


@pytest.mark.asyncio
async def test_full_happy_path_start_to_finish():
    """
    Verifies full conversation flow from start to finish:
    1. COLLECTING -> User provides details
    2. COLLECTING -> Ready for confirmation -> Recap shown, status CONFIRMING
    3. CONFIRMING -> User confirms -> Service called -> Answer presented, status ANSWERED
    4. ANSWERED -> User asks follow-up -> Answered from stored answer
    5. ANSWERED -> User provides new facts -> Transitions back to CONFIRMING with updated recap!
    """
    session = await create_session()
    srv_client = ServiceClient(use_mock=True)

    # 1. Turn 1: COLLECTING (Missing details)
    fake1 = FakeLLM(responses={
        "extractor": UnderstandOutput(
            intent=Intent.provides_info,
            patch=CasePatch(
                set_hard_core={"jurisdiction": "Ontario", "user_goal": "Stop eviction"},
                add_documents=[NewDocument(name="Tenancy Agreement", state="has")],
            ),
        ),
        "affect": AffectReading(),
        "gap_analyzer": GapAnalysis(
            gaps=[Gap(topic="Notice Date", question_hint="Notice date", why_it_matters="Deadlines", impact="high", can_skip=False)],
        ),
        "talker": "When did you receive the eviction notice?",
    })
    reply1, _ = await handle_turn(session, "My landlord in Ontario wants to evict me.", llm=fake1, service_client=srv_client, stream=False)
    assert session["status"] == "COLLECTING"
    assert "When did you receive" in reply1

    # 2. Turn 2: COLLECTING -> Ready! Transitions to CONFIRMING with recap
    fake2 = FakeLLM(responses={
        "extractor": UnderstandOutput(
            intent=Intent.provides_info,
            patch=CasePatch(add_facts=[NewFact(text="Notice received on Oct 1", source="user_said", evidence="Oct 1")]),
        ),
        "affect": AffectReading(),
        "gap_analyzer": GapAnalysis(gaps=[]),
        "confirm_summary": "Thank you. You live in Ontario and received an eviction notice on Oct 1. You want to stop it. Is this correct?",
    })
    reply2, _ = await handle_turn(session, "I got it on Oct 1.", llm=fake2, service_client=srv_client, stream=False)
    assert session["status"] == "CONFIRMING"
    assert "Thank you" in reply2

    # 3. Turn 3: CONFIRMING -> User confirms -> Service called -> Status ANSWERED, Answer presented
    fake3 = FakeLLM(responses={
        "extractor": UnderstandOutput(intent=Intent.confirms),
        "affect": AffectReading(),
        "rewriter": ServiceQuery(
            query_text="Tenant in Ontario received eviction notice on Oct 1. Has tenancy agreement.",
            jurisdiction="Ontario",
            user_goal="Stop eviction",
            documents=[{"name": "Tenancy Agreement", "state": "has"}],
        ),
        "presenter": "Based on our legal service: A landlord cannot evict without a tribunal order. What you can do next: 1. File a dispute.\n\nIMPORTANT NOTICE: This is not legal advice.",
        "faithfulness": FaithfulnessReport(ok=True),
    })
    reply3, _ = await handle_turn(session, "Yes, that is correct.", llm=fake3, service_client=srv_client, stream=False)
    assert session["status"] == "ANSWERED"
    assert "tribunal order" in reply3
    assert session["last_service_answer"] is not None

    # 4. Turn 4: ANSWERED -> User asks follow-up -> Answered in followup mode
    fake4 = FakeLLM(responses={
        "extractor": UnderstandOutput(intent=Intent.asks_legal_question, patch=CasePatch()),
        "affect": AffectReading(),
        "presenter": "The service indicates that you must file a dispute before the stated termination date.",
        "faithfulness": FaithfulnessReport(ok=True),
    })
    reply4, _ = await handle_turn(session, "How long do I have to file?", llm=fake4, service_client=srv_client, stream=False)
    assert session["status"] == "ANSWERED"
    assert "stated termination date" in reply4

    # 5. Turn 5: ANSWERED -> User provides new facts -> Returns to CONFIRMING with updated recap!
    fake5 = FakeLLM(responses={
        "extractor": UnderstandOutput(
            intent=Intent.provides_info,
            patch=CasePatch(add_facts=[NewFact(text="Landlord accepted rent yesterday", source="user_said", evidence="accepted rent")]),
        ),
        "affect": AffectReading(),
        "confirm_summary": "I have updated your case: the landlord accepted rent yesterday. Shall I submit this new detail to the legal team?",
    })
    reply5, _ = await handle_turn(session, "Also, my landlord accepted rent yesterday.", llm=fake5, service_client=srv_client, stream=False)
    assert session["status"] == "CONFIRMING"
    assert "accepted rent" in reply5
