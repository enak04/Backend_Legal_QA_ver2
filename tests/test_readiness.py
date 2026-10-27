"""
tests/test_readiness.py - Tests for readiness evaluation and gap tracking logic.

Verifies:
- Readiness truth table:
  1. Missing jurisdiction (with/without goal, with/without impatient user) -> NOT ready.
  2. Impatient user without hard core -> NOT ready.
  3. Impatient user with hard core -> READY.
  4. Only skippable gaps (no active high-impact gaps) -> READY.
  5. MAX_COLLECTING_TURNS reached with hard core -> READY.
  6. High-impact non-skippable gap -> NOT ready.
  7. High-impact skippable gap asked < 2 times -> NOT ready.
  8. High-impact skippable gap asked >= 2 times -> READY (ignored).
- asked_counts logic:
  - Topic asked twice records an assumption ('Unknown: <topic>') in case.assumptions.
  - Topic asked twice is filtered out so it is never asked a third time.
"""

import pytest

from src.config import Settings, get_settings
from src.readiness import filter_gaps_to_ask, is_ready, track_asked_gap
from src.schemas import CaseFile, Gap, GapAnalysis, HardCore


@pytest.fixture
def dummy_settings():
    s = get_settings()
    return Settings(
        provider=s.provider,
        llm_api_key="sk-test",
        fast_model=s.fast_model,
        smart_model=s.smart_model,
        mongodb_uri="mongodb://localhost:27017",
        db_name="test_db",
        use_mock_service=True,
        service_url=s.service_url,
        service_api_key="test_key",
        max_questions_per_turn=1,
        history_window=10,
        max_collecting_turns=5,
        session_ttl_days=30,
        debug=False,
    )



def test_missing_jurisdiction_never_ready(dummy_settings):
    """Missing jurisdiction must fail readiness even if user wants answer now or turns are exhausted."""
    case = CaseFile(hard_core=HardCore(jurisdiction=None, user_goal="Prevent eviction"))
    gaps = GapAnalysis(gaps=[])

    ready, reason = is_ready(
        case=case,
        gap_analysis=gaps,
        wants_answer_now=False,
        turn_count=1,
        settings=dummy_settings,
    )
    assert ready is False
    assert "jurisdiction" in reason

    # Even with wants_answer_now=True
    ready_impatient, reason_impatient = is_ready(
        case=case,
        gap_analysis=gaps,
        wants_answer_now=True,
        turn_count=1,
        settings=dummy_settings,
    )
    assert ready_impatient is False
    assert "jurisdiction" in reason_impatient

    # Even at MAX_COLLECTING_TURNS
    ready_max_turns, _ = is_ready(
        case=case,
        gap_analysis=gaps,
        wants_answer_now=False,
        turn_count=10,
        settings=dummy_settings,
    )
    assert ready_max_turns is False


def test_missing_user_goal_never_ready(dummy_settings):
    """Missing user_goal must fail readiness even if user wants answer now."""
    case = CaseFile(hard_core=HardCore(jurisdiction="California", user_goal=None))
    gaps = GapAnalysis(gaps=[])

    ready, reason = is_ready(
        case=case,
        gap_analysis=gaps,
        wants_answer_now=True,
        turn_count=1,
        settings=dummy_settings,
    )
    assert ready is False
    assert "user_goal" in reason


def test_impatient_user_with_hard_core_is_ready(dummy_settings):
    """If user wants answer now and hard core details exist, chatbot is ready."""
    case = CaseFile(hard_core=HardCore(jurisdiction="California", user_goal="Get deposit back"))
    # Gaps exist, but user is impatient
    gaps = GapAnalysis(gaps=[
        Gap(topic="lease document", question_hint="Provide lease", why_it_matters="Helps verify terms", impact="high", example_answer="Here is copy", can_skip=False)
    ])

    ready, reason = is_ready(
        case=case,
        gap_analysis=gaps,
        wants_answer_now=True,
        turn_count=1,
        settings=dummy_settings,
    )
    assert ready is True
    assert "immediately" in reason


def test_only_skippable_gaps_is_ready(dummy_settings):
    """If there are no high-impact gaps, the chatbot is ready."""
    case = CaseFile(hard_core=HardCore(jurisdiction="Ontario", user_goal="Stop unlawful rent increase"))
    # Only medium / low skippable gaps
    gaps = GapAnalysis(gaps=[
        Gap(topic="lease start date", question_hint="When did lease start", why_it_matters="Establishes history", impact="medium", example_answer="May 2022", can_skip=True),
        Gap(topic="landlord name", question_hint="Landlord name", why_it_matters="Identifies party", impact="low", example_answer="ABC Holdings", can_skip=True),
    ])

    ready, reason = is_ready(
        case=case,
        gap_analysis=gaps,
        wants_answer_now=False,
        turn_count=2,
        settings=dummy_settings,
    )
    assert ready is True
    assert "No high-impact gaps remaining" in reason


def test_max_collecting_turns_reached(dummy_settings):
    """When MAX_COLLECTING_TURNS is reached with hard core present, offers to proceed."""
    case = CaseFile(hard_core=HardCore(jurisdiction="New York", user_goal="Fix heating"))
    gaps = GapAnalysis(gaps=[
        Gap(topic="notice date", question_hint="Notice date", why_it_matters="Check timeline", impact="high", example_answer="Yesterday", can_skip=False)
    ])

    ready, reason = is_ready(
        case=case,
        gap_analysis=gaps,
        wants_answer_now=False,
        turn_count=5,  # == max_collecting_turns
        settings=dummy_settings,
    )
    assert ready is True
    assert "I think I have enough, shall I summarize?" in reason


def test_high_impact_gap_blocks_readiness(dummy_settings):
    """A high-impact gap blocks readiness if it hasn't been asked twice and cannot be skipped."""
    case = CaseFile(hard_core=HardCore(jurisdiction="Texas", user_goal="Dispute repair deduction"))
    gaps = GapAnalysis(gaps=[
        Gap(topic="move_out_notice", question_hint="Did you give written notice", why_it_matters="Required for deposit claim", impact="high", example_answer="Yes on June 1", can_skip=False)
    ])

    ready, reason = is_ready(
        case=case,
        gap_analysis=gaps,
        wants_answer_now=False,
        turn_count=2,
        settings=dummy_settings,
        asked_counts={"move_out_notice": 1},
    )
    assert ready is False
    assert "High-impact gap remaining: move_out_notice" in reason


def test_high_impact_skippable_gap_asked_twice_is_ignored(dummy_settings):
    """
    A high-impact gap with can_skip=True that was asked twice without answer
    is ignored and recorded as an assumption ('Unknown: <topic>').
    """
    case = CaseFile(hard_core=HardCore(jurisdiction="Texas", user_goal="Dispute repair deduction"))
    gap = Gap(
        topic="move_out_inspection_report",
        question_hint="Do you have the move-out inspection report?",
        why_it_matters="Evidence of apartment condition",
        impact="high",
        example_answer="Yes, signed copy",
        can_skip=True,
    )
    gaps = GapAnalysis(gaps=[gap])
    session = {"asked_counts": {"move_out_inspection_report": 2}}

    ready, reason = is_ready(
        case=case,
        gap_analysis=gaps,
        wants_answer_now=False,
        turn_count=2,
        settings=dummy_settings,
        session=session,
    )
    assert ready is True
    assert "No high-impact gaps remaining" in reason
    assert "Unknown: move_out_inspection_report" in case.assumptions


def test_asked_counts_and_stop_asking(dummy_settings):
    """
    Tests tracking asked_counts and filtering out topics asked twice so they are never asked a 3rd time.
    """
    case = CaseFile(hard_core=HardCore(jurisdiction="Ontario", user_goal="Dispute eviction"))
    gap1 = Gap(topic="Notice Date", question_hint="When served", why_it_matters="Timeline check", impact="high", example_answer="Last week", can_skip=True)
    gap2 = Gap(topic="Rent Receipt", question_hint="Rent receipts", why_it_matters="Proof of payment", impact="medium", example_answer="Bank statement", can_skip=True)

    session = {"asked_counts": {}}

    # Turn 1: Select gap to ask
    to_ask = filter_gaps_to_ask(session=session, case=case, gaps=[gap1, gap2], max_questions=1)
    assert len(to_ask) == 1
    assert to_ask[0].topic == "Notice Date"
    # Track asked count
    cnt1 = track_asked_gap(session, to_ask[0].topic)
    assert cnt1 == 1

    # Turn 2: User didn't answer Notice Date, still top gap
    to_ask_2 = filter_gaps_to_ask(session=session, case=case, gaps=[gap1, gap2], max_questions=1)
    assert len(to_ask_2) == 1
    assert to_ask_2[0].topic == "Notice Date"
    cnt2 = track_asked_gap(session, to_ask_2[0].topic)
    assert cnt2 == 2

    # Turn 3: Notice Date asked twice! filter_gaps_to_ask must skip it and record assumption
    to_ask_3 = filter_gaps_to_ask(session=session, case=case, gaps=[gap1, gap2], max_questions=1)
    assert len(to_ask_3) == 1
    # Notice Date was skipped, so gap2 ("Rent Receipt") is asked!
    assert to_ask_3[0].topic == "Rent Receipt"
    # Notice Date recorded as assumption
    assert "Unknown: Notice Date" in case.assumptions
