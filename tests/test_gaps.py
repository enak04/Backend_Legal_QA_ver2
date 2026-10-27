"""
tests/test_gaps.py - Tests for gap analysis and pending_service_gaps injection.

Verifies:
- find_gaps calls LLM with gap_analyzer prompt on fast tier.
- Pending service gaps in session are prepended as high-impact gaps.
- already_asked list is formatted from session asked_counts.
"""

import pytest

from src.gaps import find_gaps
from src.llm import FakeLLM
from src.schemas import CaseFile, Gap, GapAnalysis, HardCore


@pytest.mark.asyncio
async def test_find_gaps_basic_call():
    """Verify that find_gaps renders the prompt and returns a validated GapAnalysis."""
    expected_gap = Gap(
        topic="notice_type",
        question_hint="What notice form did you receive?",
        why_it_matters="Determines notice validity",
        impact="high",
        example_answer="N12",
        can_skip=False,
    )
    fake_llm = FakeLLM(responses={
        "gap_analyzer": GapAnalysis(
            gaps=[expected_gap],
            legal_area_guess="Tenancy Law",
            blocking_for_service=True,
        )
    })

    case = CaseFile(
        hard_core=HardCore(jurisdiction="Ontario", user_goal="Stop eviction"),
    )
    session = {"_id": "sess_123", "asked_counts": {}}

    result = await find_gaps(session=session, case=case, llm=fake_llm)

    assert len(result.gaps) == 1
    assert result.gaps[0].topic == "notice_type"
    assert result.legal_area_guess == "Tenancy Law"
    assert result.blocking_for_service is True

    # Check LLM call history
    call = fake_llm.call_history[0]
    assert call["prompt_name"] == "gap_analyzer"
    assert call["tier"] == "fast"
    assert "Ontario" in call["system"]
    assert "None so far" in call["system"]


@pytest.mark.asyncio
async def test_find_gaps_injects_pending_service_gaps():
    """
    Verify that when session has pending_service_gaps, they are prepended
    as high-impact gaps and set blocking_for_service to True.
    """
    fake_llm = FakeLLM(responses={
        "gap_analyzer": GapAnalysis(
            gaps=[
                Gap(topic="lease_date", question_hint="Lease date", why_it_matters="History", impact="low", can_skip=True),
            ],
            legal_area_guess="Tenancy Law",
            blocking_for_service=False,
        )
    })

    case = CaseFile(hard_core=HardCore(jurisdiction="Ontario", user_goal="Prevent eviction"))
    session = {
        "_id": "sess_456",
        "pending_service_gaps": [
            "Exact date on the N12 notice",
            Gap(topic="landlord_declaration", question_hint="Did landlord provide declaration?", why_it_matters="Statutory requirement", impact="medium", can_skip=False),
        ],
    }

    result = await find_gaps(session=session, case=case, llm=fake_llm)

    # Injected gaps must be prepended and have impact="high"
    assert len(result.gaps) == 3
    assert result.gaps[0].topic == "Exact date on the N12 notice"
    assert result.gaps[0].impact == "high"
    assert result.gaps[1].topic == "landlord_declaration"
    assert result.gaps[1].impact == "high"
    assert result.gaps[2].topic == "lease_date"
    assert result.blocking_for_service is True


@pytest.mark.asyncio
async def test_find_gaps_includes_already_asked_topics():
    """Verify that topics in asked_counts are rendered under already_asked in the prompt."""
    fake_llm = FakeLLM(responses={
        "gap_analyzer": GapAnalysis(gaps=[]),
    })

    case = CaseFile(hard_core=HardCore(jurisdiction="British Columbia", user_goal="Rent dispute"))
    session = {
        "_id": "sess_789",
        "asked_counts": {
            "monthly_rent": 1,
            "deposit_paid": 2,
        },
    }

    await find_gaps(session=session, case=case, llm=fake_llm)

    call = fake_llm.call_history[0]
    assert "- monthly_rent" in call["system"]
    assert "- deposit_paid" in call["system"]
