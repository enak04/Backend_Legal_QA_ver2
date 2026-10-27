"""
tests/test_case_ops.py - Comprehensive tests for case operations and mutations.

Verifies:
- add fact
- add with fake evidence (must be downgraded)
- update an existing fact (old marked 'corrected', new added with note)
- retract (status marked 'retracted')
- unknown id is skipped and reported
- dedupe by normalized text
- version bump
- nothing is ever deleted
- patch with a bad hard_core key is rejected
"""

import pytest

from src.case_ops import apply_patch, render_case_json_for_prompts, render_case_summary
from src.schemas import (
    CaseFile,
    CasePatch,
    Fact,
    HardCore,
    ItemRetract,
    ItemUpdate,
    NewFact,
    NewTimelineEvent,
    Source,
    Status,
    TimelineEvent,
)


def test_add_fact():
    """Verify adding a new fact with valid evidence."""
    case = CaseFile()
    patch = CasePatch(
        add_facts=[
            NewFact(text="I paid $1000 rent", source=Source.user_said, evidence="$1000 rent", confidence=1.0)
        ]
    )
    user_msg = "Last week I paid $1000 rent to my landlord."

    updated_case, report = apply_patch(case, patch, user_msg, turn=1)

    assert len(updated_case.facts) == 1
    f = updated_case.facts[0]
    assert f.id == "f1"
    assert f.text == "I paid $1000 rent"
    assert f.source == Source.user_said
    assert f.confidence == 1.0
    assert f.status == Status.active
    assert f.turn == 1
    assert updated_case.version == 1
    assert "f1" in report["added_facts"]


def test_add_with_fake_evidence_downgrades():
    """Verify that absent evidence quote causes downgrade to inferred and caps confidence at 0.5."""
    case = CaseFile()
    patch = CasePatch(
        add_facts=[
            NewFact(text="Landlord locked the door", source=Source.user_said, evidence="locked the door", confidence=0.95)
        ]
    )
    user_msg = "My landlord called me yesterday on the phone."

    updated_case, report = apply_patch(case, patch, user_msg, turn=1)

    assert len(updated_case.facts) == 1
    f = updated_case.facts[0]
    assert f.source == Source.inferred
    assert f.confidence == 0.5
    assert len(report["downgraded_evidence"]) == 1


def test_update_existing_fact():
    """
    Verify that updating a fact marks the old one as 'corrected'
    and adds a new active fact linked by a note.
    """
    case = CaseFile(
        facts=[
            Fact(id="f1", text="Rent was $1000", source=Source.user_said, status=Status.active, turn=1)
        ],
        version=1,
    )
    patch = CasePatch(
        update=[
            ItemUpdate(
                kind="fact",
                id="f1",
                changes={"text": "Rent was actually $1200"},
                reason="User corrected the figure",
            )
        ]
    )

    updated_case, report = apply_patch(case, patch, "Actually rent was $1200", turn=2)

    # Total facts must now be 2 (nothing deleted)
    assert len(updated_case.facts) == 2
    old_f = updated_case.facts[0]
    new_f = updated_case.facts[1]

    assert old_f.id == "f1"
    assert old_f.status == Status.corrected

    assert new_f.id == "f2"
    assert new_f.text == "Rent was actually $1200"
    assert new_f.status == Status.active
    assert "f1" in new_f.note
    assert "User corrected the figure" in new_f.note
    assert len(report["updated_items"]) == 1


def test_retract():
    """Verify that retracting an item sets status to 'retracted' without deleting it."""
    case = CaseFile(
        facts=[
            Fact(id="f1", text="Deposit was paid in cash", status=Status.active)
        ],
        version=1,
    )
    patch = CasePatch(
        retract=[
            ItemRetract(kind="fact", id="f1", reason="I was mistaken about the cash payment")
        ]
    )

    updated_case, report = apply_patch(case, patch, "I was mistaken", turn=2)

    assert len(updated_case.facts) == 1
    f = updated_case.facts[0]
    assert f.id == "f1"
    assert f.status == Status.retracted
    assert len(report["retracted_items"]) == 1


def test_unknown_id_is_skipped():
    """Verify that unknown IDs in update/retract are skipped safely and reported."""
    case = CaseFile(
        facts=[Fact(id="f1", text="Lease signed", status=Status.active)]
    )
    patch = CasePatch(
        update=[ItemUpdate(kind="fact", id="f999", changes={"text": "invalid"}, reason="test")],
        retract=[ItemRetract(kind="fact", id="f888", reason="test")],
    )

    updated_case, report = apply_patch(case, patch, "hello", turn=2)

    assert "f999" in report["skipped_unknown_ids"]
    assert "f888" in report["skipped_unknown_ids"]
    assert len(updated_case.facts) == 1
    assert updated_case.facts[0].status == Status.active


def test_dedupe_normalized_text():
    """Verify that duplicates with different whitespace/casing are not added twice."""
    case = CaseFile(
        facts=[Fact(id="f1", text="Received an eviction notice", status=Status.active)]
    )
    patch = CasePatch(
        add_facts=[
            NewFact(text="  received an EVICTION notice  ", source=Source.user_said, evidence="eviction notice")
        ]
    )

    updated_case, report = apply_patch(case, patch, "eviction notice", turn=2)

    assert len(updated_case.facts) == 1
    assert len(report["skipped_duplicates"]) == 1


def test_version_bump():
    """Verify that each apply_patch increments version by 1."""
    case = CaseFile(version=0)
    case, _ = apply_patch(case, CasePatch(), "msg 1", turn=1)
    assert case.version == 1
    case, _ = apply_patch(case, CasePatch(), "msg 2", turn=2)
    assert case.version == 2


def test_nothing_is_ever_deleted():
    """Verify that applying multiple updates and retracts never reduces collection counts."""
    case = CaseFile(
        facts=[Fact(id="f1", text="Fact 1", status=Status.active)],
        timeline=[TimelineEvent(id="t1", when_text="2026", description="Event 1", source=Source.user_said)],
    )

    patch = CasePatch(
        retract=[
            ItemRetract(kind="fact", id="f1", reason="Retracted"),
            ItemRetract(kind="timeline", id="t1", reason="Retracted"),
        ]
    )
    updated, _ = apply_patch(case, patch, "no", turn=1)

    assert len(updated.facts) == 1
    assert updated.facts[0].status == Status.retracted
    assert len(updated.timeline) == 1
    assert updated.timeline[0].status == Status.retracted


def test_patch_with_bad_hard_core_key_rejected():
    """Verify that a patch containing an unknown hard_core key is rejected with ValueError."""
    with pytest.raises(ValueError, match="Invalid hard_core key"):
        CasePatch(set_hard_core={"fake_illegal_field": "some_value"})


def test_render_case_summary_only_shows_active():
    """Verify that render_case_summary only presents ACTIVE items."""
    case = CaseFile(
        hard_core=HardCore(jurisdiction="New York", user_goal="Stop eviction"),
        facts=[
            Fact(id="f1", text="Old corrected fact", status=Status.corrected),
            Fact(id="f2", text="Retracted fact", status=Status.retracted),
            Fact(id="f3", text="Active valid fact", status=Status.active),
        ],
    )
    summary = render_case_summary(case)

    assert "New York" in summary
    assert "Stop eviction" in summary
    assert "Active valid fact" in summary
    assert "Old corrected fact" not in summary
    assert "Retracted fact" not in summary
