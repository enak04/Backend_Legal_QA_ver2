"""
src/case_ops.py - Case file operations and mutations.

Applies validated patches to the CaseFile:
- Assigns deterministic sequential IDs (f1, t1, p1, d1, dl1...).
- Never deletes items: retractions mark status='retracted', updates mark previous fact='corrected'
  and add a new active fact linked by a note.
- Deduplicates new items by normalized text.
- Anti-hallucination guard: if source='user_said' but evidence quote is absent from the user message,
  downgrades source to 'inferred' and caps confidence at 0.5.
- Bumps case version.
- Validates output with Pydantic.
- Renders compact active summaries and prompt-ready JSON.
"""

import json
import re
from typing import Any

from src.schemas import (
    CaseFile,
    CasePatch,
    Deadline,
    Document,
    Fact,
    HardCore,
    Person,
    Source,
    Status,
    TimelineEvent,
    Urgency,
)


def _next_id(items: list[Any], prefix: str) -> str:
    """Finds the next sequential ID for items with given prefix (e.g. f1, f2...)."""
    max_num = 0
    for it in items:
        item_id = getattr(it, "id", "")
        if item_id.startswith(prefix):
            suffix = item_id[len(prefix):]
            if suffix.isdigit():
                max_num = max(max_num, int(suffix))
    return f"{prefix}{max_num + 1}"


def _normalize_text(text: str) -> str:
    """Normalizes string for deduplication."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _verify_anti_hallucination(
    source: Source,
    evidence: str,
    confidence: float,
    user_message: str,
) -> tuple[Source, float, bool]:
    """
    Checks if evidence quote appears (case-insensitive) in the actual user message.
    If source is user_said and evidence is missing/not found, downgrades to inferred and caps confidence at 0.5.
    """
    if source == Source.user_said:
        ev = evidence.strip().lower()
        if not ev or ev not in user_message.lower():
            return Source.inferred, min(confidence, 0.5), True
    return source, confidence, False


def apply_patch(
    case: CaseFile | dict,
    patch: CasePatch | dict,
    user_message: str,
    turn: int = 1,
) -> tuple[CaseFile, dict]:
    """
    Applies a CasePatch to an existing CaseFile and returns (new_case, change_report).
    """
    # Ensure typed models
    if isinstance(case, dict):
        current_case = CaseFile.model_validate(case)
    else:
        current_case = case.model_copy(deep=True)

    if isinstance(patch, dict):
        patch_obj = CasePatch.model_validate(patch)
    else:
        patch_obj = patch

    change_report: dict[str, list[Any]] = {
        "added_facts": [],
        "added_timeline": [],
        "added_people": [],
        "added_documents": [],
        "added_deadlines": [],
        "updated_items": [],
        "retracted_items": [],
        "skipped_unknown_ids": [],
        "skipped_duplicates": [],
        "downgraded_evidence": [],
    }

    # 1. Update HardCore fields
    if patch_obj.set_hard_core:
        # Validate that all keys are legitimate HardCore fields
        valid_hardcore_keys = set(HardCore.model_fields.keys())
        bad_keys = set(patch_obj.set_hard_core.keys()) - valid_hardcore_keys
        if bad_keys:
            raise ValueError(f"Invalid hard_core key(s): {bad_keys}")

        for k, v in patch_obj.set_hard_core.items():
            if k == "urgency":
                if isinstance(v, dict):
                    current_case.hard_core.urgency = Urgency.model_validate(v)
                elif isinstance(v, Urgency):
                    current_case.hard_core.urgency = v
            else:
                setattr(current_case.hard_core, k, v)

    # 2. Add Facts (with dedupe & anti-hallucination check)
    existing_fact_texts = {_normalize_text(f.text) for f in current_case.facts if f.status == Status.active}
    for nf in patch_obj.add_facts:
        norm = _normalize_text(nf.text)
        if norm in existing_fact_texts:
            change_report["skipped_duplicates"].append(nf.text)
            continue

        src, conf, downgraded = _verify_anti_hallucination(nf.source, nf.evidence, nf.confidence, user_message)
        if downgraded:
            change_report["downgraded_evidence"].append(nf.text)

        new_id = _next_id(current_case.facts, "f")
        fact_item = Fact(
            id=new_id,
            text=nf.text,
            source=src,
            confidence=conf,
            status=Status.active,
            evidence=nf.evidence,
            turn=turn,
        )
        current_case.facts.append(fact_item)
        existing_fact_texts.add(norm)
        change_report["added_facts"].append(new_id)

    # 3. Add Timeline Events
    existing_events = {_normalize_text(t.description) for t in current_case.timeline if t.status == Status.active}
    for ne in patch_obj.add_timeline:
        norm = _normalize_text(ne.description)
        if norm in existing_events:
            change_report["skipped_duplicates"].append(ne.description)
            continue

        src, conf, _ = _verify_anti_hallucination(ne.source, "", ne.confidence, user_message)
        new_id = _next_id(current_case.timeline, "t")
        event_item = TimelineEvent(
            id=new_id,
            when_text=ne.when_text,
            when_iso=ne.when_iso,
            description=ne.description,
            source=src,
            confidence=conf,
            status=Status.active,
        )
        current_case.timeline.append(event_item)
        existing_events.add(norm)
        change_report["added_timeline"].append(new_id)

    # 4. Add People
    existing_people = {_normalize_text(f"{p.role}:{p.name or ''}") for p in current_case.people if p.status == Status.active}
    for np in patch_obj.add_people:
        norm = _normalize_text(f"{np.role}:{np.name or ''}")
        if norm in existing_people:
            change_report["skipped_duplicates"].append(norm)
            continue

        new_id = _next_id(current_case.people, "p")
        person_item = Person(
            id=new_id,
            role=np.role,
            name=np.name,
            notes=np.notes,
            status=Status.active,
        )
        current_case.people.append(person_item)
        existing_people.add(norm)
        change_report["added_people"].append(new_id)

    # 5. Add Documents
    existing_docs = {_normalize_text(d.name) for d in current_case.documents if d.status == Status.active}
    for nd in patch_obj.add_documents:
        norm = _normalize_text(nd.name)
        if norm in existing_docs:
            change_report["skipped_duplicates"].append(nd.name)
            continue

        new_id = _next_id(current_case.documents, "d")
        doc_item = Document(
            id=new_id,
            name=nd.name,
            state=nd.state,
            notes=nd.notes,
            status=Status.active,
        )
        current_case.documents.append(doc_item)
        existing_docs.add(norm)
        change_report["added_documents"].append(new_id)

    # 6. Add Deadlines
    existing_deadlines = {_normalize_text(dl.description) for dl in current_case.deadlines if dl.status == Status.active}
    for ndl in patch_obj.add_deadlines:
        norm = _normalize_text(ndl.description)
        if norm in existing_deadlines:
            change_report["skipped_duplicates"].append(ndl.description)
            continue

        src, _, _ = _verify_anti_hallucination(ndl.source, "", 1.0, user_message)
        new_id = _next_id(current_case.deadlines, "dl")
        deadline_item = Deadline(
            id=new_id,
            description=ndl.description,
            when_text=ndl.when_text,
            when_iso=ndl.when_iso,
            source=src,
            status=Status.active,
        )
        current_case.deadlines.append(deadline_item)
        existing_deadlines.add(norm)
        change_report["added_deadlines"].append(new_id)

    # 7. Apply Updates
    for up in patch_obj.update:
        target_id = up.id
        # Search in facts
        target_fact = next((f for f in current_case.facts if f.id == target_id), None)
        if target_fact:
            # Fact update: mark old as corrected, add new linked fact
            target_fact.status = Status.corrected
            new_id = _next_id(current_case.facts, "f")

            fact_data = target_fact.model_dump()
            fact_data.update(up.changes)
            fact_data["id"] = new_id
            fact_data["status"] = Status.active
            fact_data["turn"] = turn
            fact_data["note"] = f"Corrects and replaces {target_id}: {up.reason}".strip()

            new_fact = Fact.model_validate(fact_data)
            current_case.facts.append(new_fact)
            change_report["updated_items"].append({
                "kind": "fact",
                "old_id": target_id,
                "new_id": new_id,
                "reason": up.reason,
            })
            continue

        # Search in timeline
        target_event = next((t for t in current_case.timeline if t.id == target_id), None)
        if target_event:
            for k, v in up.changes.items():
                if hasattr(target_event, k):
                    setattr(target_event, k, v)
            target_event.note = f"{target_event.note} (Updated: {up.reason})".strip()
            change_report["updated_items"].append({"kind": "timeline", "id": target_id, "reason": up.reason})
            continue

        # Search in people
        target_person = next((p for p in current_case.people if p.id == target_id), None)
        if target_person:
            for k, v in up.changes.items():
                if hasattr(target_person, k):
                    setattr(target_person, k, v)
            change_report["updated_items"].append({"kind": "person", "id": target_id, "reason": up.reason})
            continue

        # Search in documents
        target_doc = next((d for d in current_case.documents if d.id == target_id), None)
        if target_doc:
            for k, v in up.changes.items():
                if hasattr(target_doc, k):
                    setattr(target_doc, k, v)
            change_report["updated_items"].append({"kind": "document", "id": target_id, "reason": up.reason})
            continue

        # Search in deadlines
        target_dl = next((dl for dl in current_case.deadlines if dl.id == target_id), None)
        if target_dl:
            for k, v in up.changes.items():
                if hasattr(target_dl, k):
                    setattr(target_dl, k, v)
            change_report["updated_items"].append({"kind": "deadline", "id": target_id, "reason": up.reason})
            continue

        # If not found in any list: skip unknown ID
        change_report["skipped_unknown_ids"].append(target_id)

    # 8. Apply Retracts
    all_items = (
        current_case.facts
        + current_case.timeline
        + current_case.people
        + current_case.documents
        + current_case.deadlines
    )
    for ret in patch_obj.retract:
        found_item = next((it for it in all_items if it.id == ret.id), None)
        if found_item:
            found_item.status = Status.retracted
            if hasattr(found_item, "note"):
                found_item.note = f"{getattr(found_item, 'note', '')} (Retracted: {ret.reason})".strip()
            change_report["retracted_items"].append({"id": ret.id, "reason": ret.reason})
        else:
            change_report["skipped_unknown_ids"].append(ret.id)

    # 9. Lists updates (questions and assumptions)
    for q in patch_obj.add_user_questions:
        if q not in current_case.user_questions:
            current_case.user_questions.append(q)

    for oq in patch_obj.add_open_questions:
        if oq not in current_case.open_questions:
            current_case.open_questions.append(oq)

    for rq in patch_obj.resolve_open_questions:
        if rq in current_case.open_questions:
            current_case.open_questions.remove(rq)

    for asm in patch_obj.add_assumptions:
        if asm not in current_case.assumptions:
            current_case.assumptions.append(asm)

    # 10. Bump Version
    current_case.version += 1

    # 11. Final Validation
    validated_case = CaseFile.model_validate(current_case)
    return validated_case, change_report


def render_case_summary(case: CaseFile | dict) -> str:
    """
    Renders a compact plain-text summary of ACTIVE items only, with hard core first.
    Used for injecting case context into conversational prompts.
    """
    if isinstance(case, dict):
        c = CaseFile.model_validate(case)
    else:
        c = case

    lines: list[str] = []

    # Hard Core First
    hc = c.hard_core
    lines.append(f"Jurisdiction: {hc.jurisdiction or 'Unknown'} (confidence: {hc.jurisdiction_confidence:.1f})")
    lines.append(f"User goal: {hc.user_goal or 'Not yet specified'}")
    lines.append(f"Urgency: {hc.urgency.level.value} - {hc.urgency.reason or 'No reason provided'}")
    lines.append(f"Legal area guess: {hc.legal_area_guess or 'Unclear'}")
    lines.append(f"Language: {hc.language}")

    # Active Facts
    active_facts = [f for f in c.facts if f.status == Status.active]
    lines.append("\nFacts:")
    if active_facts:
        for f in active_facts:
            lines.append(f"- {f.text}")
    else:
        lines.append("- (nothing yet)")

    # Active Timeline
    active_timeline = [t for t in c.timeline if t.status == Status.active]
    if active_timeline:
        lines.append("\nTimeline:")
        for t in active_timeline:
            lines.append(f"- {t.when_text}: {t.description}")

    # Active People
    active_people = [p for p in c.people if p.status == Status.active]
    if active_people:
        lines.append("\nPeople involved:")
        for p in active_people:
            name_str = f" ({p.name})" if p.name else ""
            lines.append(f"- {p.role}{name_str}")

    # Active Documents
    active_docs = [d for d in c.documents if d.status == Status.active]
    if active_docs:
        lines.append("\nDocuments:")
        for d in active_docs:
            lines.append(f"- {d.name} [{d.state.value}]")

    # Active Deadlines
    active_deadlines = [dl for dl in c.deadlines if dl.status == Status.active]
    if active_deadlines:
        lines.append("\nDeadlines:")
        for dl in active_deadlines:
            lines.append(f"- {dl.description} by {dl.when_text}")

    # Questions
    if c.user_questions:
        lines.append("\nUser questions:")
        for uq in c.user_questions:
            lines.append(f"- {uq}")

    if c.open_questions:
        lines.append("\nOpen questions:")
        for oq in c.open_questions:
            lines.append(f"- {oq}")

    return "\n".join(lines).strip()


def render_case_json_for_prompts(case: CaseFile | dict) -> str:
    """
    Renders JSON representation of the case including IDs, used in extractor prompts.
    """
    if isinstance(case, CaseFile):
        data = case.model_dump()
    else:
        data = case
    return json.dumps(data, indent=2)
