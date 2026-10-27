"""
src/schemas.py - Pydantic v2 schemas for legal-intake-bot.

Defines all structured data models for:
- Case file entities (facts, timeline, people, documents, deadlines, hard core)
- Case patch representation (validated increments)
- Intent classification and extractor outputs (UnderstandOutput)
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Source(str, Enum):
    user_said = "user_said"
    inferred = "inferred"


class Status(str, Enum):
    active = "active"
    corrected = "corrected"
    retracted = "retracted"


class UrgencyLevel(str, Enum):
    unknown = "unknown"
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class DocumentState(str, Enum):
    has = "has"
    lost = "lost"
    dont_know = "dont_know"
    to_obtain = "to_obtain"


class Intent(str, Enum):
    provides_info = "provides_info"
    asks_legal_question = "asks_legal_question"
    asks_about_process = "asks_about_process"
    confirms = "confirms"
    corrects = "corrects"
    wants_answer_now = "wants_answer_now"
    greeting = "greeting"
    off_topic = "off_topic"
    other = "other"


class Urgency(BaseModel):
    level: UrgencyLevel = UrgencyLevel.unknown
    reason: str = ""


class HardCore(BaseModel):
    jurisdiction: str | None = None
    jurisdiction_confidence: float = 0.0
    user_goal: str | None = None
    urgency: Urgency = Field(default_factory=Urgency)
    language: str = "English"
    legal_area_guess: str | None = None


# ---------------------------------------------------------------------------
# Case File Items (with system-assigned ids and lifecycle status)
# ---------------------------------------------------------------------------

class Fact(BaseModel):
    id: str
    text: str
    source: Source = Source.user_said
    confidence: float = 1.0
    status: Status = Status.active
    evidence: str = ""  # short quote from user message
    turn: int = 0
    note: str = ""      # optional note (e.g. link to previous corrected item)


class TimelineEvent(BaseModel):
    id: str
    when_text: str
    when_iso: str | None = None
    description: str
    source: Source = Source.user_said
    confidence: float = 1.0
    status: Status = Status.active
    note: str = ""


class Person(BaseModel):
    id: str
    role: str
    name: str | None = None
    notes: str = ""
    status: Status = Status.active


class Document(BaseModel):
    id: str
    name: str
    state: DocumentState = DocumentState.dont_know
    notes: str = ""
    status: Status = Status.active


class Deadline(BaseModel):
    id: str
    description: str
    when_text: str
    when_iso: str | None = None
    source: Source = Source.user_said
    status: Status = Status.active
    note: str = ""


class CaseFile(BaseModel):
    """The master case file preserving full factual history and versioning."""
    hard_core: HardCore = Field(default_factory=HardCore)
    facts: list[Fact] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    people: list[Person] = Field(default_factory=list)
    documents: list[Document] = Field(default_factory=list)
    deadlines: list[Deadline] = Field(default_factory=list)
    user_questions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    version: int = 0


# ---------------------------------------------------------------------------
# Patch Items (without ids - the code assigns ids)
# ---------------------------------------------------------------------------

class NewFact(BaseModel):
    text: str
    source: Source = Source.user_said
    confidence: float = 1.0
    evidence: str = ""


class NewTimelineEvent(BaseModel):
    when_text: str
    description: str
    when_iso: str | None = None
    source: Source = Source.user_said
    confidence: float = 1.0


class NewPerson(BaseModel):
    role: str
    name: str | None = None
    notes: str = ""


class NewDocument(BaseModel):
    name: str
    state: DocumentState = DocumentState.dont_know
    notes: str = ""


class NewDeadline(BaseModel):
    description: str
    when_text: str
    when_iso: str | None = None
    source: Source = Source.user_said


class ItemUpdate(BaseModel):
    kind: str  # "fact", "timeline", "person", "document", "deadline"
    id: str
    changes: dict[str, Any]
    reason: str = ""


class ItemRetract(BaseModel):
    kind: str
    id: str
    reason: str = ""


class CasePatch(BaseModel):
    """Validated increment proposed by the extractor LLM."""
    set_hard_core: dict[str, Any] = Field(default_factory=dict)
    add_facts: list[NewFact] = Field(default_factory=list)
    add_timeline: list[NewTimelineEvent] = Field(default_factory=list)
    add_people: list[NewPerson] = Field(default_factory=list)
    add_documents: list[NewDocument] = Field(default_factory=list)
    add_deadlines: list[NewDeadline] = Field(default_factory=list)
    add_user_questions: list[str] = Field(default_factory=list)
    add_open_questions: list[str] = Field(default_factory=list)
    resolve_open_questions: list[str] = Field(default_factory=list)
    add_assumptions: list[str] = Field(default_factory=list)
    update: list[ItemUpdate] = Field(default_factory=list)
    retract: list[ItemRetract] = Field(default_factory=list)

    @field_validator("set_hard_core")
    @classmethod
    def validate_hard_core_keys(cls, v: dict[str, Any]) -> dict[str, Any]:
        valid_keys = set(HardCore.model_fields.keys())
        bad_keys = set(v.keys()) - valid_keys
        if bad_keys:
            raise ValueError(f"Invalid hard_core key(s): {bad_keys}. Allowed keys: {valid_keys}")
        return v


class UnderstandOutput(BaseModel):
    """Output schema for the 'understand' / extractor LLM step."""
    intent: Intent = Intent.provides_info
    wants_answer_now: bool = False
    language: str = "English"
    patch: CasePatch = Field(default_factory=CasePatch)
    change_summary: str = ""


class AffectReading(BaseModel):
    """Output schema for the 'affect' emotion and crisis detector LLM step."""
    distress: int = Field(default=0, ge=0, le=3)
    anger: int = Field(default=0, ge=0, le=3)
    confusion: int = Field(default=0, ge=0, le=3)
    shame: int = Field(default=0, ge=0, le=3)
    urgency: int = Field(default=0, ge=0, le=3)
    crisis_self_harm: bool = False
    crisis_violence_or_abuse: bool = False
    crisis_detention: bool = False
    language: str = "English"
    style_note: str = ""


class SafetyDecision(BaseModel):
    """Represents a safety escalation decision."""
    triggered: bool = False
    category: str | None = None
    reply_text: str | None = None
    source: str | None = None  # "keywords" or "affect"


class Gap(BaseModel):
    """Represents a missing piece of information needed to advise on the case."""
    topic: str
    question_hint: str
    why_it_matters: str
    impact: str = "medium"  # "high" | "medium" | "low"
    example_answer: str = ""
    can_skip: bool = False

    @field_validator("impact", mode="before")
    @classmethod
    def normalize_impact(cls, v: Any) -> str:
        if isinstance(v, str):
            v_lower = v.strip().lower()
            if v_lower in ("high", "medium", "low"):
                return v_lower
        return "medium"


class GapAnalysis(BaseModel):
    """Output schema for the 'gap_analyzer' LLM step."""
    gaps: list[Gap] = Field(default_factory=list, max_length=3)
    legal_area_guess: str | None = None
    blocking_for_service: bool = False

    @field_validator("gaps", mode="before")
    @classmethod
    def limit_gaps(cls, v: Any) -> Any:
        if isinstance(v, list) and len(v) > 3:
            return v[:3]
        return v


class ServiceQueryFact(BaseModel):
    """Fact representation for external service query."""
    text: str
    certainty: str = "stated"  # "stated" | "uncertain" | "assumed"


class ServiceQuery(BaseModel):
    """Compiled self-contained query sent to the external legal answering service."""
    query_text: str = ""
    jurisdiction: str | None = None
    legal_area: str | None = None
    user_goal: str | None = None
    urgency: dict[str, Any] = Field(default_factory=lambda: {"level": "unknown", "reason": ""})
    facts: list[ServiceQueryFact] = Field(default_factory=list)
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    parties: list[dict[str, Any]] = Field(default_factory=list)
    documents: list[dict[str, Any]] = Field(default_factory=list)
    deadlines: list[dict[str, Any]] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    questions_to_answer: list[str] = Field(default_factory=list)
    user_language: str = "English"

    @field_validator("facts", mode="before")
    @classmethod
    def normalize_facts(cls, v: Any) -> Any:
        if isinstance(v, list):
            res = []
            for item in v:
                if isinstance(item, dict):
                    res.append(ServiceQueryFact(**item))
                elif isinstance(item, ServiceQueryFact):
                    res.append(item)
                elif isinstance(item, str):
                    res.append(ServiceQueryFact(text=item, certainty="stated"))
            return res
        return v


class ServiceResponse(BaseModel):
    """Response returned from the external legal answering service."""
    status: str = "answered"  # "answered" | "needs_more_info" | "error"
    answer: str | None = None
    sections: dict[str, str] | None = None  # e.g. summary, next_steps, caveats
    missing_info: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)


class FaithfulnessReport(BaseModel):
    """Result of faithfulness validation checking presenter text against source answer."""
    unsupported_claims: list[str] = Field(default_factory=list)
    omitted_critical: list[str] = Field(default_factory=list)
    ok: bool = True



