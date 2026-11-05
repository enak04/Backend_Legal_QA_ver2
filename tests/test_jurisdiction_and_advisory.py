"""
tests/test_jurisdiction_and_advisory.py - Tests for Indian jurisdiction resolution,
statutory limitation warnings, electronic evidence advisories, and document clause extraction.
"""

import pytest

from src.case_ops import apply_patch, render_case_json_for_prompts, render_case_summary
from src.document_parser import extract_text_from_file
from src.jurisdiction import canonicalize_jurisdiction, resolve_indian_jurisdiction
from src.legal_advisory import get_electronic_evidence_guidance, get_limitation_warning
from src.readiness import is_ready
from src.schemas import (
    CaseFile,
    CasePatch,
    Fact,
    GapAnalysis,
    HardCore,
    NewFact,
    Source,
    Status,
)


def test_resolve_indian_jurisdiction_cities():
    """Verify city resolution to State, High Court, and State Acts."""
    # Bangalore / Bengaluru
    bengaluru = resolve_indian_jurisdiction("I work in Bangalore")
    assert bengaluru is not None
    assert bengaluru["city"] == "Bengaluru"
    assert bengaluru["state"] == "Karnataka"
    assert "Karnataka" in bengaluru["high_court"]
    assert "Karnataka Shops" in bengaluru["state_shop_act"]

    # Mumbai
    mumbai = resolve_indian_jurisdiction("Office in Mumbai, Bandra")
    assert mumbai is not None
    assert mumbai["state"] == "Maharashtra"
    assert "Bombay High Court" in mumbai["high_court"]

    # Delhi
    delhi = resolve_indian_jurisdiction("Located in New Delhi")
    assert delhi is not None
    assert delhi["state"] == "NCT of Delhi"
    assert "Delhi High Court" in delhi["high_court"]

    # Unknown jurisdiction returns None
    assert resolve_indian_jurisdiction("Toronto, Canada") is None
    assert resolve_indian_jurisdiction(None) is None


def test_canonicalize_jurisdiction():
    """Verify canonicalize_jurisdiction produces 'City, State' format."""
    assert canonicalize_jurisdiction("bangalore") == "Bengaluru, Karnataka"
    assert canonicalize_jurisdiction("Bengaluru, Karnataka") == "Bengaluru, Karnataka"
    assert canonicalize_jurisdiction("Gurgaon") == "Gurugram, Haryana"
    assert canonicalize_jurisdiction("California") == "California"
    assert canonicalize_jurisdiction(None) is None


def test_limitation_warnings():
    """Verify statutory limitation alerts for employment, tenancy, cheque bounce, consumer."""
    # Employment
    facts_emp = [Fact(id="f1", text="Employer terminated me without prior notice")]
    emp_warn = get_limitation_warning("Employment Law", facts_emp)
    assert emp_warn is not None
    assert "Payment of Wages Act" in emp_warn or "Industrial Disputes Act" in emp_warn
    assert "time-barred" in emp_warn

    # Tenancy
    facts_ten = [Fact(id="f1", text="Landlord is trying to evict me and keep my deposit")]
    ten_warn = get_limitation_warning("Tenancy Law", facts_ten)
    assert ten_warn is not None
    assert "Specific Relief Act" in ten_warn
    assert "6 months" in ten_warn

    # Cheque bounce
    facts_chk = [Fact(id="f1", text="Received cheque bounce memo from bank")]
    chk_warn = get_limitation_warning(None, facts_chk)
    assert chk_warn is not None
    assert "Section 138" in chk_warn
    assert "30 days" in chk_warn

    # None for unrelated
    facts_other = [Fact(id="f1", text="Neighbor dog barks loudly")]
    assert get_limitation_warning("Nuisance", facts_other) is None


def test_electronic_evidence_guidance():
    """Verify Section 63 BSA guidance when digital evidence is mentioned."""
    # Digital evidence keywords present
    facts_with_digital = [
        Fact(id="f1", text="I received an email confirmation of termination"),
        Fact(id="f2", text="I have WhatsApp screenshots of conversation with HR"),
    ]
    guidance = get_electronic_evidence_guidance(facts_with_digital)
    assert guidance is not None
    assert "Section 63" in guidance
    assert "Bharatiya Sakshya Adhiniyam" in guidance or "65B" in guidance

    # No digital evidence
    facts_paper = [Fact(id="f1", text="Signed physical paper document with ink")]
    assert get_electronic_evidence_guidance(facts_paper) is None
    assert get_electronic_evidence_guidance([]) is None


def test_apply_patch_canonicalizes_jurisdiction():
    """Verify apply_patch canonicalizes raw city into City, State."""
    case = CaseFile()
    patch = CasePatch(set_hard_core={"jurisdiction": "bangalore"})
    updated_case, _ = apply_patch(case, patch, "I live in bangalore", turn=1)
    assert updated_case.hard_core.jurisdiction == "Bengaluru, Karnataka"

    # Enriched in render_case_json_for_prompts
    json_repr = render_case_json_for_prompts(updated_case)
    assert "governing_jurisdiction_details" in json_repr
    assert "Karnataka Shops and Commercial Establishments Act" in json_repr


def test_smart_clause_extraction(tmp_path):
    """Verify that multi-page contracts extract critical clauses when exceeding limit."""
    # Create a long synthetic employment agreement
    agreement_text = (
        "EMPLOYMENT AGREEMENT\n\nThis Agreement is entered between Acme Corp and Rahul.\n\n"
        + ("Standard non-essential boiler plate terms and definitions.\n\n" * 150)
        + "TERMINATION AND NOTICE PERIOD CLAUSE:\nEither party may terminate this employment by giving "
        + "90 days written notice or gross salary in lieu of notice period. Severance shall be 3 months pay.\n\n"
        + ("More general standard operational clauses and handbook policies.\n\n" * 100)
        + "IN WITNESS WHEREOF, the parties have executed this Agreement on November 5, 2026.\nSignatures."
    )

    content_bytes = agreement_text.encode("utf-8")
    extracted = extract_text_from_file(content=content_bytes, filename="contract.txt", max_length=2000)
    assert len(extracted) <= 6000  # Within bounded size
    assert "TERMINATION AND NOTICE PERIOD CLAUSE" in extracted
    assert "90 days written notice" in extracted
    assert "[... Extracted Critical Clauses ...]" in extracted


def test_goal_inference_from_user_questions():
    """Verify that a case with user questions auto-infers user_goal if missing."""
    from src.config import get_settings
    settings = get_settings()

    case = CaseFile(
        hard_core=HardCore(jurisdiction="Bengaluru, Karnataka", user_goal=None),
        user_questions=["How can they terminate me suddenly without notice?"],
    )
    gaps = GapAnalysis(gaps=[])

    ready, reason = is_ready(
        case=case,
        gap_analysis=gaps,
        wants_answer_now=False,
        turn_count=2,
        settings=settings,
    )
    assert ready is True
    assert "Assess legal rights and remedies" in case.hard_core.user_goal
