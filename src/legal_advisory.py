"""
src/legal_advisory.py - Specialized statutory limitation warnings and evidentiary guidelines.

Provides:
- Statutory limitation period alerts for common Indian legal disputes.
- Electronic evidence compliance advisory (Section 63 of Bharatiya Sakshya Adhiniyam, 2023 / Section 65B IEA).
"""

from typing import Any


def get_limitation_warning(dispute_type: str | None, facts: list[dict | Any] | None = None) -> str | None:
    """
    Returns relevant statutory limitation and timeline warnings based on legal area and facts.
    """
    dtype = (dispute_type or "").lower()
    facts_text = " ".join(
        str(f.get("text", "") if isinstance(f, dict) else getattr(f, "text", "")).lower()
        for f in (facts or [])
    )

    # 1. Employment disputes
    if "employment" in dtype or any(w in facts_text for w in ("terminated", "fired", "salary", "wages", "severance")):
        return (
            "Statutory Timeline Alert: Under the Payment of Wages Act, 1936 and Industrial Disputes Act, 1947, "
            "claims for recovery of unpaid wages or challenging unlawful termination must be initiated within "
            "prescribed statutory periods (typically 12 months to 3 years). Serving a formal legal representation or demand "
            "promptly is advised to avoid claims becoming time-barred."
        )

    # 2. Tenancy disputes
    if "tenancy" in dtype or any(w in facts_text for w in ("landlord", "evict", "rent", "tenant", "deposit")):
        return (
            "Statutory Timeline Alert: For unlawful eviction or dispossession without due process of law, "
            "a summary suit for recovery of possession under Section 6 of the Specific Relief Act, 1963 must strictly "
            "be filed within 6 months from the date of dispossession."
        )

    # 3. Cheque bounce / Financial
    if any(w in facts_text for w in ("cheque", "check bounce", "dishonour", "138")):
        return (
            "Statutory Timeline Alert: Under Section 138 of the Negotiable Instruments Act, a formal demand notice "
            "must strictly be sent within 30 days of receiving the bank memo, giving 15 days to pay before a complaint can be filed."
        )

    # 4. Consumer disputes
    if "consumer" in dtype or any(w in facts_text for w in ("defective", "warranty", "refund", "consumer complaint")):
        return (
            "Statutory Timeline Alert: Under Section 69 of the Consumer Protection Act, 2019, consumer complaints "
            "must be filed within 2 years from the date on which the cause of action arose."
        )

    return None


def get_electronic_evidence_guidance(facts: list[dict | Any] | None = None) -> str | None:
    """
    Detects if electronic records (CCTV, WhatsApp, emails, recordings) are present,
    and returns Section 63 BSA / Section 65B electronic evidence guidance.
    """
    if not facts:
        return None

    facts_text = " ".join(
        str(f.get("text", "") if isinstance(f, dict) else getattr(f, "text", "")).lower()
        for f in facts
    )

    digital_evidence_keywords = (
        "cctv", "whatsapp", "email", "e-mail", "recording", "audio recording",
        "video recording", "screenshot", "sms", "chat", "call record"
    )

    if any(kw in facts_text for kw in digital_evidence_keywords):
        return (
            "Electronic Evidence Admissibility Note: In Indian courts and statutory tribunals, digital evidence "
            "(emails, WhatsApp chats, CCTV footage, audio recordings) is admissible under Section 63 of the "
            "Bharatiya Sakshya Adhiniyam, 2023 (BSA) / Section 65B of the Indian Evidence Act. "
            "Ensure you preserve the original electronic device/storage medium with untampered metadata and file hashes, "
            "as a Section 63 electronic certificate will be required when presenting this evidence."
        )

    return None
