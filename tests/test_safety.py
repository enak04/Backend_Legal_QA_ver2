"""
tests/test_safety.py - Comprehensive tests for keyword detection, false-positive resistance, and safety escalation priority.
"""

import pytest

from src.safety import decide, detect_keywords
from src.schemas import AffectReading


# ---------------------------------------------------------------------------
# Test Suite: Over 40 test sentences covering all categories and false positives
# ---------------------------------------------------------------------------

SAFETY_TEST_SENTENCES = [
    # --- Self Harm (True Positives: English, Hinglish, Spanish) ---
    ("I want to end my life", "self_harm"),
    ("I am thinking of committing suicide", "self_harm"),
    ("I might kill myself tonight", "self_harm"),
    ("I just want to die", "self_harm"),
    ("I have no reason to live anymore", "self_harm"),
    ("I feel like cutting myself", "self_harm"),
    ("I would be better off dead", "self_harm"),
    ("I am going to hang myself", "self_harm"),
    ("main mar jaunga agar yeh case haar gaya", "self_harm"),
    ("jaan de dunga main ab", "self_harm"),
    ("mera marne ka man kar raha hai", "self_harm"),
    ("khudkushi karne ki soch raha hu", "self_harm"),
    ("mujhe aatmhatya karni padegi", "self_harm"),
    ("quiero suicidarme porque no puedo mas", "self_harm"),
    ("estoy pensando en quitarme la vida", "self_harm"),
    ("tengo ganas de matarme", "self_harm"),
    ("ya no quiero vivir con este dolor", "self_harm"),

    # --- Violence / Abuse (True Positives: English, Hinglish, Spanish) ---
    ("My spouse threatened to kill me yesterday", "violence_abuse"),
    ("He hit me and gave me a black eye", "violence_abuse"),
    ("My ex is beating me whenever he comes over", "violence_abuse"),
    ("I am a victim of domestic violence", "violence_abuse"),
    ("He choked me until I passed out", "violence_abuse"),
    ("My neighbor is stalking me at my job", "violence_abuse"),
    ("He said he will shoot me if I call the police", "violence_abuse"),
    ("I was a victim of sexual assault", "violence_abuse"),
    ("woh mere sath maar peet karta hai", "violence_abuse"),
    ("usne mujhe jaan se marne ki dhamki di", "violence_abuse"),
    ("woh mujhe peet raha hai har roz", "violence_abuse"),
    ("mi pareja me amenazó de muerte", "violence_abuse"),
    ("sufro de violencia doméstica en casa", "violence_abuse"),
    ("él me está pegando constantemente", "violence_abuse"),

    # --- Child Safety (True Positives) ---
    ("I suspect child abuse in the home", "child_safety"),
    ("He is hurting my child", "child_safety"),
    ("My neighbor is hitting the kids severely", "child_safety"),
    ("The child neglect is getting terrible", "child_safety"),
    ("woh chote bachhe ko maarna shuru kar deta hai", "child_safety"),
    ("hay abuso infantil en esa casa", "child_safety"),

    # --- Arrest / Detention (True Positives) ---
    ("police arrested my brother this morning", "arrest_detention"),
    ("I got arrested last night", "arrest_detention"),
    ("My son is in police custody right now", "arrest_detention"),
    ("They put in jail without bail", "arrest_detention"),
    ("There is a warrant for my arrest", "arrest_detention"),
    ("police ne pakad liya mere bhai ko", "arrest_detention"),
    ("woh jail mein hai kal raat se", "arrest_detention"),
    ("la policía arrestó a mi hermano hoy", "arrest_detention"),
    ("él fue arrestado anoche", "arrest_detention"),

    # --- False Positives (Must NOT trigger any safety category -> None) ---
    ("I love watching arrested development on TV", None),
    ("I am dying to know what the legal answer is", None),
    ("Please kill the lights before leaving", None),
    ("I got a killer deal on my apartment lease", None),
    ("We are just killing time waiting for court", None),
    ("My car has a dead battery", None),
    ("The negotiations hit a dead end", None),
    ("He suffered a cardiac arrest last year", None),
    ("my landlord is harassing me", None),
    ("My landlord is annoying and calls repeatedly", None),
    ("I need help with my employment contract", None),
    ("Can you explain how child custody works in a divorce?", None),
]


@pytest.mark.parametrize("sentence,expected_category", SAFETY_TEST_SENTENCES)
def test_keyword_detector_benchmark(sentence: str, expected_category: str | None):
    """
    Tests sentence against the keyword detector ensuring accurate classification
    and strict resilience against false positives.
    """
    detected = detect_keywords(sentence)
    assert detected == expected_category, f"Failed on sentence: '{sentence}'. Expected: {expected_category}, got: {detected}"


def test_decide_priority_self_harm_over_violence():
    """Verify self_harm takes strict precedence over violence_abuse."""
    # Keyword matches violence, but affect has self-harm
    affect = AffectReading(crisis_self_harm=True, crisis_violence_or_abuse=True)
    decision = decide("He hit me and beat me", affect=affect)

    assert decision.triggered is True
    assert decision.category == "self_harm"


def test_decide_priority_violence_over_arrest():
    """Verify violence_abuse takes strict precedence over arrest_detention."""
    affect = AffectReading(crisis_violence_or_abuse=True, crisis_detention=True)
    decision = decide("police arrested my brother", affect=affect)

    assert decision.triggered is True
    assert decision.category == "violence_abuse"


def test_decide_priority_child_safety_over_arrest():
    """Verify child_safety takes strict precedence over arrest_detention."""
    affect = AffectReading(crisis_detention=True)
    decision = decide("He is hurting my child", affect=affect)

    assert decision.triggered is True
    assert decision.category == "child_safety"


def test_decide_urgency_without_crisis_does_not_trigger():
    """Verify that urgency alone (deadline soon) does NOT trigger hard-coded safety message."""
    affect = AffectReading(urgency=3, crisis_self_harm=False, crisis_violence_or_abuse=False, crisis_detention=False)
    decision = decide("I have a court deadline in 2 days", affect=affect)

    assert decision.triggered is False
    assert decision.category is None
    assert decision.reply_text is None
