"""
src/safety.py - Pure code safety filters and keyword detection.

Implements rule-based safety screening for:
- self_harm
- violence_abuse
- child_safety
- arrest_detention

Evaluates both text keyword regexes and model affect flags.
Priority: self_harm > violence_abuse > child_safety > arrest_detention.
Urgency without crisis (e.g. deadlines) does not trigger safety messages.
"""

import json
import re
from pathlib import Path
from typing import Any

from src.schemas import AffectReading, SafetyDecision

# Path to helpline numbers and reviewed safety messages
HELPLINES_PATH = Path(__file__).resolve().parent.parent / "config" / "helplines.json"

# Cached helplines config
_helplines_data: dict[str, Any] | None = None


def get_helplines_config() -> dict[str, Any]:
    global _helplines_data
    if _helplines_data is None:
        if HELPLINES_PATH.is_file():
            _helplines_data = json.loads(HELPLINES_PATH.read_text(encoding="utf-8"))
        else:
            _helplines_data = {}
    return _helplines_data


def _resolve_language_key(language_str: str) -> str:
    """Normalizes language name to config key ('en', 'hi', 'hinglish', 'es')."""
    lang = language_str.strip().lower()
    if "hinglish" in lang:
        return "hinglish"
    if "hindi" in lang:
        return "hi"
    if "spanish" in lang or "español" in lang:
        return "es"
    return "en"


def get_helpline_reply(category: str, language: str = "English") -> str:
    """Retrieves the pre-approved helpline text message for a category and language."""
    config = get_helplines_config()
    categories = config.get("categories", {})
    cat_data = categories.get(category, {})
    messages = cat_data.get("messages", {})

    lang_key = _resolve_language_key(language)
    reply = messages.get(lang_key) or messages.get("en")
    if not reply:
        return (
            "[REVIEW WITH A NATIVE SPEAKER AND A LAWYER] If you are in immediate danger or distress, "
            "please contact emergency services or a local crisis line immediately."
        )
    return reply


# ---------------------------------------------------------------------------
# Regex Patterns with Word Boundaries and False-Positive Exclusion
# ---------------------------------------------------------------------------

# Self harm patterns
SELF_HARM_REGEX = re.compile(
    r"\b("
    r"kill\s+myself|end\s+my\s+life|suicide|suicidal|commit(?:ting)?\s+suicide|want\s+to\s+die|harm\s+myself|"
    r"hurt\s+myself|cutting\s+myself|hang\s+myself|better\s+off\s+dead|no\s+reason\s+to\s+live|"
    r"suicidarme|quitarme\s+la\s+vida|matarme|hacerme\s+daño|no\s+quiero\s+vivir|"
    r"mar\s+jaunga|mar\s+jaungi|jaan\s+de\s+dunga|jaan\s+de\s+dungi|khudkushi|aatmhatya|"
    r"marne\s+ja\s+raha|marne\s+ka\s+man|jeena\s+nahi\s+chahta|आत्महत्या|मर\s+जाऊंगा"
    r")\b",
    re.IGNORECASE,
)

# Violence / Abuse patterns (excluding civil disputes / ordinary harassment like landlord harassment)
VIOLENCE_ABUSE_REGEX = re.compile(
    r"\b("
    r"hit\s+me|beat\s+me|beating\s+me|abusing\s+me|domestic\s+violence|threatened\s+to\s+kill|"
    r"strangled|choked\s+me|stalking\s+me|sexual\s+assault|rape|raped|will\s+shoot\s+me|"
    r"stabbed|going\s+to\s+kill\s+him|going\s+to\s+shoot\s+her|will\s+murder|"
    r"maar\s+peet|jaan\s+se\s+marne\s+ki\s+dhamki|peet\s+raha\s+hai|domestic\s+abuse|"
    r"violencia\s+doméstica|me\s+está\s+pegando|me\s+amenazó\s+de\s+muerte|abuso\s+sexual"
    r")\b",
    re.IGNORECASE,
)

# Child safety patterns
CHILD_SAFETY_REGEX = re.compile(
    r"\b("
    r"child\s+abuse|child\s+neglect|hurting\s+my\s+child|hurting\s+the\s+baby|"
    r"molesting\s+a\s+child|pedophile|hitting\s+the\s+kids|child\s+endangerment|"
    r"bachhe\s+ko\s+maarna|bacho\s+par\s+atyachar|abuso\s+infantil|maltrato\s+infantil"
    r")\b",
    re.IGNORECASE,
)

# Arrest / Detention patterns (excluding 'arrested development', 'cardiac arrest')
ARREST_DETENTION_REGEX = re.compile(
    r"\b("
    r"police\s+arrested|arrested\s+my|got\s+arrested|in\s+police\s+custody|put\s+in\s+jail|"
    r"sent\s+to\s+jail|detained\s+by\s+police|facing\s+arrest|handcuffed|held\s+in\s+custody|"
    r"warrant\s+for\s+my\s+arrest|"
    r"police\s+ne\s+pakad\s+liya|police\s+arrest|jail\s+mein\s+hai|hiraasat\s+mein|arrest\s+kar\s+liya|"
    r"policía\s+arrestó|fue\s+arrestado|en\s+la\s+cárcel"
    r")\b",
    re.IGNORECASE,
)

# Explicit exclusion for idioms and false positives
FALSE_POSITIVES_EXCLUSIONS = [
    re.compile(r"\barrested\s+development\b", re.IGNORECASE),
    re.compile(r"\bcardiac\s+arrest\b", re.IGNORECASE),
    re.compile(r"\bdying\s+to\s+know\b", re.IGNORECASE),
    re.compile(r"\bkill\s+the\s+lights\b", re.IGNORECASE),
    re.compile(r"\bkiller\s+deal\b", re.IGNORECASE),
    re.compile(r"\bkilling\s+time\b", re.IGNORECASE),
    re.compile(r"\bdead\s+battery\b", re.IGNORECASE),
    re.compile(r"\bdead\s+end\b", re.IGNORECASE),
]


def detect_keywords(message: str) -> str | None:
    """
    Evaluates message against safety regexes in priority order:
    self_harm > violence_abuse > child_safety > arrest_detention.
    Returns the category name or None.
    """
    clean_msg = message.strip()
    if not clean_msg:
        return None

    # Check for known false positive idioms if matched in isolation
    is_pure_false_positive = any(fp.search(clean_msg) for fp in FALSE_POSITIVES_EXCLUSIONS)

    # 1. Self Harm
    if SELF_HARM_REGEX.search(clean_msg):
        return "self_harm"

    # 2. Violence / Abuse
    if VIOLENCE_ABUSE_REGEX.search(clean_msg):
        return "violence_abuse"

    # 3. Child Safety
    if CHILD_SAFETY_REGEX.search(clean_msg):
        return "child_safety"

    # 4. Arrest / Detention
    if ARREST_DETENTION_REGEX.search(clean_msg):
        if is_pure_false_positive and not any(kw in clean_msg.lower() for kw in ("police", "jail", "cops", "custody", "warrant")):
            return None
        return "arrest_detention"

    return None


def decide(
    message: str,
    affect: AffectReading | None = None,
    language: str = "English",
) -> SafetyDecision:
    """
    Makes safety escalation decision based on keyword hit OR affect crisis flags.
    Priority: self_harm > violence_abuse > child_safety > arrest_detention.
    """
    keyword_cat = detect_keywords(message)

    # Check categories by strict priority
    # Priority 1: self_harm
    if keyword_cat == "self_harm":
        return SafetyDecision(
            triggered=True,
            category="self_harm",
            reply_text=get_helpline_reply("self_harm", language),
            source="keywords",
        )
    if affect and affect.crisis_self_harm:
        return SafetyDecision(
            triggered=True,
            category="self_harm",
            reply_text=get_helpline_reply("self_harm", language),
            source="affect",
        )

    # Priority 2: violence_abuse
    if keyword_cat == "violence_abuse":
        return SafetyDecision(
            triggered=True,
            category="violence_abuse",
            reply_text=get_helpline_reply("violence_abuse", language),
            source="keywords",
        )
    if affect and affect.crisis_violence_or_abuse:
        return SafetyDecision(
            triggered=True,
            category="violence_abuse",
            reply_text=get_helpline_reply("violence_abuse", language),
            source="affect",
        )

    # Priority 3: child_safety
    if keyword_cat == "child_safety":
        return SafetyDecision(
            triggered=True,
            category="child_safety",
            reply_text=get_helpline_reply("child_safety", language),
            source="keywords",
        )

    # Priority 4: arrest_detention
    if keyword_cat == "arrest_detention":
        return SafetyDecision(
            triggered=True,
            category="arrest_detention",
            reply_text=get_helpline_reply("arrest_detention", language),
            source="keywords",
        )
    if affect and affect.crisis_detention:
        return SafetyDecision(
            triggered=True,
            category="arrest_detention",
            reply_text=get_helpline_reply("arrest_detention", language),
            source="affect",
        )

    # No crisis triggered
    return SafetyDecision(triggered=False, category=None, reply_text=None, source=None)
