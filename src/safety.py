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
import unicodedata
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
    """Normalizes language name to config key ('en', 'hi', 'hinglish'). Anything else -> English."""
    lang = language_str.strip().lower()
    if "hinglish" in lang:
        return "hinglish"
    if "hindi" in lang or "devanagari" in lang:
        return "hi"
    return "en"


def get_helpline_reply(category: str, language: str = "English") -> str:
    """
    Retrieves the pre-approved helpline text for a category and language.
    If the category has a reviewed 'lead_in' line, it is placed before the message so the
    reply reads like a person responding, while the reviewed message itself stays verbatim.
    """
    config = get_helplines_config()
    categories = config.get("categories", {})
    cat_data = categories.get(category, {})
    messages = cat_data.get("messages", {})
    lead_ins = cat_data.get("lead_in", {})

    lang_key = _resolve_language_key(language)
    # Use one language for both lead-in and message (falling back to English together)
    msg_lang = lang_key if messages.get(lang_key) else "en"
    reply = messages.get(msg_lang)
    if not reply:
        return (
            "If you are in immediate danger or distress, "
            "please contact emergency services or a local crisis line immediately."
        )
    lead_in = lead_ins.get(msg_lang)
    return f"{lead_in} {reply}" if lead_in else reply


# Next-turn guidance for the talker after a safety message, so it follows up like a person would
SAFETY_FOLLOW_UP_NOTES: dict[str, str] = {
    "self_harm": "Earlier you shared crisis support details because they mentioned wanting to harm themselves. Before anything legal, check gently and briefly whether they are safe right now. Do not repeat the helpline details.",
    "violence_abuse": "Earlier you shared safety contacts because they may be in danger from someone. Check briefly whether they are somewhere safe right now, then carry on with their situation. Do not repeat the helpline details.",
    "child_safety": "Earlier you shared child-safety contacts. Check briefly whether the child is safe right now, then carry on. Do not repeat the helpline details.",
    "arrest_detention": "Earlier you shared legal-aid contacts because someone has been arrested or detained. Carry on helping with the situation; useful next things are who was detained and where they are being held. Do not repeat the helpline details.",
}


# ---------------------------------------------------------------------------
# Regex Patterns with Word Boundaries and False-Positive Exclusion
# ---------------------------------------------------------------------------

# Self harm patterns
SELF_HARM_REGEX = re.compile(
    r"\b("
    r"kill\s+myself|end\s+my\s+life|suicide|suicidal|commit(?:ting)?\s+suicide|want\s+to\s+die|harm\s+myself|"
    r"hurt\s+myself|cutting\s+myself|hang\s+myself|better\s+off\s+dead|no\s+reason\s+to\s+live|"
    r"mar\s+jaunga|mar\s+jaungi|jaan\s+de\s+dunga|jaan\s+de\s+dungi|khudkushi|aatmhatya|"
    r"marne\s+ja\s+raha|marne\s+ka\s+man|jeena\s+nahi\s+chahta"
    r")\b",
    re.IGNORECASE,
)

# Violence / Abuse patterns (excluding civil disputes / ordinary harassment like landlord harassment)
VIOLENCE_ABUSE_REGEX = re.compile(
    r"\b("
    r"hit\s+me|beat\s+me|beating\s+me|abusing\s+me|domestic\s+violence|threatened\s+to\s+kill|"
    r"strangled|choked\s+me|stalking\s+me|sexual\s+assault|rape|raped|will\s+shoot\s+me|"
    r"stabbed|going\s+to\s+kill\s+him|going\s+to\s+shoot\s+her|will\s+murder|"
    r"maar\s+peet|jaan\s+se\s+marne\s+ki\s+dhamki|peet\s+raha\s+hai|domestic\s+abuse"
    r")\b",
    re.IGNORECASE,
)

# Child safety patterns
CHILD_SAFETY_REGEX = re.compile(
    r"\b("
    r"child\s+abuse|child\s+neglect|hurting\s+my\s+child|hurting\s+the\s+baby|"
    r"molesting\s+a\s+child|pedophile|hitting\s+the\s+kids|child\s+endangerment|"
    r"bachhe\s+ko\s+maarna|bacho\s+par\s+atyachar"
    r")\b",
    re.IGNORECASE,
)

# Arrest / Detention patterns (excluding 'arrested development', 'cardiac arrest')
ARREST_DETENTION_REGEX = re.compile(
    r"\b("
    r"police\s+arrested|arrested\s+my|got\s+arrested|in\s+police\s+custody|put\s+in\s+jail|"
    r"sent\s+to\s+jail|detained\s+by\s+police|facing\s+arrest|handcuffed|held\s+in\s+custody|"
    r"warrant\s+for\s+my\s+arrest|"
    r"police\s+ne\s+pakad\s+liya|police\s+arrest|jail\s+mein\s+hai|hiraasat\s+mein|arrest\s+kar\s+liya"
    r")\b",
    re.IGNORECASE,
)

# Hindi in Devanagari script. Kept separate from the patterns above because \b does not work
# reliably with Devanagari vowel signs (they are not "word" characters to the regex engine), so
# "\bआत्महत्या\b" never matches. These phrases are specific enough to match anywhere.
# Messages are NFC-normalised first; nukta (\u093C) is optional, and chandrabindu/anusvara are
# treated as the same.
_N = "\u093C?"   # optional nukta: ख़ / ख, ज़ / ज
_AN = "[\u0901\u0902]"   # ँ or ं

def _dev(pattern: str) -> re.Pattern:
    """Compiles a Devanagari pattern after NFC-normalising it, the same way messages are."""
    return re.compile(unicodedata.normalize("NFC", pattern))


DEVANAGARI_SELF_HARM_REGEX = _dev(
    "आत्महत्या|ख" + _N + "ुदकुशी|"
    "मर\\s+जा(?:ऊ|उ)" + _AN + "?ग[ाी]|"
    "(?:अपनी\\s+)?जान\\s+दे\\s+द(?:ू|ु)" + _AN + "?ग[ाी]|"
    "जान\\s+ले\\s+ल(?:ू|ु)" + _AN + "?ग[ाी]|"
    "जीना\\s+नहीं\\s+चाहत[ाी]|मरना\\s+चाहत[ाी]|मरने\\s+का\\s+मन|"
    "ख" + _N + "ुद\\s+को\\s+(?:नुकसान|ख" + _N + "त्म)"
)
DEVANAGARI_VIOLENCE_ABUSE_REGEX = _dev(
    "मार\\s*-?\\s*पीट|घरेलू\\s+हिंसा|बलात्कार|यौन\\s+(?:उत्पीड़न|उत्पीडन|शोषण)|"
    "जान\\s+से\\s+मारने\\s+की\\s+धमकी|"
    "मुझे\\s+(?:रोज" + _N + "\\s+)?(?:मारत[ाीे]|पीटत[ाीे]|पीट\\s+रह[ाीे])"
)
DEVANAGARI_CHILD_SAFETY_REGEX = _dev(
    "बाल\\s+(?:शोषण|उत्पीड़न|उत्पीडन)|"
    "बच्च(?:े|ों)\\s+(?:को|पर)\\s+(?:मारत[ाीे]|पीटत[ाीे]|अत्याचार)"
)
DEVANAGARI_ARREST_DETENTION_REGEX = _dev(
    "गिरफ" + _N + "्तार|हिरासत\\s+में|पुलिस\\s+ने\\s+.{0,20}पकड़|जेल\\s+(?:में\\s+है|भेज)"
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


# Explicit safe confirmation detection (user replying that they are currently safe)
SAFE_CONFIRMATION_REGEX = re.compile(
    r"\b("
    r"(?:yes,?\s*)?(?:i\s*am|i'?m|we\s*are|we'?re|currently|feel)\s*(?:completely\s*)?(?:safe|fine|okay|ok|in\s+no\s+danger|out\s+of\s+danger|not\s+in\s+danger)|"
    r"not\s+in\s+(?:immediate\s+)?danger|out\s+of\s+danger|no\s+danger\s+here|no\s+danger\s+now|"
    r"safe\s+(?:u|you)\s+idiot"
    r")\b|"
    r"^(?:yes,?\s*)?safe(?:[.!,?].*)?$",
    re.IGNORECASE,
)


def is_safe_confirmation(message: str) -> bool:
    """Detects whether user is confirming that they are safe or out of danger."""
    clean = message.strip()
    return bool(SAFE_CONFIRMATION_REGEX.search(clean))


def detect_keywords(message: str) -> str | None:
    """
    Evaluates message against safety regexes in priority order:
    self_harm > violence_abuse > child_safety > arrest_detention.
    Returns the category name or None.
    """
    clean_msg = unicodedata.normalize("NFC", message.strip())
    if not clean_msg:
        return None

    # Check for known false positive idioms if matched in isolation
    is_pure_false_positive = any(fp.search(clean_msg) for fp in FALSE_POSITIVES_EXCLUSIONS)

    # 1. Self Harm
    if SELF_HARM_REGEX.search(clean_msg) or DEVANAGARI_SELF_HARM_REGEX.search(clean_msg):
        return "self_harm"

    # 2. Violence / Abuse
    if VIOLENCE_ABUSE_REGEX.search(clean_msg) or DEVANAGARI_VIOLENCE_ABUSE_REGEX.search(clean_msg):
        return "violence_abuse"

    # 3. Child Safety
    if CHILD_SAFETY_REGEX.search(clean_msg) or DEVANAGARI_CHILD_SAFETY_REGEX.search(clean_msg):
        return "child_safety"

    # 4. Arrest / Detention
    if DEVANAGARI_ARREST_DETENTION_REGEX.search(clean_msg):
        return "arrest_detention"
    if ARREST_DETENTION_REGEX.search(clean_msg):
        if is_pure_false_positive and not any(kw in clean_msg.lower() for kw in ("police", "jail", "cops", "custody", "warrant")):
            return None
        return "arrest_detention"

    return None


def decide(
    message: str,
    affect: AffectReading | None = None,
    language: str = "English",
    existing_safety_flags: list[str] | None = None,
) -> SafetyDecision:
    """
    Makes safety escalation decision based on keyword hit OR affect crisis flags.
    Priority: self_harm > violence_abuse > child_safety > arrest_detention.
    """
    clean_msg = message.strip()
    user_confirms_safe = is_safe_confirmation(clean_msg)
    keyword_cat = detect_keywords(clean_msg)

    # If the user explicitly states they are safe/not in danger, and no active crisis keywords
    # are present in the current message, do NOT trigger safety from lingering affect flags.
    if user_confirms_safe and not keyword_cat:
        return SafetyDecision(triggered=False, category=None, reply_text=None, source=None)

    # Check categories by strict priority
    # Priority 1: self_harm
    if keyword_cat == "self_harm":
        return SafetyDecision(
            triggered=True,
            category="self_harm",
            reply_text=get_helpline_reply("self_harm", language),
            source="keywords",
        )
    if affect and affect.crisis_self_harm and not user_confirms_safe:
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
    if affect and affect.crisis_violence_or_abuse and not user_confirms_safe:
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
    if affect and affect.crisis_detention and not user_confirms_safe:
        return SafetyDecision(
            triggered=True,
            category="arrest_detention",
            reply_text=get_helpline_reply("arrest_detention", language),
            source="affect",
        )

    # No crisis triggered
    return SafetyDecision(triggered=False, category=None, reply_text=None, source=None)
