"""
src/messages.py - Reviewed pre-set text messages for user interactions.

All constants contain sensible placeholder text with the marker [LAWYER TO REVIEW].
The DISCLAIMER explicitly states that the chatbot collects information and does NOT give legal advice.
"""

WELCOME = (
    "Hello and welcome. [LAWYER TO REVIEW] I am here to help you share the details of your legal "
    "situation so that our legal service team can prepare an accurate response for you."
)

DISCLAIMER = (
    "[LAWYER TO REVIEW] IMPORTANT LEGAL DISCLAIMER: I am an automated conversational intake assistant, "
    "NOT an attorney, and I CANNOT provide legal advice. I will only collect factual information about "
    "your issue and forward it to our qualified legal answer service. Nothing discussed here creates "
    "an attorney-client relationship, cites legal rules, or predicts outcomes."
)

PRIVACY_NOTICE = (
    "[LAWYER TO REVIEW] PRIVACY NOTICE: The facts and details you share will be securely recorded and "
    "summarized solely to help generate your legal inquiry. Please do not provide sensitive credentials, "
    "such as bank account numbers, passwords, or government identity numbers."
)

CONSENT_PROMPT = (
    "[LAWYER TO REVIEW] To proceed with your intake, please confirm that you understand and agree to the above terms. "
    "Do you consent to proceed? (Type 'Y' or 'yes' to agree): "
)

GENERIC_ERROR_REPLY = (
    "[LAWYER TO REVIEW] I am sorry, but I had trouble processing that response. Could you please rephrase "
    "or repeat what you just said?"
)
