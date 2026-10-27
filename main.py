"""
main.py - Interactive terminal chat for legal-intake-bot.

Workflow:
1. Shows WELCOME, DISCLAIMER, and PRIVACY_NOTICE.
2. Prompts user for consent (requires typing 'Y' or 'yes').
3. Creates a session in MongoDB with a timestamp.
4. Loops: reads user message, saves to DB, streams talker reply, saves reply to DB.
5. Supports resuming previous sessions with: python main.py --resume <SESSION_ID>
6. Supports --debug flag for verbose error and latency output.
Type 'quit' to exit.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import sys

from src.config import get_settings
from src.db import create_session, get_recent_messages, get_session
from src.llm import FakeLLM, LLM
from src.messages import CONSENT_PROMPT, DISCLAIMER, GENERIC_ERROR_REPLY, PRIVACY_NOTICE, WELCOME
from src.pipeline import handle_turn


async def run_chat(resume_id: str | None = None, debug: bool = False) -> None:
    settings = get_settings()

    # Initialize LLM client: use FakeLLM if provider is 'fake' or API key is not yet set
    if settings.provider == "fake" or not settings.llm_api_key:
        if debug:
            print("[DEBUG] Using FakeLLM (provider is 'fake' or no LLM_API_KEY set)")
        from src.schemas import AffectReading, CasePatch, Intent, NewFact, UnderstandOutput
        llm_client = FakeLLM(responses={
            "extractor": UnderstandOutput(
                intent=Intent.provides_info,
                language="English",
                patch=CasePatch(
                    add_facts=[NewFact(text="User shared initial context.", source="inferred", confidence=0.8)]
                ),
                change_summary="Recorded initial details",
            ),
            "affect": AffectReading(
                distress=1,
                anger=0,
                confusion=0,
                shame=0,
                urgency=1,
                crisis_self_harm=False,
                crisis_violence_or_abuse=False,
                crisis_detention=False,
                language="English",
                style_note="Keep the tone calm and supportive.",
            ),
            "talker": "I hear you, and I am here to help you outline what happened. Could you please share a little more detail about the main issue you are facing?",
        })
    else:
        if debug:
            print(f"[DEBUG] Using production LLM ({settings.smart_model})")
        llm_client = LLM()

    # Session initialization or resumption
    session: dict
    if resume_id:
        existing_session = await get_session(resume_id)
        if not existing_session:
            print(f"\n[Error] Could not find session with ID: {resume_id}")
            print("Starting a new session instead.\n")
            resume_id = None
        else:
            session = existing_session
            session_id = session["_id"]
            print(f"\n=== Resumed Session: {session_id} ===")
            past_messages = await get_recent_messages(session_id, n=20)
            if past_messages:
                print("--- Previous Messages ---")
                for m in past_messages:
                    sender = "You" if m.get("role") == "user" else "Assistant"
                    print(f"{sender}: {m.get('text', '')}")
                print("-------------------------")

    if not resume_id:
        print("\n" + "=" * 60)
        print(WELCOME)
        print("-" * 60)
        print(DISCLAIMER)
        print("-" * 60)
        print(PRIVACY_NOTICE)
        print("=" * 60 + "\n")

        # Consent prompt loop
        while True:
            try:
                user_choice = input(CONSENT_PROMPT).strip()
            except (KeyboardInterrupt, EOFError):
                print("\nSession aborted.")
                return

            if user_choice.lower() in ("quit", "exit"):
                print("Exiting.")
                return

            if user_choice.lower() in ("y", "yes"):
                break
            else:
                print("Consent is required to use this service. Please type 'Y' to agree or 'quit' to exit.\n")

        # Create session in DB
        now = datetime.now(timezone.utc)
        session = await create_session(
            consent=True,
            consent_ts=now,
            language="en",
        )
        session_id = session["_id"]
        print(f"\n[Session created] ID: {session_id}")
        print(f"(You can resume this chat at any time using: python main.py --resume {session_id})\n")

    session_id = session["_id"]
    print("Chat started. Type 'quit' or 'exit' at any time to leave.\n")

    # Interactive turn loop
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("\nThank you for sharing your details. Take care!")
            break

        try:
            print("\nAssistant: ", end="", flush=True)
            reply, decision = await handle_turn(
                session=session,
                user_message=user_input,
                llm=llm_client,
                stream=True,
                debug=debug,
            )

            if reply == "READY_FOR_CONFIRMATION":
                print("[READY_FOR_CONFIRMATION: We have collected enough details. Ready to confirm facts with user.]\n")
            elif hasattr(reply, "__aiter__"):
                async for chunk in reply:
                    print(chunk, end="", flush=True)
                print("\n")
            else:
                print(str(reply) + "\n")


        except Exception as e:
            if debug:
                print(f"\n[DEBUG ERROR] {e}", file=sys.stderr)
            print(f"\n{GENERIC_ERROR_REPLY}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="legal-intake-bot terminal chat")
    parser.add_argument("--resume", type=str, help="Session ID to resume")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug output")

    args = parser.parse_args()

    try:
        asyncio.run(run_chat(resume_id=args.resume, debug=args.debug))
    except KeyboardInterrupt:
        print("\nChat terminated.")


if __name__ == "__main__":
    main()
