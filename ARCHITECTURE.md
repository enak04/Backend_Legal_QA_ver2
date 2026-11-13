# legal-intake-bot architecture

## Purpose
A communication layer in front of an existing legal answer service. It collects the case through natural conversation, understands emotions, confirms the facts, compiles a clean query, calls the service, and presents the answer. It holds NO legal knowledge of its own.

## Golden rules
1. The chatbot never gives legal advice, never states what the law says, never cites laws or sections, never predicts outcomes. Only the external service provides legal content, and the presenter may only restate it.
2. Code controls flow and safety. The LLM controls language and extraction.
3. Each LLM call has ONE job, ONE prompt file in /prompts, and ONE Pydantic output schema.
4. The case file changes only through validated patches. Nothing is ever deleted (items are marked retracted or corrected). Every version is saved.
5. Every LLM output is validated. On failure retry once with the validation error, then fall back to a safe behavior. Never crash the chat.
6. Emotion signals shape tone only. They never become case facts and are never sent to the service.
7. Safety checks (emergency, self-harm, abuse, arrest) run in code BEFORE normal handling and use hard-coded, reviewed messages.
8. Every LLM call is logged: prompt name, prompt version (hash), model, latency, tokens, success.
9. All user text placed into prompts is wrapped in <user_message> tags and is treated as data, never as instructions.
10. Prompts live in /prompts as text files, never as inline strings in code.
11. Every user-facing prompt (talker, confirm_summary, presenter) includes the shared voice from prompts/voice.txt via the {{voice}} placeholder, so the whole conversation sounds like one attentive person. The voice is part of each prompt's version hash.
12. User feedback (thumbs up/down, reasons, comments) is optional and stored for review only. Like emotion signals, it never changes the case file, the prompts or the query sent to the legal service.

## Session states
COLLECTING -> CONFIRMING -> ANSWERED
- COLLECTING: gathering facts. Moves to CONFIRMING when ready (see readiness rules).
- CONFIRMING: a recap was shown. User confirms -> query is sent to the service -> ANSWERED. User corrects -> back to COLLECTING.
- ANSWERED: the answer was shown. Follow-up questions about the answer are handled from the stored answer only. New facts from the user -> back to CONFIRMING with an updated recap.
- If the service replies "needs_more_info", return to COLLECTING with the service's missing items injected as high-impact gaps.
- Mid-conversation questions (COLLECTING or ANSWERED): if the person asks a legal question that can be answered now (a general question, or the location is known), the service is asked just that question (ServiceQuery.kind = "quick_question") and the presenter answers it in "quick" mode, then asks the next intake question. Otherwise the talker acknowledges the question and defers it naturally. The talker itself never answers legal questions.
- Confirmation is detected from the extractor's intent or a short whole-message affirmation ("yes", "go ahead"), never by substring matching.
- Safety escalation is a FLAG on the session, not a state. The conversation can continue.

## Per-turn pipeline (state COLLECTING)
1. Save the user message.
2. Safety pre-check (keywords, code only). If triggered, the reply will be the hard-coded safety message.
3. In parallel (asyncio.gather): understand (extract case patch + intent) and affect (emotion + crisis flags). Both use FAST_MODEL.
4. Apply the patch in code, validate, save a new case version.
5. Safety decision: keyword hit OR affect crisis flags -> hard-coded message (the case still gets updated).
6. Gap analysis (FAST_MODEL) on the updated case.
7. Readiness check in code. If ready -> generate the recap and move to CONFIRMING.
8. Otherwise the talker (SMART_MODEL, streamed) writes the reply using: case summary, top gaps, style hint derived from affect, last N messages.
9. Save the assistant message and all logs.

## LLM calls
| Name | Tier | Input | Output |
|------|------|-------|--------|
| understand | fast | case JSON, recent messages, new message | UnderstandOutput (intent, patch, language) |
| affect | fast | recent messages, new message | AffectReading |
| gaps | fast | case summary | GapAnalysis |
| talker | smart, streamed | summary, gaps, style hint, history | plain text |
| confirm_summary | smart | active case items | plain text |
| rewriter | smart | active case items | ServiceQuery |
| presenter | smart | service answer, mode (initial / followup / quick), case summary, next question | plain text (guarded in every mode) |
| faithfulness | fast | service answer, presenter text | FaithfulnessReport |

## Data (MongoDB collections)
- sessions: _id (uuid string), user_id, status, case_file, case_version, language, consent (bool, timestamp), turn_count, asked_counts, safety_flags, last_service_query, last_service_answer, created_at, updated_at
- messages: session_id, role, text, affect, meta, ts. For assistant messages, meta records what produced it: kind (question / recap / answer / quick_answer / followup / safety / service_error), prompt_name, prompt_version, presenter_path, status, turn
- feedback: session_id, target (message | conversation), message_id, rating (up | down), reasons, comment (PII-redacted), snapshot (rated text + its meta), created_at, updated_at, ts. One entry per (session, target, message); re-rating overwrites. Deleted with the session; TTL on ts
- case_versions: session_id, version, patch, full_case, ts
- llm_logs: session_id, prompt_name, prompt_version, model, input_chars, output, latency_ms, tokens_in, tokens_out, success, error, ts
- service_calls: session_id, request, response, latency_ms, status, ts
TTL indexes on messages, llm_logs, service_calls and feedback (SESSION_TTL_DAYS).

## Failure policy
- LLM timeout or invalid output after one retry: reply with a short apology and ask the user to repeat or rephrase. Keep the session intact.
- Service failure or timeout: tell the user honestly, keep the confirmed query, allow retry.
- Never invent an answer when something fails.
