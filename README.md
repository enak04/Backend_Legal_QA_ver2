# legal-intake-bot

A conversational intake chatbot for a legal-help service.

## What This Project Is

`legal-intake-bot` acts as a friendly, empathetic front-line communication layer before a separate external legal answering service. It listens to a user's situation, collects key facts, asks relevant follow-up questions, monitors emotional state, confirms the facts back with the user, and sends a structured query to the legal answering service. Finally, it presents the service's answer back to the user in conversational language.

> **Crucial Rule:** The chatbot **NEVER** gives legal advice, never states what the law says, and never predicts legal outcomes. All legal answers originate solely from the external legal service.

---

## Setup Instructions

### 1. Create a Virtual Environment

Open your terminal in the project root directory and run:

```bash
python -m venv .venv
```

Activate the virtual environment:
- **Windows (Command Prompt / PowerShell):**
  ```powershell
  .venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 2. Install Dependencies

Install the project dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to a new `.env` file and fill in your credentials:

```bash
copy .env.example .env     # On Windows
# cp .env.example .env     # On macOS/Linux
```

Update your `.env` file with:
- `LLM_API_KEY`: Your LLM provider API key.
- `MONGODB_URI`: Your MongoDB connection string.
- Desired model names and configuration settings.

### 4. Running the Project

- **Run the Web API & Interface (FastAPI):**
  ```bash
  uvicorn app:app --reload
  ```
  Then open your browser to `http://localhost:8000` to interact with the web chat interface.

- **Run tests:**
  ```bash
  pytest
  ```
- **Run the CLI / Interactive entry point:**
  ```bash
  python main.py
  ```
- **Run the Evaluation Harness:**
  ```bash
  python evals/run_evals.py --n 3
  ```

---

## What Each File Does

| File / Folder | Purpose in Plain Language |
|---|---|
| `ARCHITECTURE.md` | The core design document and single source of truth for architecture, golden rules, states, and data models. |
| `README.md` | Beginner guide explaining what the bot does, how to set it up, and what every file is for. |
| `requirements.txt` | Lists Python library dependencies (FastAPI, uvicorn, Pydantic v2, pymongo, httpx, python-dotenv, pytest). |
| `.env.example` | Template containing all required environment variables and their default settings. |
| `.gitignore` | Tells Git which files to ignore (like `.venv`, secrets in `.env`, and cache files). |
| `main.py` | Command-line entry point to launch and test the chatbot interactively. |
| `app.py` | FastAPI application with session management, SSE streaming, authentication, and concurrency serialization. |
| `web/index.html` | Clean, mobile-friendly single-page chat interface with consent verification and "Delete my data". |
| `config/helplines.json` | Contains verified crisis, emergency, and support helpline telephone numbers. |
| `prompts/understand.txt` | Prompt instructions for the fast model to extract facts and user intent. |
| `prompts/affect.txt` | Prompt instructions for the fast model to detect user emotions and crisis flags. |
| `prompts/gap_analyzer.txt` | Prompt instructions for the senior lawyer persona to identify missing facts and gaps. |
| `prompts/talker.txt` | Prompt instructions for the smart model to write warm, conversational replies. |
| `prompts/confirm_summary.txt` | Prompt instructions for the smart model to summarize facts before confirmation. |
| `prompts/rewriter.txt` | Prompt instructions for compiling confirmed facts into an external service query. |
| `prompts/presenter.txt` | Prompt instructions for presenting legal answers in actionable, readable, or informative modes. |
| `prompts/faithfulness.txt` | Prompt instructions for checking that the chatbot's explanation matches the legal service answer. |
| `prompts/extractor.txt` | Prompt instructions for the extractor LLM to output intent and structured case patches. |
| `src/__init__.py` | Marks `src` as an importable Python package. |
| `src/config.py` | Reads and validates environment variables from `.env` using Pydantic settings. |
| `src/llm.py` | Wrapper for LLM calls with automatic retry, latency logging, and validation error handling. |
| `src/db.py` | Manages connection to MongoDB collections (sessions, messages, case versions, logs) and TTL indexes. |
| `src/schemas.py` | Defines Pydantic data schemas for messages, case patches, affect readings, gaps, and logs. |
| `src/prompts.py` | Loads `.txt` files from `/prompts` and computes version hashes for audit logging. |
| `src/messages.py` | Stores pre-approved, hard-coded safety, crisis, and fallback text messages. |
| `src/case_ops.py` | Applies validated patches to the user case file and records immutable versions. |
| `src/understand.py` | Implements the fast-tier call to extract case patches, intent, and language. |
| `src/affect.py` | Implements the fast-tier emotion and crisis detector (affect signals influence tone only). |
| `src/safety.py` | Code-based safety filter that immediately intercepts emergencies, self-harm, or abuse keywords. |
| `src/gaps.py` | Discovers missing case elements (gaps) to guide the conversation toward readiness. |
| `src/readiness.py` | Pure code evaluation of case readiness, gap tracking, and assumption recording. |
| `src/talker.py` | Streams conversational replies with numbered gap blocks using the smart LLM tier. |
| `src/pipeline.py` | Coordinates per-turn pipeline steps (safety -> understand + affect -> patch -> gaps -> readiness -> talker). |
| `src/confirm.py` | Handles presenting case recap to the user and capturing their confirmation or correction. |
| `src/service_client.py` | Sends the confirmed query to the external legal answering service (or local mock service). |
| `src/presenter.py` | Formats and delivers the legal service's answer to the user in their chosen mode. |
| `src/redact.py` | Redacts sensitive personally identifiable information (PII) before logging. |
| `evals/personas/` | Directory holding simulated user profiles for evaluating conversation flow and safety. |
| `evals/run_evals.py` | Script that executes automated persona evaluations against the chatbot. |
| `evals/thresholds.json` | Benchmark thresholds for safety, accuracy, and turn limits in evaluations. |
| `tests/__init__.py` | Marks `tests` as a Python test package. |
| `tests/conftest.py` | Pytest fixtures and mock environment variable configuration. |
| `tests/test_config.py` | Tests for settings loading and missing key validation. |
| `tests/test_db.py` | Tests for database helpers and mongomock fixtures. |
| `tests/test_llm.py` | Tests for FakeLLM, Pydantic validation, and streaming. |
| `tests/test_prompts.py` | Tests for prompt loading, hash computation, and variable rendering. |
| `tests/test_talker.py` | Tests for talker prompt assembly and `<user_message>` tag wrapping. |
| `tests/test_case_ops.py` | Tests for case operations, patch application, deduplication, and anti-hallucination downgrading. |
| `tests/test_understand.py` | Tests for the understand step using extractor prompt and FakeLLM. |
| `tests/test_affect.py` | Tests for affect readings and style hint derivation rules. |
| `tests/test_safety.py` | Over 55-sentence benchmark tests for keyword safety, false positive resistance, and priority order. |
| `tests/test_pipeline.py` | Pipeline integration tests verifying parallel execution, crisis escalation, and subsequent turn warnings. |
| `tests/test_gaps.py` | Tests for gap analyzer prompt execution and pending service gap injection. |
| `tests/test_readiness.py` | Tests for readiness truth table, hard core constraints, and asked_counts logic. |
| `tests/test_tenancy_simulation.py` | Multi-turn tenancy dispute conversation reaching READY_FOR_CONFIRMATION within 4-8 turns. |
| `tests/test_confirm_service_presenter.py` | Tests for confirmation, service call, presenter guards, faithfulness checks, and state transitions. |

