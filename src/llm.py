"""
src/llm.py - LLM client wrapper supporting text, structured JSON, and streaming.

Provides:
- LLM: Async client using httpx with 30s timeout, exponential backoff retries on 429/5xx,
  Pydantic validation with 1-retry fallback for JSON, and comprehensive logging to llm_logs.
- FakeLLM: Mock client with identical interface for deterministic testing without external API calls.
"""

import asyncio
from collections.abc import AsyncGenerator
import json
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from src.config import get_settings
from src.db import log_llm_call
from src.prompts import load_prompt

T = TypeVar("T", bound=BaseModel)


class LLMOutputError(Exception):
    """Raised when an LLM fails to return valid output matching the requested schema."""
    pass


class BaseLLM:
    """Abstract base class defining the common LLM interface."""

    def _resolve_model(self, tier: str) -> str:
        settings = get_settings()
        if tier == "fast":
            return settings.fast_model
        elif tier == "smart":
            return settings.smart_model
        return tier

    def _get_prompt_version(self, prompt_name: str) -> str:
        try:
            _, version = load_prompt(prompt_name)
            return version
        except Exception:
            return "unknown"


class LLM(BaseLLM):
    """
    Production-ready asynchronous LLM client.
    Supports OpenAI-compatible APIs via httpx.
    """

    def __init__(self, api_key: str | None = None, base_url: str = "https://api.openai.com/v1"):
        settings = get_settings()
        self.api_key = api_key or settings.llm_api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(30.0, connect=10.0)

    async def _request_with_retry(self, payload: dict, is_stream: bool = False) -> Any:
        """
        Executes HTTP POST request with up to 2 retries on rate limit (429)
        and server errors (5xx) using exponential backoff.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        retries = 2
        backoff = 1.0

        for attempt in range(retries + 1):
            try:
                client = httpx.AsyncClient(timeout=self.timeout)
                if is_stream:
                    request = client.build_request("POST", url, json=payload, headers=headers)
                    response = await client.send(request, stream=True)
                else:
                    async with client:
                        response = await client.post(url, json=payload, headers=headers)

                if response.status_code in (429, 500, 502, 503, 504):
                    if attempt < retries:
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue
                response.raise_for_status()
                return response
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                if attempt < retries:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                else:
                    raise e

    async def text(
        self,
        prompt_name: str,
        system: str,
        messages: list[dict],
        tier: str,
        temperature: float = 0.3,
        session_id: str | None = None,
    ) -> str:
        """Calls the LLM and returns the response as plain text."""
        model = self._resolve_model(tier)
        prompt_version = self._get_prompt_version(prompt_name)
        start_time = time.time()

        formatted_messages = [{"role": "system", "content": system}] + messages
        input_chars = sum(len(m.get("content", "")) for m in formatted_messages)

        payload = {
            "model": model,
            "messages": formatted_messages,
            "temperature": temperature,
        }

        tokens_in = 0
        tokens_out = 0
        output_text = ""
        success = False
        error_msg: str | None = None

        try:
            response = await self._request_with_retry(payload)
            data = response.json()
            output_text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            tokens_in = usage.get("prompt_tokens", input_chars // 4)
            tokens_out = usage.get("completion_tokens", len(output_text) // 4)
            success = True
            return output_text
        except Exception as e:
            error_msg = str(e)
            raise e
        finally:
            latency_ms = int((time.time() - start_time) * 1000)
            await log_llm_call(
                session_id=session_id,
                prompt_name=prompt_name,
                prompt_version=prompt_version,
                model=model,
                input_chars=input_chars,
                output=output_text,
                latency_ms=latency_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                success=success,
                error=error_msg,
            )

    async def json(
        self,
        prompt_name: str,
        system: str,
        messages: list[dict],
        schema: type[T],
        tier: str,
        session_id: str | None = None,
    ) -> T:
        """
        Calls the LLM requesting structured JSON output matching the Pydantic schema.
        Validates with Pydantic. If validation fails, retries ONCE appending the error.
        If it fails again, raises LLMOutputError.
        """
        model = self._resolve_model(tier)
        prompt_version = self._get_prompt_version(prompt_name)
        start_time = time.time()

        # Add schema instruction into system message
        schema_json_str = json.dumps(schema.model_json_schema())
        enhanced_system = (
            f"{system}\n\nYou must respond strictly with valid JSON conforming to this JSON Schema:\n"
            f"{schema_json_str}"
        )

        current_messages = list(messages)
        last_raw_output = ""
        error_msg: str | None = None
        success = False
        tokens_in = 0
        tokens_out = 0
        input_chars = len(enhanced_system) + sum(len(m.get("content", "")) for m in current_messages)

        for attempt in range(2):
            formatted_messages = [{"role": "system", "content": enhanced_system}] + current_messages
            payload = {
                "model": model,
                "messages": formatted_messages,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            }

            try:
                response = await self._request_with_retry(payload)
                data = response.json()
                last_raw_output = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                tokens_in += usage.get("prompt_tokens", input_chars // 4)
                tokens_out += usage.get("completion_tokens", len(last_raw_output) // 4)

                validated_instance = schema.model_validate_json(last_raw_output)
                success = True
                latency_ms = int((time.time() - start_time) * 1000)
                await log_llm_call(
                    session_id=session_id,
                    prompt_name=prompt_name,
                    prompt_version=prompt_version,
                    model=model,
                    input_chars=input_chars,
                    output=last_raw_output,
                    latency_ms=latency_ms,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    success=True,
                )
                return validated_instance
            except (ValidationError, json.JSONDecodeError, KeyError) as val_err:
                error_msg = str(val_err)
                if attempt == 0:
                    # Retry ONCE appending the error message
                    current_messages.append({"role": "assistant", "content": last_raw_output})
                    current_messages.append({
                        "role": "user",
                        "content": f"The response was invalid: {error_msg}. Please correct it and output strictly valid JSON matching the schema.",
                    })
                    continue
                else:
                    latency_ms = int((time.time() - start_time) * 1000)
                    await log_llm_call(
                        session_id=session_id,
                        prompt_name=prompt_name,
                        prompt_version=prompt_version,
                        model=model,
                        input_chars=input_chars,
                        output=last_raw_output,
                        latency_ms=latency_ms,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        success=False,
                        error=error_msg,
                    )
                    raise LLMOutputError(
                        f"LLM output failed validation for {schema.__name__} after retry: {error_msg}"
                    ) from val_err
            except Exception as e:
                error_msg = str(e)
                latency_ms = int((time.time() - start_time) * 1000)
                await log_llm_call(
                    session_id=session_id,
                    prompt_name=prompt_name,
                    prompt_version=prompt_version,
                    model=model,
                    input_chars=input_chars,
                    output=last_raw_output,
                    latency_ms=latency_ms,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    success=False,
                    error=error_msg,
                )
                raise e

        raise LLMOutputError("LLM call failed.")

    async def stream(
        self,
        prompt_name: str,
        system: str,
        messages: list[dict],
        tier: str,
        session_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Streams the LLM response chunk by chunk as an async generator."""
        model = self._resolve_model(tier)
        prompt_version = self._get_prompt_version(prompt_name)
        start_time = time.time()

        formatted_messages = [{"role": "system", "content": system}] + messages
        input_chars = sum(len(m.get("content", "")) for m in formatted_messages)

        payload = {
            "model": model,
            "messages": formatted_messages,
            "stream": True,
        }

        full_output_parts: list[str] = []
        success = False
        error_msg: str | None = None

        try:
            response = await self._request_with_retry(payload, is_stream=True)
            async with response:
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        line_data = line[len("data: "):].strip()
                        if line_data == "[DONE]":
                            break
                        try:
                            chunk_json = json.loads(line_data)
                            delta = chunk_json["choices"][0]["delta"]
                            content_piece = delta.get("content", "")
                            if content_piece:
                                full_output_parts.append(content_piece)
                                yield content_piece
                        except json.JSONDecodeError:
                            continue
            success = True
        except Exception as e:
            error_msg = str(e)
            raise e
        finally:
            output_text = "".join(full_output_parts)
            latency_ms = int((time.time() - start_time) * 1000)
            await log_llm_call(
                session_id=session_id,
                prompt_name=prompt_name,
                prompt_version=prompt_version,
                model=model,
                input_chars=input_chars,
                output=output_text,
                latency_ms=latency_ms,
                tokens_in=input_chars // 4,
                tokens_out=len(output_text) // 4,
                success=success,
                error=error_msg,
            )


class FakeLLM(BaseLLM):
    """
    Mock LLM providing identical interface with deterministic canned responses.
    Allows all tests to run without network connectivity or API keys.
    """

    def __init__(self, responses: dict[str, Any] | None = None):
        self.responses = responses or {}
        self.call_history: list[dict] = []

    async def text(
        self,
        prompt_name: str,
        system: str,
        messages: list[dict],
        tier: str,
        temperature: float = 0.3,
        session_id: str | None = None,
    ) -> str:
        model = self._resolve_model(tier)
        prompt_version = self._get_prompt_version(prompt_name)
        start_time = time.time()

        input_chars = len(system) + sum(len(m.get("content", "")) for m in messages)
        raw_response = self.responses.get(prompt_name, "Mock response for " + prompt_name)
        output_text = str(raw_response)

        self.call_history.append({
            "call_type": "text",
            "prompt_name": prompt_name,
            "system": system,
            "messages": messages,
            "tier": tier,
            "session_id": session_id,
        })

        latency_ms = int((time.time() - start_time) * 1000)
        await log_llm_call(
            session_id=session_id,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            model=model,
            input_chars=input_chars,
            output=output_text,
            latency_ms=latency_ms,
            tokens_in=input_chars // 4,
            tokens_out=len(output_text) // 4,
            success=True,
        )
        return output_text

    async def json(
        self,
        prompt_name: str,
        system: str,
        messages: list[dict],
        schema: type[T],
        tier: str,
        session_id: str | None = None,
    ) -> T:
        model = self._resolve_model(tier)
        prompt_version = self._get_prompt_version(prompt_name)
        start_time = time.time()

        input_chars = len(system) + sum(len(m.get("content", "")) for m in messages)
        canned = self.responses.get(prompt_name)

        self.call_history.append({
            "call_type": "json",
            "prompt_name": prompt_name,
            "system": system,
            "messages": messages,
            "tier": tier,
            "session_id": session_id,
        })

        # Try validating the canned response
        try:
            if isinstance(canned, schema):
                validated = canned
            elif isinstance(canned, dict):
                validated = schema.model_validate(canned)
            elif isinstance(canned, str):
                validated = schema.model_validate_json(canned)
            elif canned is None:
                # Attempt to build an empty/default schema instance
                validated = schema.model_validate({})
            else:
                validated = schema.model_validate(canned)

            output_text = validated.model_dump_json()
            latency_ms = int((time.time() - start_time) * 1000)
            await log_llm_call(
                session_id=session_id,
                prompt_name=prompt_name,
                prompt_version=prompt_version,
                model=model,
                input_chars=input_chars,
                output=output_text,
                latency_ms=latency_ms,
                tokens_in=input_chars // 4,
                tokens_out=len(output_text) // 4,
                success=True,
            )
            return validated
        except (ValidationError, TypeError) as err:
            output_text = str(canned)
            latency_ms = int((time.time() - start_time) * 1000)
            await log_llm_call(
                session_id=session_id,
                prompt_name=prompt_name,
                prompt_version=prompt_version,
                model=model,
                input_chars=input_chars,
                output=output_text,
                latency_ms=latency_ms,
                tokens_in=input_chars // 4,
                tokens_out=len(output_text) // 4,
                success=False,
                error=str(err),
            )
            raise LLMOutputError(f"FakeLLM validation error for {schema.__name__}: {err}") from err

    async def stream(
        self,
        prompt_name: str,
        system: str,
        messages: list[dict],
        tier: str,
        session_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        model = self._resolve_model(tier)
        prompt_version = self._get_prompt_version(prompt_name)
        start_time = time.time()

        input_chars = len(system) + sum(len(m.get("content", "")) for m in messages)
        raw_response = self.responses.get(prompt_name, "Mock stream response for " + prompt_name)
        full_text = str(raw_response)

        self.call_history.append({
            "call_type": "stream",
            "prompt_name": prompt_name,
            "system": system,
            "messages": messages,
            "tier": tier,
            "session_id": session_id,
        })

        words = full_text.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == len(words) - 1 else word + " "
            await asyncio.sleep(0.01)
            yield chunk

        latency_ms = int((time.time() - start_time) * 1000)
        await log_llm_call(
            session_id=session_id,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            model=model,
            input_chars=input_chars,
            output=full_text,
            latency_ms=latency_ms,
            tokens_in=input_chars // 4,
            tokens_out=len(full_text) // 4,
            success=True,
        )
