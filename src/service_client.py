"""
src/service_client.py - Client for sending confirmed queries to external legal service or mock service.

Supports:
- Real mode: HTTP POST to SERVICE_URL with SERVICE_API_KEY, 60s timeout, 1 retry.
- Mock mode (USE_MOCK_SERVICE=true): Returns realistic canned tenancy answers with sections,
  or returns needs_more_info with missing_info=["Do you have a written rental agreement?"]
  when the query contains no documents.
- Logs every call to the service_calls MongoDB collection.
"""

import asyncio
import time

import httpx

from src.config import get_settings
from src.db import log_service_call
from src.schemas import ServiceQuery, ServiceResponse


class ServiceClient:
    """External legal answering service client."""

    def __init__(
        self,
        service_url: str | None = None,
        api_key: str | None = None,
        use_mock: bool | None = None,
    ):
        settings = get_settings()
        self.service_url = (service_url or settings.service_url).rstrip("/")
        self.api_key = api_key or settings.service_api_key
        self.use_mock = settings.use_mock_service if use_mock is None else use_mock
        self.timeout = httpx.Timeout(60.0, connect=10.0)

    async def ask(
        self,
        query: ServiceQuery,
        session_id: str | None = None,
    ) -> ServiceResponse:
        """
        Sends a ServiceQuery to the legal answering service and returns a ServiceResponse.
        Logs details to service_calls collection.
        """
        start_time = time.time()
        sid = session_id or "unknown"
        request_data = query.model_dump()

        if self.use_mock:
            resp = self._mock_ask(query)
            latency_ms = int((time.time() - start_time) * 1000)
            await log_service_call(
                session_id=sid,
                request=request_data,
                response=resp.model_dump(),
                latency_ms=latency_ms,
                status=resp.status,
            )
            return resp

        # Real mode with 1 retry
        retries = 1
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.service_url}/answer",
                        json=request_data,
                        headers=headers,
                    )
                    response.raise_for_status()
                    data = response.json()
                    resp = ServiceResponse.model_validate(data)
                    latency_ms = int((time.time() - start_time) * 1000)
                    await log_service_call(
                        session_id=sid,
                        request=request_data,
                        response=resp.model_dump(),
                        latency_ms=latency_ms,
                        status=resp.status,
                    )
                    return resp
            except Exception as e:
                last_error = e
                if attempt < retries:
                    await asyncio.sleep(1.0)

        # On persistent failure
        latency_ms = int((time.time() - start_time) * 1000)
        err_resp = ServiceResponse(
            status="error",
            answer=None,
            missing_info=[],
            citations=[],
        )
        await log_service_call(
            session_id=sid,
            request=request_data,
            response={"error": str(last_error), "status": "error"},
            latency_ms=latency_ms,
            status="error",
        )
        return err_resp

    def _mock_ask(self, query: ServiceQuery) -> ServiceResponse:
        """Realistic mock answering service for testing and development."""
        # If no documents are listed in the query, trigger needs_more_info
        if not query.documents:
            return ServiceResponse(
                status="needs_more_info",
                answer=None,
                missing_info=["Do you have a written rental agreement?"],
                citations=[],
            )

        summary = (
            "A landlord cannot unilaterally evict a tenant without serving an official notice and "
            "obtaining an order from the tenancy tribunal. Tenants have the right to dispute any eviction notice."
        )
        next_steps = (
            "1. Check the official termination date on your notice.\n"
            "2. File a dispute application with the tenancy tribunal before the deadline.\n"
            "3. Continue paying your lawful rent and keep written records of all transactions."
        )
        caveats = (
            "Statutory dispute deadlines are strictly enforced. Missing a dispute deadline may result in an "
            "automatic eviction order. Please consult a qualified legal practitioner or community legal clinic."
        )
        full_answer = f"{summary}\n\nWhat you can do next:\n{next_steps}\n\nKeep in mind:\n{caveats}"

        return ServiceResponse(
            status="answered",
            answer=full_answer,
            sections={
                "summary": summary,
                "next_steps": next_steps,
                "caveats": caveats,
            },
            missing_info=[],
            citations=["Residential Tenancies Act"],
        )
