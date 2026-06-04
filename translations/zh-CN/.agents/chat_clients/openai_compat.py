"""
OpenAI-compatible chat client for providers that only implement the legacy
`/v1/chat/completions` endpoint (e.g. MiniMax).

The Microsoft Agent Framework's `OpenAIChatClient` is hard-wired to OpenAI's
new `/v1/responses` endpoint, which many third-party OpenAI-compatible
providers do not implement yet — they still answer `/v1/chat/completions`.
This module implements a `BaseChatClient` subclass that talks to the legacy
endpoint directly via `httpx`, so the same `chat_client.as_agent(...)` code
keeps working with those providers.

Exports:
    OpenAICompatChatClient: drop-in replacement for `OpenAIChatClient` that
        targets `/v1/chat/completions`.

Environment variables consumed (all optional, but api_key is required to use
this client):
    <PROVIDER>_API_KEY          API key, e.g. `MINIMAX_API_KEY`.
    <PROVIDER>_BASE_URL         Base URL including scheme + host + /v1,
                                e.g. `https://api.minimax.io/v1`.
    <PROVIDER>_MODEL_ID         Model name, e.g. `MiniMax-M2.7`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Awaitable, Mapping, Sequence

import httpx

from agent_framework import BaseChatClient, ChatResponse, Message
from agent_framework._types import Content
from agent_framework._tools import FunctionInvocationLayer

logger = logging.getLogger(__name__)


class OpenAICompatChatClient(FunctionInvocationLayer, BaseChatClient):
    """Minimal `BaseChatClient` implementation against
    `/v1/chat/completions`.

    Supports:
      - non-streaming chat completions
      - system / user / assistant / tool messages
      - function calling (OpenAI `tools`/`tool_choice` format)
      - Pydantic-style structured output via `response_format`
        (`{"type": "json_schema", "json_schema": {...}}` or
        `{"type": "json_object"}`)

    Streaming is intentionally not implemented — the course notebooks in this
    repo do not use it. Raising `NotImplementedError` is the safest signal
    if a caller ever asks for it.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float = 180.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__()
        self.model = model
        self._api_key = api_key
        # Strip trailing slash so we can safely join with `/chat/completions`.
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        # 0 disables retries; retry only on transient network errors
        # (timeouts, connection drops) — never on 4xx auth failures.
        self._max_retries = max(0, max_retries)

    @property
    def service_url(self) -> str:
        return self._base_url

    # ------------------------------------------------------------------ #
    # BaseChatClient contract
    # ------------------------------------------------------------------ #

    def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> Awaitable[ChatResponse]:
        if stream:
            raise NotImplementedError(
                "OpenAICompatChatClient does not implement streaming. "
                "Use non-streaming agent runs (default)."
            )
        return self._get_response(messages, options)

    async def _get_response(
        self,
        messages: Sequence[Message],
        options: Mapping[str, Any],
    ) -> ChatResponse:
        validated = await self._validate_options(options)
        payload = self._build_payload(messages, validated)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # Retry transient network failures (timeout, connection reset) with
        # linear backoff. 4xx/5xx with a body are NOT retried — they almost
        # always mean a real provider error (auth, schema, billing) and
        # would just waste time.
        attempts = self._max_retries + 1
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(
                        f"{self._base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                break  # got a response (even if 4xx) — don't retry
            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
            ) as exc:
                last_exc = exc
                if attempt >= attempts:
                    raise RuntimeError(
                        f"Chat completions request failed after {attempts} "
                        f"attempt(s) (model={self.model}): {exc}"
                    ) from exc
                backoff = min(2 ** (attempt - 1), 8)
                logger.warning(
                    "Chat completions transient error (attempt %d/%d): %s. "
                    "Retrying in %ds.",
                    attempt,
                    attempts,
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)

        if resp.status_code >= 400:
            # Bubble up provider error body verbatim — much easier to debug
            # than a generic exception.
            raise RuntimeError(
                f"Chat completions request failed "
                f"(status={resp.status_code}, model={self.model}): "
                f"{resp.text[:500]}"
            )

        data = resp.json()
        return self._parse_response(data)

    # ------------------------------------------------------------------ #
    # Payload construction: agent_framework Message -> OpenAI chat format
    # ------------------------------------------------------------------ #

    def _build_payload(
        self,
        messages: Sequence[Message],
        options: Mapping[str, Any],
    ) -> dict[str, Any]:
        chat_messages: list[dict[str, Any]] = []
        for m in messages:
            chat_messages.extend(self._convert_message(m))

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
        }

        # Temperature / max_tokens etc. come through `options` from
        # `ChatOptions`; copy over any that are non-None.
        for key in ("temperature", "top_p", "max_tokens", "stop", "seed"):
            value = options.get(key)
            if value is not None:
                payload[key] = value

        if tools := options.get("tools"):
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": _resolve_tool_parameters(t),
                    },
                }
                for t in tools
            ]
            payload["tool_choice"] = "auto"

        # Structured output: Pydantic model -> JSON schema; pre-built dict
        # passes through; `None` means no constraint.
        response_format = options.get("response_format")
        if response_format is not None:
            payload["response_format"] = self._convert_response_format(
                response_format
            )

        return payload

    def _convert_message(self, message: Message) -> list[dict[str, Any]]:
        """Convert one `Message` into one or more OpenAI chat messages.

        A single `Message` can carry multiple `Content` items (e.g. a
        reasoning block plus a function call). We split them so each
        resulting dict has the shape the provider expects.
        """
        role = message.role
        out: list[dict[str, Any]] = []

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []

        for c in message.contents or []:
            if not isinstance(c, Content):
                # Some layers may pass plain strings/dicts; coerce.
                text_parts.append(str(c))
                continue
            ctype = c.type
            if ctype == "text" and c.text:
                text_parts.append(c.text)
            elif ctype == "function_call" and c.name:
                # OpenAI tool calls live on the assistant message itself.
                args = c.arguments
                if isinstance(args, Mapping):
                    args_str = json.dumps(args, ensure_ascii=False)
                else:
                    args_str = args or "{}"
                tool_calls.append(
                    {
                        "id": c.call_id or "",
                        "type": "function",
                        "function": {
                            "name": c.name,
                            "arguments": args_str,
                        },
                    }
                )
            elif ctype == "function_result":
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": c.call_id or "",
                        "content": c.result
                        if isinstance(c.result, str)
                        else json.dumps(c.result or "", ensure_ascii=False),
                    }
                )
            # Other content types (images, files, hosted vectors, …) are
            # intentionally ignored — this client targets text + function
            # calling only, which is what the course notebooks use.

        # If we harvested tool results, they form their own messages.
        if tool_results:
            out.extend(tool_results)

        # If we harvested tool calls, they ride along on the assistant
        # message that owns them.
        assistant_msg: dict[str, Any] | None = None
        if tool_calls:
            assistant_msg = {
                "role": "assistant",
                "content": "".join(text_parts) or None,
                "tool_calls": tool_calls,
            }
        elif text_parts and role == "assistant":
            assistant_msg = {
                "role": "assistant",
                "content": "".join(text_parts),
            }

        if assistant_msg is not None:
            out.append(assistant_msg)

        # Plain text message (user / system / developer) — fold all text
        # parts into one content string, the way OpenAI expects.
        if text_parts and not tool_calls and not tool_results:
            out.append(
                {
                    "role": self._map_role(role),
                    "content": "".join(text_parts),
                }
            )

        return out

    @staticmethod
    def _map_role(role: str) -> str:
        # `developer` is OpenAI's newer alias; most providers accept
        # `system` instead. Normalize defensively.
        if role in ("system", "user", "assistant", "tool", "developer"):
            return "system" if role == "developer" else role
        return "user"

    @staticmethod
    def _convert_response_format(response_format: Any) -> dict[str, Any]:
        # Pydantic class -> JSON Schema via model_json_schema().
        if isinstance(response_format, type) and hasattr(
            response_format, "model_json_schema"
        ):
            schema = response_format.model_json_schema()
            # OpenAI strict mode requires `additionalProperties: false`
            # on every object node; emit a best-effort version.
            schema = _enforce_additional_properties_false(schema)
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema": schema,
                    "strict": True,
                },
            }
        if isinstance(response_format, Mapping):
            # Already in OpenAI shape — pass through.
            return dict(response_format)
        # Fallback: free-form JSON.
        return {"type": "json_object"}

    # ------------------------------------------------------------------ #
    # Response parsing: OpenAI chat completion -> ChatResponse
    # ------------------------------------------------------------------ #

    def _parse_response(self, data: dict[str, Any]) -> ChatResponse:
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(
                f"Provider returned no choices: {json.dumps(data)[:500]}"
            )
        choice = choices[0]
        message = choice.get("message") or {}
        finish = choice.get("finish_reason")

        contents: list[Content] = []
        if text := message.get("content"):
            contents.append(Content(type="text", text=text))
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args_raw = fn.get("arguments") or "{}"
            try:
                args: Any = json.loads(args_raw)
            except json.JSONDecodeError:
                args = args_raw
            contents.append(
                Content(
                    type="function_call",
                    call_id=tc.get("id"),
                    name=fn.get("name"),
                    arguments=args,
                )
            )

        assistant_msg = Message(
            role="assistant",
            contents=contents,
            message_id=data.get("id"),
        )

        usage = None
        if u := data.get("usage"):
            from agent_framework._types import UsageDetails

            usage = UsageDetails(
                input_token_count=u.get("prompt_tokens"),
                output_token_count=u.get("completion_tokens"),
                total_token_count=u.get("total_tokens"),
            )

        return ChatResponse(
            messages=[assistant_msg],
            response_id=data.get("id"),
            model=data.get("model") or self.model,
            finish_reason=finish,
            usage_details=usage,
            raw_representation=data,
        )


def _resolve_tool_parameters(tool: Any) -> dict[str, Any]:
    """Normalize a `FunctionTool` / `AITool`-like object into a JSON Schema dict.

    In agent_framework 1.x, `FunctionTool.parameters` is a *method* that
    returns the schema on demand, not a stored attribute. Older versions
    exposed it as a property. Some user code also passes raw schemas
    through. Handle all three without crashing on a bound-method object
    (which is not JSON-serializable).
    """
    params = getattr(tool, "parameters", None)
    if callable(params):
        params = params()
    if params is None:
        return {"type": "object", "properties": {}}
    if isinstance(params, Mapping):
        return dict(params)
    # Last-ditch: try Pydantic-style model_json_schema on the underlying fn.
    fn = getattr(tool, "func", None) or getattr(tool, "function", None)
    if fn is not None and hasattr(fn, "model_json_schema"):
        return fn.model_json_schema()
    return {"type": "object", "properties": {}}


# ---------------------------------------------------------------------- #
# Schema hardening for OpenAI strict mode
# ---------------------------------------------------------------------- #

def _enforce_additional_properties_false(node: Any) -> Any:
    """Recursively set `additionalProperties: false` on every object node.

    OpenAI's strict JSON-schema mode rejects schemas that allow extra
    properties. Pydantic v2 already emits `additionalProperties: false`
    by default, but nested `$defs` and array item schemas can still slip
    through. A defensive recursive pass is cheap and removes a class of
    400 errors.
    """
    if isinstance(node, dict):
        if node.get("type") == "object":
            node.setdefault("additionalProperties", False)
        for v in node.values():
            _enforce_additional_properties_false(v)
    elif isinstance(node, list):
        for v in node:
            _enforce_additional_properties_false(v)
    return node


# ---------------------------------------------------------------------- #
# Convenience factory: build a client straight from environment variables
# ---------------------------------------------------------------------- #

def from_env(prefix: str) -> OpenAICompatChatClient:
    """Build a client from `<PREFIX>_API_KEY/_BASE_URL/_MODEL_ID` env vars.

    Example:
        from translations.zh_CN.agents.chat_clients.openai_compat import from_env
        client = from_env("MINIMAX")
    """
    api_key = os.environ[f"{prefix}_API_KEY"]
    base_url = os.environ.get(
        f"{prefix}_BASE_URL", "https://api.minimax.io/v1"
    )
    model = os.environ.get(f"{prefix}_MODEL_ID", "MiniMax-M2.7")
    return OpenAICompatChatClient(
        model=model, api_key=api_key, base_url=base_url
    )
