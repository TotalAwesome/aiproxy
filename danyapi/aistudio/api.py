from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import HTTPException

from ..accounts import account_lock
from ..api.retry import RETRYABLE_HTTP_STATUSES
from ..api.shaping import _apply_limits
from ..api.sse import _sse, _stream_error_sse
from ..config import settings
from ..sseutil import StreamStopFilter, split_stop
from ..tokens import StreamBudget, estimate_tokens, trim_to_tokens
from ..tools import DsmlFilter, strip_dsml
from ..usage import record_usage_dict
from .client import GENERATE_URL, AistudioError, StreamParser, build_contents, build_request, event_parts

log = logging.getLogger("danyapi.aistudio.api")

DONE_LINE = "data: [DONE]\n\n"
PROVIDER = "aistudio"
DEFAULT_MODEL = "gemini-flash-latest"

TOKEN_ATTEMPTS = 2


def _status_for(error: AistudioError) -> int:
    if error.status in (401, 403):
        return error.status
    if error.status in RETRYABLE_HTTP_STATUSES:
        return 502
    if 400 <= error.status < 500:
        return 400
    return 502


def _http_error(error: AistudioError) -> HTTPException:
    return HTTPException(_status_for(error), error.message)


def _binding_for(contents: list[Any]) -> str:
    texts: list[str] = []
    for content in contents:
        parts = content[0] if isinstance(content, list) and content else []
        for part in parts or []:
            if isinstance(part, list) and len(part) > 1 and isinstance(part[1], str):
                texts.append(part[1])
    return hashlib.sha256(" ".join(texts).encode("utf-8")).hexdigest()


def _usage_for(prompt_text: str, content: str) -> dict:
    prompt_tokens = estimate_tokens(prompt_text)
    completion_tokens = estimate_tokens(content)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }


async def _mint(account: Any, binding: str) -> str:
    try:
        return await account.browser.mint_token(binding)
    except Exception as exc:
        raise HTTPException(502, f"aistudio botguard failed: {exc}") from exc


async def _open_stream(account: Any, contents: list[Any], model: str, limits: dict[str, Any]):
    prompt_text = " ".join(
        part[1]
        for content in contents
        for part in (content[0] if isinstance(content, list) and content else [])
        if isinstance(part, list) and len(part) > 1 and isinstance(part[1], str)
    )
    binding = _binding_for(contents)
    last_error: AistudioError | None = None
    for attempt in range(TOKEN_ATTEMPTS):
        await account.refresh_cookies()
        token = await _mint(account, binding)
        body = build_request(
            model,
            contents,
            token,
            max_tokens=limits.get("max_tokens"),
            temperature=limits.get("temperature"),
            top_p=limits.get("top_p"),
            top_k=limits.get("top_k"),
        )
        try:
            response = await account.transport.stream(GENERATE_URL, body)
        except AistudioError as exc:
            last_error = exc
            if exc.is_auth and attempt + 1 < TOKEN_ATTEMPTS:
                await asyncio.sleep(1.0)
                continue
            raise _http_error(exc) from exc
        return response, prompt_text
    if last_error is not None:
        raise _http_error(last_error)
    raise HTTPException(502, "aistudio request failed")


async def _iter_text(account: Any, contents: list[Any], model: str, limits: dict[str, Any]) -> AsyncIterator[tuple[str, str]]:
    response = None
    parser = StreamParser()
    try:
        response, _ = await _open_stream(account, contents, model, limits)
        async for chunk in response.aiter_stream():
            for event in parser.feed(chunk):
                for kind, text in event_parts(event):
                    yield kind, text
    except AistudioError as exc:
        raise _http_error(exc) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"aistudio transport error: {exc}") from exc
    finally:
        if response is not None:
            await response.aclose()


def _collect_limits(max_tokens: int | None, temperature: float | None, top_p: float | None, top_k: int | None) -> dict[str, Any]:
    return {"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p, "top_k": top_k}


async def collect_non_stream(
    account: Any,
    messages: Any,
    model: str = DEFAULT_MODEL,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    stop: Any = None,
    max_tokens: int | None = None,
    user: str | None = None,
    session_id: str | None = None,
) -> dict:
    contents, _system = build_contents(messages)
    limits = _collect_limits(max_tokens, temperature, top_p, top_k)
    content: list[str] = []
    async with account_lock(account.sem, settings.acquire_timeout):
        async for kind, text in _iter_text(account, contents, model, limits):
            if kind == "content":
                content.append(text)
    joined = "".join(content)
    text, limit_finish = _apply_limits(strip_dsml(joined), max_tokens, stop)
    finish = "length" if limit_finish == "length" else "stop"
    prompt_text = " ".join(
        part[1]
        for item in contents
        for part in (item[0] if isinstance(item, list) and item else [])
        if isinstance(part, list) and len(part) > 1 and isinstance(part[1], str)
    )
    usage = _usage_for(prompt_text, text)
    record_usage_dict(PROVIDER, model, usage, user=user, session_id=session_id)
    message: dict[str, Any] = {"role": "assistant", "content": text}
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": "fp_aistudio_chat",
        "choices": [{"index": 0, "message": message, "finish_reason": finish, "logprobs": None}],
        "usage": usage,
        "session_id": session_id,
    }


def _chunk(chunk_id: str, created: int, model: str, delta: dict, finish: str | None = None, usage: dict | None = None) -> str:
    payload: dict[str, Any] = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage is not None:
        payload["usage"] = usage
    return _sse(payload)


async def stream_openai(
    account: Any,
    messages: Any,
    model: str = DEFAULT_MODEL,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    stop: Any = None,
    max_tokens: int | None = None,
    include_usage: bool = False,
    user: str | None = None,
    session_id: str | None = None,
) -> AsyncIterator[str]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    contents, _system = build_contents(messages)
    limits = _collect_limits(max_tokens, temperature, top_p, top_k)
    content: list[str] = []
    emitted = False
    finish = "stop"
    stop_markers = split_stop(stop)
    stop_filter = StreamStopFilter(stop_markers) if stop_markers else None
    budget = StreamBudget(max_tokens, trim_to_tokens)
    stop_hit = False
    content_filter = DsmlFilter()
    prompt_text = " ".join(
        part[1]
        for item in contents
        for part in (item[0] if isinstance(item, list) and item else [])
        if isinstance(part, list) and len(part) > 1 and isinstance(part[1], str)
    )

    try:
        async with contextlib.AsyncExitStack() as stack:
            await stack.enter_async_context(account_lock(account.sem, None))
            yield _chunk(chunk_id, created, model, {"role": "assistant"})
            async for kind, text in _iter_text(account, contents, model, limits):
                if kind != "content":
                    continue
                piece = content_filter.feed(text)
                if stop_filter is not None and not stop_hit:
                    piece, hit = stop_filter.feed(piece)
                    stop_hit = stop_hit or hit
                elif stop_hit:
                    piece = ""
                piece = budget.feed(piece)
                if piece:
                    content.append(piece)
                    emitted = True
                    yield _chunk(chunk_id, created, model, {"content": piece})
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        record_usage_dict(PROVIDER, model, _usage_for(prompt_text, "".join(content)), user=user, session_id=session_id)
        for line in _stream_error_sse(chunk_id, created, model, detail, session_id):
            yield line
        return

    tail = content_filter.flush()
    if stop_filter is not None and not stop_hit:
        tail, hit = stop_filter.feed(tail)
        stop_hit = stop_hit or hit
        if stop_hit:
            tail = ""
        else:
            tail += stop_filter.flush()
    else:
        tail = "" if stop_hit else tail
    tail = budget.feed(tail)
    if tail:
        content.append(tail)
        emitted = True
        yield _chunk(chunk_id, created, model, {"content": tail})
    if budget.done:
        finish = "length"
    if not emitted:
        yield _chunk(chunk_id, created, model, {"content": ""})
    yield _chunk(chunk_id, created, model, {}, finish)
    usage = _usage_for(prompt_text, "".join(content))
    record_usage_dict(PROVIDER, model, usage, user=user, session_id=session_id)
    if include_usage:
        yield _chunk(chunk_id, created, model, {}, None, usage=usage)
    yield DONE_LINE
