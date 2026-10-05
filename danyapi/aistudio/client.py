from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
import uuid
from collections.abc import Iterator
from typing import Any

import httpcore

from .doh import DohNetworkBackend, DohResolver, build_ssl_context

log = logging.getLogger("danyapi.aistudio.client")

GENERATE_URL = "https://alkalimakersuite-pa.clients6.google.com/$rpc/google.internal.alkali.applications.makersuite.v1.MakerSuiteService/GenerateContent"
LIST_MODELS_URL = "https://alkalimakersuite-pa.clients6.google.com/$rpc/google.internal.alkali.applications.makersuite.v1.MakerSuiteService/ListModels"
MODEL_PREFIX = "models/"
WEB_API_KEY = "AIzaSyDdP816MREB3SkjZO04QXbjsigfcI0GWOs"
STUDIO_ORIGIN = "https://aistudio.google.com"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36"
CLIENT_EXT = "CAASA1JGUhgBMAE4BEAAUARYAWICRlJwAHgBkAEAmAEB"

SAFETY_SETTINGS = [[None, None, 7, 5], [None, None, 8, 5], [None, None, 9, 5], [None, None, 10, 5]]
BASE_CONFIG: list[Any] = [None, None, None, 65536, 1, 0.95, 64, None, None, None, None, None, None, 1, None, None, [1, None, None, 2]]
CONFIG_MAX_TOKENS = 3
CONFIG_TEMPERATURE = 4
CONFIG_TOP_P = 5
CONFIG_TOP_K = 6

CHAT_ACTIONS = frozenset({"generateContent", "countTokens"})


class AistudioError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message

    @property
    def is_auth(self) -> bool:
        return self.status in (401, 403)

    @property
    def is_retryable(self) -> bool:
        return self.status in (429, 500, 502, 503, 504)


def _visit_id() -> str:
    raw = uuid.uuid4().bytes
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return f"v1_{encoded}"


def parse_cookies(raw: str) -> dict[str, str]:
    jar: dict[str, str] = {}
    for part in raw.split(";"):
        name, separator, value = part.strip().partition("=")
        if separator and name:
            jar[name.strip()] = value.strip()
    return jar


def auth_header(cookies: dict[str, str], timestamp: int | None = None) -> str:
    sapisid = cookies.get("SAPISID") or cookies.get("__Secure-1PAPISID") or cookies.get("APISID") or ""
    if not sapisid:
        raise AistudioError(401, "aistudio cookies have no SAPISID")
    ts = int(time.time()) if timestamp is None else timestamp
    digest = hashlib.sha1(f"{ts} {sapisid} {STUDIO_ORIGIN}".encode()).hexdigest()
    return f"SAPISIDHASH {ts}_{digest}"


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in ("text", "input_text"):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _part(text: str) -> list[Any]:
    return [None, text]


def build_contents(messages: Any) -> tuple[list[Any], list[str]]:
    system_chunks: list[str] = []
    contents: list[Any] = []
    for message in messages or []:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        if role in ("system", "developer"):
            text = _text_of(content).strip()
            if text:
                system_chunks.append(text)
            continue
        if role in ("tool", "function"):
            name = getattr(message, "name", None) or "tool"
            contents.append([[_part(f"[tool result for {name}]\n{_text_of(content)}")], "user"])
            continue
        text = _text_of(content)
        if not text:
            continue
        contents.append([[_part(text)], "model" if role == "assistant" else "user"])
    if not contents:
        contents = [[[_part("Hello")], "user"]]
    if system_chunks:
        lead = "Follow these instructions for the rest of the conversation:\n" + "\n\n".join(system_chunks)
        first_parts = contents[0][0]
        first_role = contents[0][1]
        contents[0] = [[_part(f"{lead}\n\n{first_parts[0][1]}")], first_role]
    return contents, system_chunks


def build_request(
    model: str,
    contents: list[Any],
    token: str,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> bytes:
    config = list(BASE_CONFIG)
    if max_tokens is not None:
        config[CONFIG_MAX_TOKENS] = max_tokens
    if temperature is not None:
        config[CONFIG_TEMPERATURE] = temperature
    if top_p is not None:
        config[CONFIG_TOP_P] = top_p
    if top_k is not None:
        config[CONFIG_TOP_K] = top_k
    model_id = model if model.startswith(MODEL_PREFIX) else f"{MODEL_PREFIX}{model}"
    return json.dumps([model_id, contents, SAFETY_SETTINGS, config, token], separators=(",", ":")).encode()


def build_list_models_request() -> bytes:
    return b"[]"


class AistudioTransport:
    def __init__(self, cookies: str, *, doh_url: str = "", timeout: float = 60.0) -> None:
        self.timeout = timeout
        self.cookies = parse_cookies(cookies)
        self.visit_id = _visit_id()
        backend = None
        if doh_url:
            backend = DohNetworkBackend(DohResolver(doh_url))
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=build_ssl_context(),
            network_backend=backend,
            http1=True,
            http2=False,
            max_connections=8,
            max_keepalive_connections=4,
            keepalive_expiry=30.0,
        )

    def update_cookies(self, raw: str) -> None:
        self.cookies = parse_cookies(raw)

    def _headers(self, url: str, body: bytes) -> list[tuple[bytes, bytes]]:
        host = url.split("/", 3)[2]
        headers = [
            (b"host", host.encode()),
            (b"content-length", str(len(body)).encode()),
            (b"content-type", b"application/json+protobuf"),
            (b"authorization", auth_header(self.cookies).encode()),
            (b"x-goog-api-key", WEB_API_KEY.encode()),
            (b"x-goog-authuser", b"0"),
            (b"x-user-agent", b"grpc-web-javascript/0.1"),
            (b"x-goog-ext-519733851-bin", CLIENT_EXT.encode()),
            (b"x-aistudio-visit-id", self.visit_id.encode()),
            (b"cookie", "; ".join(f"{name}={value}" for name, value in self.cookies.items()).encode()),
            (b"user-agent", USER_AGENT.encode()),
            (b"origin", STUDIO_ORIGIN.encode()),
            (b"referer", (STUDIO_ORIGIN + "/").encode()),
            (b"accept", b"*/*"),
        ]
        return headers

    async def stream(self, url: str, body: bytes) -> httpcore.Response:
        request = httpcore.Request("POST", url, headers=self._headers(url, body), content=body)
        try:
            response = await self._pool.handle_async_request(request)
        except httpcore.NetworkError as exc:
            raise AistudioError(502, f"aistudio transport error: {exc}") from exc
        if response.status != 200:
            try:
                text = (await response.aread()).decode("utf-8", errors="replace")
            finally:
                await response.aclose()
            raise AistudioError(response.status, _error_message(response.status, text))
        return response

    async def request(self, url: str, body: bytes) -> bytes:
        response = await self.stream(url, body)
        try:
            return await response.aread()
        finally:
            await response.aclose()

    async def aclose(self) -> None:
        await self._pool.aclose()


def _error_message(status: int, text: str) -> str:
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return f"aistudio error {status}: {text[:300]}"
    message = ""
    if isinstance(payload, list) and len(payload) > 1 and isinstance(payload[1], str):
        message = payload[1]
    body = message or text[:300]
    if status == 403 and "region" in body.lower():
        return f"aistudio error 403: {body} (check AISTUDIO_DOH_URL)"
    return f"aistudio error {status}: {body}"


class StreamParser:
    def __init__(self) -> None:
        self._buffer = ""
        self._pending = False
        self._pos = 0
        self._depth = 0
        self._frame_start = -1
        self._in_string = False
        self._escaped = False

    def feed(self, chunk: bytes) -> Iterator[list[Any]]:
        if chunk:
            self._buffer += chunk.decode("utf-8", errors="replace")
        if not self._pending:
            start = self._buffer.find("[")
            if start < 0:
                self._buffer = ""
                self._pos = 0
                return
            self._buffer = self._buffer[start:]
            self._pending = True
            self._pos = 1
            self._depth = 1
            self._frame_start = -1
            self._in_string = False
            self._escaped = False
        text = self._buffer
        events: list[list[Any]] = []
        index = self._pos
        while index < len(text):
            char = text[index]
            if self._in_string:
                if self._escaped:
                    self._escaped = False
                elif char == "\\":
                    self._escaped = True
                elif char == '"':
                    self._in_string = False
            elif char == '"':
                self._in_string = True
            elif char == "[":
                self._depth += 1
                if self._depth == 2 and self._frame_start < 0:
                    self._frame_start = index
            elif char == "]":
                first = self._depth == 2
                self._depth -= 1
                if first and self._frame_start >= 0:
                    raw = text[self._frame_start : index + 1]
                    self._frame_start = -1
                    try:
                        parsed = json.loads(raw)
                    except ValueError:
                        parsed = None
                    if isinstance(parsed, list):
                        events.append(parsed)
                if self._depth <= 0:
                    self._buffer = text[index + 1 :]
                    self._pending = False
                    self._pos = 0
                    self._depth = 0
                    self._frame_start = -1
                    self._in_string = False
                    self._escaped = False
                    yield from events
                    return
            index += 1
        self._pos = index
        yield from events


def event_parts(event: Any) -> Iterator[tuple[str, str]]:
    if not isinstance(event, list) or not event:
        return
    candidates = event[0]
    if not isinstance(candidates, list):
        return
    for candidate in candidates:
        if not isinstance(candidate, list) or not candidate:
            continue
        content = candidate[0]
        if not isinstance(content, list) or not content:
            continue
        parts = content[0]
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, list) or len(part) < 2:
                continue
            text = part[1]
            if not isinstance(text, str) or not text:
                continue
            thought = len(part) > 12 and part[12] == 1
            yield ("thought" if thought else "content"), text


def parse_models(payload: bytes) -> list[dict]:
    try:
        data = json.loads(payload)
    except (TypeError, ValueError):
        return []
    if not isinstance(data, list) or not data or not isinstance(data[0], list):
        return []
    models: list[dict] = []
    for entry in data[0]:
        if not isinstance(entry, list) or not entry:
            continue
        raw_id = entry[0]
        if not isinstance(raw_id, str) or not raw_id:
            continue
        actions = entry[7] if len(entry) > 7 and isinstance(entry[7], list) else []
        if actions and not CHAT_ACTIONS.intersection(str(action) for action in actions):
            continue
        model_id = raw_id.removeprefix(MODEL_PREFIX)
        record: dict[str, Any] = {
            "id": model_id,
            "name": entry[3] if len(entry) > 3 and isinstance(entry[3], str) else model_id,
            "owned_by": "aistudio",
            "model_type": "chat",
        }
        if len(entry) > 5 and isinstance(entry[5], int):
            record["context_length"] = entry[5]
        if len(entry) > 6 and isinstance(entry[6], int):
            record["max_output_tokens"] = entry[6]
        models.append(record)
    return models
