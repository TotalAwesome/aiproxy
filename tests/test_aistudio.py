from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

from danyapi.aistudio import api as aistudio_api
from danyapi.aistudio.browser import BrowserError, parse_login
from danyapi.aistudio.client import (
    SAFETY_SETTINGS,
    AistudioError,
    StreamParser,
    auth_header,
    build_contents,
    build_request,
    event_parts,
    parse_cookies,
    parse_models,
)
from danyapi.api.models import _resolve_provider


def _event(text: str, thought: bool = False) -> list:
    part: list[object] = [None, text]
    if thought:
        part = [None] * 12 + [1]
        part[1] = text
    content = [[part], "model"]
    return [[[content, 1]]]


class _Message:
    def __init__(self, role: str, content, name: str | None = None) -> None:
        self.role = role
        self.content = content
        self.name = name


def test_parse_login_requires_a_pair():
    assert parse_login("user@example.com:secret") == ("user@example.com", "secret")
    assert parse_login("user@example.com:a:b") == ("user@example.com", "a:b")
    with pytest.raises(BrowserError):
        parse_login("user@example.com")
    with pytest.raises(BrowserError):
        parse_login(":secret")


def test_parse_cookies_keeps_values_with_equals():
    assert parse_cookies("A=1; B=x=y; C=3") == {"A": "1", "B": "x=y", "C": "3"}


def test_auth_header_hashes_the_sapisid_cookie():
    digest = hashlib.sha1(b"1700000000 sapisid https://aistudio.google.com").hexdigest()
    assert auth_header({"SAPISID": "sapisid"}, timestamp=1700000000) == f"SAPISIDHASH 1700000000_{digest}"
    with pytest.raises(AistudioError):
        auth_header({})


def test_build_contents_folds_system_into_the_first_user_message():
    contents, system = build_contents(
        [
            _Message("system", "Be terse."),
            _Message("user", "Hello"),
            _Message("assistant", "Hi"),
            _Message("user", "How are you?"),
        ]
    )
    assert system == ["Be terse."]
    assert [content[1] for content in contents] == ["user", "model", "user"]
    assert contents[0][0][0][1] == "Follow these instructions for the rest of the conversation:\nBe terse.\n\nHello"
    assert contents[1][0][0][1] == "Hi"
    assert contents[2][0][0][1] == "How are you?"


def test_build_contents_maps_tool_results_and_defaults():
    contents, _ = build_contents([_Message("tool", "42", name="calc")])
    assert contents == [[[[None, "[tool result for calc]\n42"]], "user"]]
    contents, _ = build_contents([])
    assert contents == [[[[None, "Hello"]], "user"]]


def test_build_request_shape_and_overrides():
    body = json.loads(build_request("gemini-flash-latest", [[[[None, "hi"]], "user"]], "tok", max_tokens=123, temperature=0.2, top_p=0.8, top_k=7))
    assert body[0] == "models/gemini-flash-latest"
    assert body[1] == [[[[None, "hi"]], "user"]]
    assert body[2] == SAFETY_SETTINGS
    assert body[3][3] == 123
    assert body[3][4] == 0.2
    assert body[3][5] == 0.8
    assert body[3][6] == 7
    assert body[4] == "tok"


def test_stream_parser_handles_split_and_multiple_frames():
    parser = StreamParser()
    events = list(parser.feed(b"[[1,2],[3,")) + list(parser.feed(b"4]]"))
    assert events == [[1, 2], [3, 4]]


def test_stream_parser_keeps_the_incomplete_tail():
    parser = StreamParser()
    assert list(parser.feed(b"[[1,2],")) == [[1, 2]]
    assert list(parser.feed(b"[3")) == []
    assert list(parser.feed(b",4]]")) == [[3, 4]]


def test_stream_parser_feeds_event_parts():
    part = [None, "hello"]
    content = [[part], "model"]
    candidate = [content, 1]
    event = [[candidate]]
    parser = StreamParser()
    events = list(parser.feed(json.dumps([event]).encode()))
    assert events == [event]
    texts = [text for item in events for kind, text in event_parts(item) if kind == "content"]
    assert texts == ["hello"]


def test_event_parts_separates_thoughts_and_ignores_signature_only():
    thought = [None, "thinking", None, None, None, None, None, None, None, None, None, None, 1]
    answer = [None, "answer"]
    signature = [None, "", None, None, None, None, None, None, None, None, None, None, None, None, "sig"]
    candidate = [[[thought, answer, signature], "model"], 1]
    assert list(event_parts([[candidate], None, None, None])) == [("thought", "thinking"), ("content", "answer")]


def test_parse_models_filters_non_chat_actions():
    payload = json.dumps(
        [
            [
                ["models/gemini-flash-latest", None, "1.0", "Gemini Flash", "desc", 1000, 100, ["generateContent", "countTokens"]],
                ["models/imagen-4", None, "1.0", "Imagen", "desc", 1000, 100, ["predict"]],
                ["models/gemini-3-pro-preview", None, "1.0", "Gemini 3 Pro", "desc", 2000, 200, ["generateContent"]],
            ]
        ]
    ).encode()
    models = parse_models(payload)
    assert [model["id"] for model in models] == ["gemini-flash-latest", "gemini-3-pro-preview"]
    assert models[0]["owned_by"] == "aistudio"
    assert models[0]["context_length"] == 1000
    assert models[0]["max_output_tokens"] == 100
    assert parse_models(b"not json") == []


def test_resolve_provider_routes_gemini_ids_to_aistudio():
    for model in ("gemini-flash-latest", "gemini-3-pro-preview", "antigravity-preview-09-2026", "deep-research-preview-04-2026"):
        assert _resolve_provider(model) == "aistudio"


def test_api_error_mapping():
    assert aistudio_api._status_for(AistudioError(401, "bad auth")) == 401
    assert aistudio_api._status_for(AistudioError(403, "permission")) == 403
    assert aistudio_api._status_for(AistudioError(429, "slow down")) == 502
    assert aistudio_api._status_for(AistudioError(500, "boom")) == 502
    assert aistudio_api._status_for(AistudioError(400, "bad")) == 400


def test_binding_uses_content_text_only():
    contents = [[[[None, "head"], [None, "tail"]], "user"], [[[None, "answer"]], "model"]]
    expected = hashlib.sha256(b"head tail answer").hexdigest()
    assert aistudio_api._binding_for(contents) == expected


class _FakeResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def aiter_stream(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        return None


class _FakeTransport:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.requests: list[tuple[str, bytes]] = []

    async def stream(self, url: str, body: bytes) -> _FakeResponse:
        self.requests.append((url, body))
        return _FakeResponse(self.chunks)


class _FakeBrowser:
    def __init__(self) -> None:
        self.bindings: list[str] = []

    async def mint_token(self, binding: str) -> str:
        self.bindings.append(binding)
        return "token-1"

    async def cookies(self) -> str:
        return "SAPISID=sapisid"


class _FakeAccount:
    def __init__(self, chunks: list[bytes]) -> None:
        self.transport = _FakeTransport(chunks)
        self.browser = _FakeBrowser()
        self.sem = asyncio.Semaphore(1)

    async def refresh_cookies(self) -> None:
        return None


def _response_body(*texts: str) -> list[bytes]:
    events = [_event(text) for text in texts]
    raw = json.dumps(events).encode()
    middle = len(raw) // 2
    return [raw[:middle], raw[middle:]]


def _sse_texts(lines: list[str]) -> list[str]:
    texts: list[str] = []
    for line in lines:
        if not line.startswith("data: ") or line.strip() == "data: [DONE]":
            continue
        payload = json.loads(line[6:])
        delta = payload["choices"][0]["delta"].get("content")
        if delta:
            texts.append(delta)
    return texts


@pytest.mark.asyncio
async def test_stream_openai_emits_deltas_and_done():
    account = _FakeAccount(_response_body("Hel", "lo"))
    lines = [line async for line in aistudio_api.stream_openai(account, [_Message("user", "hi")], model="gemini-flash-latest")]
    assert _sse_texts(lines) == ["Hel", "lo"]
    assert lines[-1] == aistudio_api.DONE_LINE
    assert account.browser.bindings == [_binding_text("hi")]
    assert account.transport.requests[0][0] == aistudio_api.GENERATE_URL


def _binding_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.asyncio
async def test_collect_non_stream_joins_deltas_and_reports_usage():
    account = _FakeAccount(_response_body("Hel", "lo"))
    result = await aistudio_api.collect_non_stream(account, [_Message("user", "hi")], model="gemini-flash-latest")
    assert result["choices"][0]["message"]["content"] == "Hello"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["total_tokens"] > 0
    assert result["object"] == "chat.completion"


@pytest.mark.asyncio
async def test_collect_non_stream_applies_stop_sequences():
    account = _FakeAccount(_response_body("Hello STOP world"))
    result = await aistudio_api.collect_non_stream(account, [_Message("user", "hi")], model="gemini-flash-latest", stop=["STOP"])
    assert result["choices"][0]["message"]["content"] == "Hello "
