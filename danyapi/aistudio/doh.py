from __future__ import annotations

import asyncio
import logging
import secrets
import ssl
import struct
import time
from collections.abc import Iterable

import certifi
import httpcore
import httpx

log = logging.getLogger("danyapi.aistudio.doh")

DEFAULT_DOH_URL = "https://xbox-dns.ru/dns-query"

DOH_HOST_SUFFIXES = (
    ".google.com",
    ".googleapis.com",
    ".gstatic.com",
    ".googleusercontent.com",
    ".google.ru",
)

RESOLVE_TTL = 300.0
RESOLVE_TIMEOUT = 10.0
MAX_CACHE_ENTRIES = 64
DNS_TYPE_A = 1
DNS_TYPE_CNAME = 5
DNS_CLASS_IN = 1


class DohError(Exception):
    pass


def _encode_name(name: str) -> bytes:
    parts = [part for part in name.split(".") if part]
    encoded = bytearray()
    for part in parts:
        raw = part.encode("idna")
        if not raw or len(raw) > 63:
            raise DohError(f"invalid dns label in {name!r}")
        encoded.append(len(raw))
        encoded.extend(raw)
    encoded.append(0)
    return bytes(encoded)


def build_query(name: str) -> bytes:
    header = struct.pack("!HHHHHH", secrets.randbelow(65536), 0x0100, 1, 0, 0, 0)
    question = _encode_name(name) + struct.pack("!HH", DNS_TYPE_A, DNS_CLASS_IN)
    return header + question


def _read_name(data: bytes, offset: int) -> tuple[str, int]:
    labels: list[str] = []
    position = offset
    while True:
        if position >= len(data):
            raise DohError("truncated dns name")
        length = data[position]
        if length == 0:
            position += 1
            break
        if length & 0xC0 == 0xC0:
            if position + 1 >= len(data):
                raise DohError("truncated dns pointer")
            pointer = struct.unpack("!H", data[position : position + 2])[0] & 0x3FFF
            nested, _ = _read_name(data, pointer)
            if nested:
                labels.append(nested)
            position += 2
            break
        position += 1
        if position + length > len(data):
            raise DohError("truncated dns label")
        labels.append(data[position : position + length].decode("ascii", errors="replace"))
        position += length
    return ".".join(labels), position


def parse_answers(data: bytes) -> list[str]:
    if len(data) < 12:
        raise DohError("dns response is too short")
    _, _, question_count, answer_count, _, _ = struct.unpack("!HHHHHH", data[:12])
    offset = 12
    for _ in range(question_count):
        _, offset = _read_name(data, offset)
        offset += 4
    addresses: list[str] = []
    for _ in range(answer_count):
        _, offset = _read_name(data, offset)
        if offset + 10 > len(data):
            raise DohError("truncated dns answer")
        record_type, _, _, length = struct.unpack("!HHIH", data[offset : offset + 10])
        offset += 10
        if offset + length > len(data):
            raise DohError("truncated dns record data")
        if record_type == DNS_TYPE_A and length == 4:
            addresses.append(".".join(str(part) for part in data[offset : offset + 4]))
        offset += length
    return addresses


class DohResolver:
    def __init__(self, url: str = DEFAULT_DOH_URL, ttl: float = RESOLVE_TTL) -> None:
        self.url = url
        self.ttl = ttl
        self._cache: dict[str, tuple[float, list[str]]] = {}
        self._lock = asyncio.Lock()

    async def resolve(self, host: str) -> list[str]:
        now = time.monotonic()
        cached = self._cache.get(host)
        if cached is not None and now - cached[0] < self.ttl:
            return list(cached[1])
        async with self._lock:
            cached = self._cache.get(host)
            if cached is not None and now - cached[0] < self.ttl:
                return list(cached[1])
            addresses = await self._query(host)
            if addresses:
                self._cache[host] = (time.monotonic(), addresses)
                while len(self._cache) > MAX_CACHE_ENTRIES:
                    oldest = min(self._cache, key=lambda key: self._cache[key][0])
                    self._cache.pop(oldest, None)
            return list(addresses)

    async def _query(self, host: str) -> list[str]:
        headers = {
            "accept": "application/dns-message",
            "content-type": "application/dns-message",
        }
        async with httpx.AsyncClient(timeout=RESOLVE_TIMEOUT) as client:
            response = await client.post(self.url, content=build_query(host), headers=headers)
        if response.status_code != 200:
            raise DohError(f"doh answered {response.status_code} for {host}")
        addresses = parse_answers(response.content)
        if not addresses:
            raise DohError(f"doh returned no address for {host}")
        return addresses


class DohNetworkBackend(httpcore.AnyIOBackend):
    def __init__(self, resolver: DohResolver, suffixes: Iterable[str] = DOH_HOST_SUFFIXES) -> None:
        self._resolver = resolver
        self._suffixes = tuple(suffixes)

    def _in_scope(self, host: str) -> bool:
        lowered = host.lower().rstrip(".")
        return any(lowered == suffix.lstrip(".") or lowered.endswith(suffix) for suffix in self._suffixes)

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable | None = None,
    ) -> httpcore.AsyncNetworkStream:
        if not self._in_scope(host):
            return await super().connect_tcp(host, port, timeout=timeout, local_address=local_address, socket_options=socket_options)
        try:
            addresses = await self._resolver.resolve(host)
        except Exception as exc:
            log.warning("doh resolve failed for %s: %s", host, exc)
            addresses = []
        last_error: httpcore.ConnectError | None = None
        for address in addresses:
            try:
                return await super().connect_tcp(address, port, timeout=timeout, local_address=local_address, socket_options=socket_options)
            except httpcore.ConnectError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        return await super().connect_tcp(host, port, timeout=timeout, local_address=local_address, socket_options=socket_options)


def build_ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where())
