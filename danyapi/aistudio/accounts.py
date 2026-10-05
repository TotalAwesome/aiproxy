from __future__ import annotations

import asyncio
import logging
from typing import Any

from .browser import BrowserError, StudioBrowser
from .client import LIST_MODELS_URL, AistudioError, AistudioTransport, build_list_models_request

log = logging.getLogger("danyapi.aistudio.accounts")


class AistudioClient:
    def __init__(self, browser: StudioBrowser, transport: AistudioTransport) -> None:
        self.browser = browser
        self.transport = transport

    async def check_auth(self) -> bool:
        try:
            await self.transport.request(LIST_MODELS_URL, build_list_models_request())
            return True
        except AistudioError as exc:
            log.warning("aistudio auth check failed: %s", exc.message)
            return False
        except Exception as exc:
            log.warning("aistudio auth check error: %s", exc)
            return False

    async def aclose(self) -> None:
        try:
            await self.transport.aclose()
        except Exception as exc:
            log.warning("aistudio transport close failed: %s", exc)
        try:
            await self.browser.stop()
        except Exception as exc:
            log.warning("aistudio browser stop failed: %s", exc)


class AistudioAccount:
    __slots__ = ("broken", "broken_at", "browser", "client", "index", "login", "sem", "stable_id", "transport")

    def __init__(self, index: int, login: str, browser: StudioBrowser, transport: AistudioTransport, stable_id: str | None = None) -> None:
        self.index = index
        self.browser = browser
        self.transport = transport
        self.client = AistudioClient(browser, transport)
        self.sem = asyncio.Semaphore(1)
        self.stable_id = stable_id
        self.broken = False
        self.broken_at: float | None = None
        self.login = login

    def mark_broken(self) -> None:
        if not self.broken:
            self.broken = True
            log.warning("aistudio account #%d marked broken", self.index)

    async def refresh_cookies(self) -> None:
        try:
            cookies = await self.browser.cookies()
        except BrowserError as exc:
            raise AistudioError(502, str(exc)) from exc
        if cookies:
            self.transport.update_cookies(cookies)

    @property
    def label(self) -> str:
        return f"aistudio#{self.index}"

    def info(self) -> dict[str, Any]:
        return {"index": self.index, "login": self.login, "broken": self.broken}
