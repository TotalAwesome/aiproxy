from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import time
from pathlib import Path
from typing import Any

from ..config import settings

log = logging.getLogger("danyapi.aistudio.browser")

STUDIO_URL = "https://aistudio.google.com/prompts/new_chat"
ACCOUNTS_URL = "https://accounts.google.com/"

LOGIN_TIMEOUT = 120.0
READY_TIMEOUT = 120.0
READY_POLL_SEC = 0.5

HOOK_SCRIPT = """
(() => {
  if (window.__danyapiHookInstalled) return;
  window.__danyapiHookInstalled = true;
  let bg;
  Object.defineProperty(window, 'botguard', {
    configurable: true,
    get() { return bg; },
    set(v) {
      if (!v || typeof v !== 'object') { bg = v; return; }
      bg = v;
      let wrapped = null;
      Object.defineProperty(v, 'a', {
        configurable: true,
        get() { return wrapped; },
        set(fn) {
          wrapped = function (...args) {
            const setup = args[1];
            if (typeof setup === 'function') {
              args[1] = function (asyncSnapshot) {
                window.__danyapiAsyncSnapshot = asyncSnapshot;
                return setup.apply(this, arguments);
              };
            }
            return fn.apply(this, args);
          };
        },
      });
    },
  });
})();
"""

READY_SCRIPT = "typeof window.__danyapiAsyncSnapshot === 'function'"

MINT_SCRIPT = """
async (binding) => {
  const fn = window.__danyapiAsyncSnapshot;
  if (typeof fn !== 'function') throw new Error('aistudio botguard snapshot is not ready');
  return await new Promise((resolve) => {
    fn((token) => resolve(token), [{ content: binding }, undefined, undefined, undefined]);
  });
}
"""


class BrowserError(Exception):
    pass


def _stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def parse_login(raw: str) -> tuple[str, str]:
    email, separator, password = raw.partition(":")
    if not separator or not email.strip() or not password:
        raise BrowserError("aistudio login must be an email:password pair")
    return email.strip(), password


class StudioBrowser:
    def __init__(self, login: str = "", state_dir: str = "", headless: bool = True, doh_url: str = "") -> None:
        self.login = login
        self.state_dir = state_dir
        self.headless = headless
        self.doh_url = doh_url
        self._manager: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._lock = asyncio.Lock()
        self._started = False

    def _state_path(self) -> Path | None:
        if not self.state_dir or not self.login:
            return None
        directory = Path(self.state_dir)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{_stable_id(self.login)}.json"

    async def start(self) -> None:
        async with self._lock:
            if self._started:
                return
            try:
                from camoufox.async_api import AsyncCamoufox
            except ImportError as exc:
                raise BrowserError("camoufox is not installed, run pip install camoufox and camoufox fetch") from exc
            launch: dict[str, Any] = {
                "headless": self.headless,
                "main_world_eval": True,
                "enable_cache": True,
                "locale": "US",
            }
            if self.doh_url:
                launch["firefox_user_prefs"] = {
                    "network.trr.mode": 3,
                    "network.trr.uri": self.doh_url,
                    "network.trr.custom_uri": self.doh_url,
                }
            try:
                manager = AsyncCamoufox(**launch)
            except TypeError:
                launch.pop("firefox_user_prefs", None)
                manager = AsyncCamoufox(**launch)
            self._manager = manager
            try:
                browser = await manager.__aenter__()
            except Exception as exc:
                self._manager = None
                raise BrowserError(f"camoufox failed to start: {exc}") from exc
            self._browser = browser
            context = await self._new_context(browser)
            self._context = context
            await context.add_init_script(HOOK_SCRIPT)
            page = await context.new_page()
            self._page = page
            await self._ensure_login()
            self._started = True

    async def _new_context(self, browser: Any) -> Any:
        state_path = self._state_path()
        state = str(state_path) if state_path is not None and state_path.is_file() else None
        try:
            return await browser.new_context(storage_state=state, locale="US")
        except Exception as exc:
            if state is not None:
                log.warning("aistudio saved state is unusable, starting clean: %s", exc)
                return await browser.new_context(locale="US")
            raise

    async def _ensure_login(self) -> None:
        page = self._page
        await page.goto(STUDIO_URL, wait_until="domcontentloaded", timeout=LOGIN_TIMEOUT * 1000)
        if await self._ready():
            return
        if not self.login:
            raise BrowserError("aistudio page is not logged in, set AISTUDIO_LOGINS or a saved state")
        if not str(page.url).startswith(ACCOUNTS_URL):
            raise BrowserError(f"aistudio is at an unexpected url: {page.url}")
        email, password = parse_login(self.login)
        log.info("logging in to aistudio as %s", email)
        try:
            await page.locator("input#identifierId").fill("")
            await page.locator("input#identifierId").type(email)
            await page.locator("#identifierNext button").click(timeout=LOGIN_TIMEOUT * 1000)
            await page.locator('input[name="Passwd"]').wait_for(state="visible", timeout=LOGIN_TIMEOUT * 1000)
            await page.locator('input[name="Passwd"]').type(password)
            await page.locator("#passwordNext button").click(timeout=LOGIN_TIMEOUT * 1000)
            await page.wait_for_url(f"{STUDIO_URL}*", timeout=LOGIN_TIMEOUT * 1000)
        except Exception as exc:
            raise BrowserError(f"aistudio login failed (2FA or captcha?): {exc}") from exc
        with contextlib.suppress(Exception):
            welcome = page.locator('mat-dialog-content .welcome-option button[aria-label="Try Gemini"]')
            if await welcome.count() > 0:
                await welcome.click()
        await self._ready(timeout=READY_TIMEOUT)
        await self._save_state()

    async def _save_state(self) -> None:
        state_path = self._state_path()
        if state_path is None or self._context is None:
            return
        try:
            await self._context.storage_state(path=str(state_path))
        except Exception as exc:
            log.warning("cannot save aistudio state: %s", exc)

    async def _ready(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            with contextlib.suppress(Exception):
                ready = await self._page.evaluate(READY_SCRIPT)
                if ready:
                    return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(READY_POLL_SEC)

    async def ensure_ready(self, timeout: float = READY_TIMEOUT) -> None:
        if self._page is None:
            raise BrowserError("browser is not started")
        if await self._ready(timeout):
            return
        raise BrowserError("aistudio botguard did not initialize in time")

    async def mint_token(self, binding: str) -> str:
        await self.start()
        await self.ensure_ready()
        token = await self._page.evaluate(MINT_SCRIPT, binding)
        if not isinstance(token, str) or not token:
            raise BrowserError("aistudio botguard returned an empty token")
        return token

    async def cookies(self) -> str:
        await self.start()
        try:
            cookies = await self._context.cookies()
        except Exception as exc:
            raise BrowserError(f"cannot read aistudio cookies: {exc}") from exc
        pairs = [f"{cookie['name']}={cookie['value']}" for cookie in cookies if "google" in str(cookie.get("domain", ""))]
        return "; ".join(pairs)

    async def stop(self) -> None:
        self._started = False
        self._page = None
        self._context = None
        browser, manager = self._browser, self._manager
        self._browser = None
        self._manager = None
        if browser is not None:
            with contextlib.suppress(Exception):
                await browser.close()
        if manager is not None:
            with contextlib.suppress(Exception):
                await manager.__aexit__(None, None, None)


def browser_settings() -> StudioBrowser:
    return StudioBrowser(
        headless=settings.aistudio_headless,
        state_dir=settings.aistudio_state_dir,
        doh_url=settings.aistudio_doh_url,
    )
