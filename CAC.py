from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_MODEL = "default-thinking"
DEFAULT_REPO = "FANATFANATA/DanyAPI"
DEFAULT_MAX_BYTES = 1024 * 1024
DEFAULT_ATTEMPTS = 3
DEFAULT_BACKOFF = 2.0
DEFAULT_CONCURRENCY = 4
CDN_REPO = "FANATFANATA/cdn"
CDN_BRANCH = "main"
CONTEXT_FILES = ("contract.md", "capabilities.md")
ENV_FILE = ROOT / ".env"
ENV_PREFIX = "CAC_"
REQUEST_TIMEOUT = 1800
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
BULLET_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
SPACE_RE = re.compile(r"\s+")
REMOTE_RE = re.compile(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?$")


def contract_dirs() -> list[pathlib.Path]:
    return [ROOT, ROOT.parent / "cdn", ROOT.parent]


def fetch_remote(name: str) -> str | None:
    url = f"https://raw.githubusercontent.com/{CDN_REPO}/{CDN_BRANCH}/{name}"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError):
        return None


def load_context_file(name: str) -> str:
    for base in contract_dirs():
        candidate = base / name
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8", errors="replace")
    remote = fetch_remote(name)
    if remote is not None:
        return remote
    raise SystemExit(f"cannot find {name} locally and cannot download it from {CDN_REPO}")


def run_collecter() -> None:
    result = subprocess.run([sys.executable, "collecter.py"], cwd=ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(f"collecter.py exited with {result.returncode}")


def split_chunks(xml_text: str, max_bytes: int) -> list[str]:
    lines = xml_text.splitlines()
    header: list[str] = []
    footer: list[str] = []
    files: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("<file ") or stripped.startswith("<file>"):
            files.append(line)
        elif stripped.startswith("<collection"):
            header.append(line)
        elif stripped.startswith("</collection>"):
            footer.append(line)
        elif not files:
            header.append(line)
    if not files:
        return [xml_text]
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    head = "\n".join(header)
    tail = "\n".join(footer)
    overhead = len(head.encode("utf-8")) + len(tail.encode("utf-8")) + 2
    for line in files:
        line_size = len(line.encode("utf-8")) + 1
        if current and current_size + line_size + overhead > max_bytes:
            chunks.append("\n".join([*header, *current, *footer]))
            current = []
            current_size = 0
        current.append(line)
        current_size += line_size
    if current:
        chunks.append("\n".join([*header, *current, *footer]))
    return chunks


class ChatError(Exception):
    def __init__(self, message: str, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


def chat_once(api_url: str, api_key: str, model: str, system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(api_url.rstrip("/") + "/chat/completions", data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        retryable = exc.code in RETRYABLE_STATUS
        raise ChatError(f"http {exc.code}", retryable) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ChatError(f"network {exc}", True) from exc
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ChatError(f"invalid json: {exc}", True) from exc
    choices = body.get("choices") or []
    if not choices:
        raise ChatError(f"no choices in response: {body}", False)
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise ChatError(f"no content in response: {body}", False)
    return content


def chat(
    api_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    attempts: int,
    backoff: float,
    log_prefix: str,
) -> str:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return chat_once(api_url, api_key, model, system_prompt, user_prompt)
        except ChatError as exc:
            last_exc = exc
            if not exc.retryable or attempt >= attempts:
                break
            delay = backoff * (2 ** (attempt - 1))
            print(f"{log_prefix} attempt {attempt}/{attempts} failed ({exc}), retrying in {delay:.1f}s", file=sys.stderr)
            time.sleep(delay)
    raise last_exc if last_exc is not None else RuntimeError("chat failed without exception")


def normalize(line: str) -> str:
    text = BULLET_RE.sub("", line.strip())
    return SPACE_RE.sub(" ", text).strip()


def merge(findings: dict[str, str], lock: threading.Lock, text: str) -> None:
    with lock:
        for raw in text.splitlines():
            item = normalize(raw)
            if not item:
                continue
            key = item.lower()
            if key not in findings:
                findings[key] = item


def detect_repo() -> str:
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return DEFAULT_REPO
    match = REMOTE_RE.search((result.stdout or "").strip())
    return match.group(1) if match else DEFAULT_REPO


def create_issue(repo: str, title: str, body: str) -> str:
    fd, path = tempfile.mkstemp(prefix="CAC.", suffix=".md")
    try:
        with open(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(body)
        result = subprocess.run(
            ["gh", "issue", "create", "--repo", repo, "--title", title, "--body-file", path],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        pathlib.Path(path).unlink(missing_ok=True)
    if result.returncode != 0:
        raise SystemExit(f"gh issue create failed: {(result.stderr or result.stdout).strip()}")
    return (result.stdout or "").strip()


def load_env_defaults() -> dict[str, str]:
    values: dict[str, str] = {}
    if not ENV_FILE.is_file():
        return values
    try:
        lines = ENV_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return values
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        stripped = stripped.removeprefix("export ").strip()
        key, separator, raw = stripped.partition("=")
        if not separator:
            continue
        key = key.strip()
        if not key.startswith(ENV_PREFIX):
            continue
        value = raw.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key[len(ENV_PREFIX) :].lower()] = value
    return values


def _env_int(values: dict[str, str], key: str, fallback: int) -> int:
    try:
        return int(values[key])
    except (KeyError, ValueError):
        return fallback


def _env_float(values: dict[str, str], key: str, fallback: float) -> float:
    try:
        return float(values[key])
    except (KeyError, ValueError):
        return fallback


def parse_args(argv: list[str]) -> argparse.Namespace:
    env = load_env_defaults()
    parser = argparse.ArgumentParser(description="run collecter.py, check the dump with default-thinking and open one issue")
    parser.add_argument("--retries", type=int, default=_env_int(env, "retries", 1))
    parser.add_argument("--api-url", default=env.get("api_url", ""))
    parser.add_argument("--api-key", default=env.get("api_key", ""))
    parser.add_argument("--model", default=env.get("model", DEFAULT_MODEL))
    parser.add_argument("--repo", default=env.get("repo", ""))
    parser.add_argument("--max-bytes", type=int, default=_env_int(env, "max_bytes", DEFAULT_MAX_BYTES))
    parser.add_argument("--attempts", type=int, default=_env_int(env, "attempts", DEFAULT_ATTEMPTS))
    parser.add_argument("--backoff", type=float, default=_env_float(env, "backoff", DEFAULT_BACKOFF))
    parser.add_argument("--concurrency", type=int, default=_env_int(env, "concurrency", DEFAULT_CONCURRENCY))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.api_url:
        parser.error("--api-url is required (or set CAC_API_URL in .env)")
    if args.retries < 1:
        parser.error("--retries must be at least 1")
    if args.max_bytes < 1:
        parser.error("--max-bytes must be at least 1")
    if args.attempts < 1:
        parser.error("--attempts must be at least 1")
    if args.backoff < 0:
        parser.error("--backoff must not be negative")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    run_collecter()
    collected = (ROOT / "collected.xml").read_text(encoding="utf-8", errors="replace")
    chunks = split_chunks(collected, args.max_bytes)
    system_prompt = f"{load_context_file('contract.md')}\n\n{load_context_file('capabilities.md')}"
    findings: dict[str, str] = {}
    lock = threading.Lock()

    jobs: list[tuple[int, int, str]] = []
    for index in range(1, args.retries + 1):
        for chunk_index, chunk in enumerate(chunks, 1):
            jobs.append((index, chunk_index, chunk))

    def run_job(job: tuple[int, int, str]) -> None:
        index, chunk_index, chunk = job
        log_prefix = f"[{index}/{args.retries}] chunk {chunk_index}/{len(chunks)}"
        print(f"{log_prefix} querying {args.model}", file=sys.stderr)
        user_prompt = f"{chunk}\n\n//check"
        try:
            text = chat(
                args.api_url,
                args.api_key,
                args.model,
                system_prompt,
                user_prompt,
                args.attempts,
                args.backoff,
                log_prefix,
            )
        except Exception as exc:
            print(f"{log_prefix} failed after {args.attempts} attempt(s): {exc}", file=sys.stderr)
            return
        merge(findings, lock, text)

    workers = min(args.concurrency, len(jobs)) if jobs else 1
    if workers <= 1:
        for job in jobs:
            run_job(job)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run_job, jobs))

    items = list(findings.values())
    repo = args.repo or detect_repo()
    body_lines = [
        "Automated report produced by CAC.py.",
        "",
        f"Model: {args.model}",
        f"Retries: {args.retries}",
        f"Chunks: {len(chunks)}",
        f"Concurrency: {workers}",
        f"Attempts: {args.attempts}",
        f"Findings: {len(items)}",
        "",
        "## Summary",
        "",
    ]
    body_lines.extend(f"- {item}" for item in items)
    if not items:
        body_lines.append("Nothing found.")
    body = "\n".join(body_lines) + "\n"
    title = f"auto check: {args.retries} retries, {len(items)} findings"
    if args.dry_run:
        print(body)
        print(f"dry run, issue not created (repo {repo})", file=sys.stderr)
        return 0
    print(create_issue(repo, title, body))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
