from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_API_URL = "http://127.0.0.1:8008/v1"
DEFAULT_MODEL = "default-thinking"
DEFAULT_REPO = "FANATFANATA/DanyAPI"
CDN_REPO = "FANATFANATA/cdn"
CDN_BRANCH = "main"
CONTEXT_FILES = ("contract.md", "capabilities.md")
REQUEST_TIMEOUT = 1800
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


def chat(api_url: str, api_key: str, model: str, system_prompt: str, user_prompt: str) -> str:
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
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        body = json.loads(response.read().decode("utf-8", errors="replace"))
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"no choices in response: {body}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"no content in response: {body}")
    return content


def normalize(line: str) -> str:
    text = BULLET_RE.sub("", line.strip())
    return SPACE_RE.sub(" ", text).strip()


def merge(findings: dict[str, str], text: str) -> None:
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
    fd, path = tempfile.mkstemp(prefix="collectandcheck.", suffix=".md")
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


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run collecter.py, check the dump with default-thinking and open one issue")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--repo", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.retries < 1:
        parser.error("--retries must be at least 1")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    run_collecter()
    collected = (ROOT / "collected.xml").read_text(encoding="utf-8", errors="replace")
    system_prompt = f"{load_context_file('contract.md')}\n\n{load_context_file('capabilities.md')}"
    user_prompt = f"{collected}\n\n//check"
    findings: dict[str, str] = {}
    for index in range(1, args.retries + 1):
        print(f"[{index}/{args.retries}] querying {args.model}", file=sys.stderr)
        try:
            text = chat(args.api_url, args.api_key, args.model, system_prompt, user_prompt)
        except Exception as exc:
            print(f"[{index}/{args.retries}] request failed: {exc}", file=sys.stderr)
            continue
        merge(findings, text)
    items = list(findings.values())
    repo = args.repo or detect_repo()
    body_lines = [
        "Automated report produced by collectandcheck.py.",
        "",
        f"Model: {args.model}",
        f"Retries: {args.retries}",
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
