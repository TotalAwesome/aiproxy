import fnmatch
import os
import pathlib
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent

XML_ILLEGAL_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f\ufdd0-\ufddf\ufffe\uffff]")


def xml_escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def xml_attr(value):
    escaped = xml_escape(value)
    escaped = escaped.replace('"', "&quot;").replace("\t", "&#9;").replace("\n", "&#10;").replace("\r", "&#13;")
    return f'"{escaped}"'


def sanitize_xml_text(text):
    return XML_ILLEGAL_RE.sub("", text)


IGNORE_PATTERNS = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".hypothesis",
    ".coverage",
    "egg-info",
    ".env",
    ".installed-release",
    ".latest-release-cache",
    ".previous-release",
    ".venv",
    "venv",
    "node_modules",
    ".idea",
    ".vscode",
    "references",
    "collected.xml",
    ".netrc",
    "_netrc",
    ".npmrc",
    ".pypirc",
    ".git-credentials",
    ".aws",
    ".ssh",
    ".gnupg",
    "kubeconfig",
    "kubeconfig.*",
}

EXCLUDE_EXTS = {".pyc", ".db", ".cache", ".wasm", ".exe", ".dll", ".so", ".obj", ".pem", ".key", ".p12", ".pfx", ".ovpn", ".ppk"}

EXCLUDE_NAME_GLOBS = {
    "*.log",
    "*.bak",
    "*.orig",
    "*.rej",
    "*.tmp",
    "*.temp",
    "*.swp",
    "*.swo",
    "*~",
    "coverage.xml",
    "*.sqlite",
    "*.sqlite3",
    "*.zip",
    "*.tar",
    "*.tar.gz",
    ".env.*",
    "*.env",
    "*.key",
    "*.pem",
    "*.secret",
    "*.tfvars",
    "*.tfvars.json",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa.pub",
    "id_ed25519.pub",
    ".netrc",
    "_netrc",
    ".npmrc",
    ".pypirc",
    ".git-credentials",
    "kubeconfig",
    "kubeconfig.*",
    "*.ovpn",
    "*.ppk",
}

LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "jsx",
    ".tsx": "tsx",
    ".html": "html",
    ".css": "css",
    ".sh": "bash",
    ".ps1": "powershell",
    ".bat": "batch",
    ".cmd": "batch",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".txt": "text",
    ".md": "markdown",
    ".json": "json",
    ".xml": "xml",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "c-header",
    ".hpp": "c++-header",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".lua": "lua",
    ".sql": "sql",
    ".svg": "svg",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "ini",
    ".dockerfile": "dockerfile",
    "dockerfile": "dockerfile",
    "makefile": "makefile",
    ".cmake": "cmake",
    ".log": "text",
    ".csv": "csv",
    ".graphql": "graphql",
    ".proto": "protobuf",
    ".tf": "terraform",
}

EXT_TO_LANG = dict(LANG_MAP)

EXCLUDE_NAMES = {"collected.xml", "collecter.py"}


def get_lang(filepath: pathlib.Path) -> str:
    name_lower = filepath.name.lower()
    if name_lower in EXT_TO_LANG:
        return EXT_TO_LANG[name_lower]
    ext = filepath.suffix.lower()
    return EXT_TO_LANG.get(ext, "unknown")


def is_ignore(rel_parts) -> bool:
    return any(p in IGNORE_PATTERNS for p in rel_parts)


def _load_ignore_files(root: pathlib.Path) -> list[list[str]]:
    patterns = []
    for name in (".gitignore",):
        candidate = root / name
        if not candidate.is_file():
            continue
        try:
            lines = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        patterns.append([line.strip() for line in lines if line.strip() and not line.startswith("#")])
    return patterns


def _matches_gitignore(rel: str, patterns: list[str]) -> bool:
    parts = pathlib.PurePosixPath(rel).parts
    ignored = False
    for raw in patterns:
        pattern = raw.rstrip("/")
        negated = pattern.startswith("!")
        if negated:
            pattern = pattern[1:]
        if not pattern:
            continue
        anchored = "/" in pattern
        clean = pattern.lstrip("/")
        hit = False
        if anchored:
            hit = fnmatch.fnmatch(rel, clean)
        else:
            for part in parts:
                if fnmatch.fnmatch(part, clean):
                    hit = True
                    break
        if hit:
            ignored = not negated
    return ignored


def should_include(path: pathlib.Path) -> bool:
    if path.name in EXCLUDE_NAMES:
        return False
    if path.suffix.lower() in EXCLUDE_EXTS:
        return False
    if any(fnmatch.fnmatch(path.name, pattern) for pattern in EXCLUDE_NAME_GLOBS):
        return False
    return True


def collect_files(root: pathlib.Path):
    ignore_files = _load_ignore_files(root)
    files = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if is_ignore(pathlib.PurePosixPath(rel).parts) or not should_include(p):
            continue
        if any(_matches_gitignore(rel, patterns) for patterns in ignore_files):
            continue
        files.append(p)
    files.sort(key=lambda p: str(p.relative_to(root)).lower())
    return files


def read_file_content(fp: pathlib.Path) -> str:
    try:
        return fp.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"ERROR: {e}"


def main():
    root = ROOT
    files = collect_files(root)
    out_path = root / "collected.xml"

    total_size = sum(fp.stat().st_size if fp.exists() else 0 for fp in files)

    tmp_fd, tmp_name = None, None
    try:
        tmp_fd, tmp_name = tempfile.mkstemp(dir=str(root), prefix="collected.", suffix=".tmp")
        with os.fdopen(tmp_fd, "w", encoding="utf-8", newline="\n") as handle:
            tmp_fd = None
            handle.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            handle.write(f"<collection source={xml_attr(str(root))} files={xml_attr(str(len(files)))} totalSize={xml_attr(str(total_size))}>\n")
            for fp in files:
                rel = fp.relative_to(root).as_posix()
                try:
                    size = fp.stat().st_size
                except OSError as exc:
                    size = 0
                    content = f"ERROR: {exc}"
                else:
                    content = read_file_content(fp)
                header = f"  <file name={xml_attr(rel)} lang={xml_attr(get_lang(fp))} size={xml_attr(str(size))}>"
                handle.write(f"{header}{xml_escape(sanitize_xml_text(content))}</file>\n")
            handle.write("</collection>\n")
        os.replace(tmp_name, out_path)
        tmp_name = None
    except OSError as exc:
        print(f"Could not write {out_path}: {exc}", file=sys.stderr)
        return 1
    finally:
        if tmp_fd is not None:
            os.close(tmp_fd)
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
    print(f"Collected {len(files)} files ({total_size} bytes) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
