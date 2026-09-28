"""Guards over the source tree. No torch.load without weights_only, no secret-shaped strings."""

import ast
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
VENDORED = SRC / "sartransfer" / "models" / "anyup"


def _own_sources():
    return [p for p in sorted(SRC.rglob("*.py")) if VENDORED not in p.parents]


def _is_torch_load(call):
    f = call.func
    return (isinstance(f, ast.Attribute) and f.attr == "load"
            and isinstance(f.value, ast.Name) and f.value.id == "torch")


def test_every_torch_load_is_weights_only():
    seen, bad = 0, []
    for p in _own_sources():
        rel = p.relative_to(REPO).as_posix()
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"), str(p))):
            if isinstance(node, ast.ImportFrom) and node.module == "torch":
                assert not any(a.name == "load" for a in node.names), f"{rel}:{node.lineno} aliases torch.load"
            if isinstance(node, ast.Call) and _is_torch_load(node):
                seen += 1
                kw = {k.arg: k.value for k in node.keywords}
                v = kw.get("weights_only")
                if not (isinstance(v, ast.Constant) and v.value is True):
                    bad.append(f"{rel}:{node.lineno}")
    assert seen >= 1, "no torch.load found, the scan is broken"
    assert not bad, f"torch.load without weights_only=True in {bad}"


# Full-token shapes, so this file does not match itself.
SECRET_PATTERNS = [re.compile(p) for p in (
    rb"hf_[A-Za-z0-9]{30,}",
    rb"ghp_[A-Za-z0-9]{36}",
    rb"github_pat_[A-Za-z0-9_]{22,}",
    rb"AKIA[0-9A-Z]{16}",
    rb"-----BEGIN [A-Z ]*PRIVATE KEY-----",
)]


def _files_to_scan():
    """What git would publish, tracked plus untracked-not-ignored. Tree walk without git."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True, check=True).stdout
        names = [n for n in out.decode("utf-8").split("\0") if n]
        return [REPO / n for n in names if (REPO / n).is_file()]
    except (OSError, subprocess.CalledProcessError):
        return [p for p in REPO.rglob("*") if p.is_file() and ".git" not in p.parts]


def test_no_secret_shaped_strings():
    files = _files_to_scan()
    assert Path(__file__).resolve() in [p.resolve() for p in files], "the scan does not cover tests/"
    hits = []
    for p in files:
        data = p.read_bytes()
        for pat in SECRET_PATTERNS:
            if pat.search(data):
                hits.append(f"{p.relative_to(REPO).as_posix()} matches {pat.pattern.decode()}")
    assert not hits, "\n".join(hits)         # names the file and the shape, never the match
