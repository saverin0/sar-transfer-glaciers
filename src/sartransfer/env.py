"""Read secrets from a .env file into os.environ, with a getpass fallback.

Two places hold the same file, because the Colab runtime cannot see local disk:

    <repository>/.env                                  (local, gitignored)
    /content/drive/MyDrive/sar-transfer/.env           (Colab, via Drive)

Values already present in os.environ always win, so an env var set by hand is
never silently overwritten by a stale file. Nothing here ever prints a secret.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

SEARCH_PATHS = (
    Path("/content/drive/MyDrive/sar-transfer/.env"),   # Colab + Drive, permanent
    Path("/content/.env"),                              # Colab, this runtime only
    Path.cwd() / ".env",
    Path(__file__).resolve().parents[2] / ".env",       # repo root, local runs
)

KNOWN = ("HF_TOKEN",)


def parse_env_file(path: str | Path) -> dict[str, str]:
    """Minimal KEY=VALUE parser. Ignores blanks, # comments and empty values.

    Rejects a value that swallowed the following line: a .env that loses a line
    break yields e.g. `HF_TOKEN=hf_abc123WANDB_API_KEY=xyz`, which would be sent
    as a broken token. Any `NAME=` inside a value counts (NAME: a capital, then
    two or more capitals, digits or underscores), except at the very end or
    before another `=` (base64 padding). Better to drop it loudly than pass it on.
    """
    out: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("'\"")
        if not val:
            continue
        merged = [k for k in KNOWN if k != key and k + "=" in val]
        if merged or re.search(r"[A-Z][A-Z0-9_]{2,}=(?!=|$)", val):   # never prints the value
            print(f"  SKIPPING {key} from {path}: value also contains {merged or 'another NAME=... entry'} --"
                  " the file is missing a line break between entries")
            continue
        out[key] = val
    return out


def load_env(path: str | Path | None = None, verbose: bool = True) -> list[str]:
    """Load the first .env found into os.environ. Returns the keys now set.

    Existing environment variables are left alone.
    """
    candidates = [Path(path)] if path else list(SEARCH_PATHS)
    used = None
    for p in candidates:
        try:
            if p.is_file():
                used = p
                break
        except OSError:
            continue

    if used is not None:
        for key, val in parse_env_file(used).items():
            os.environ.setdefault(key, val)
        if verbose:
            print(f"loaded secrets from {used}")
    elif verbose:
        print("no .env found in:", ", ".join(str(p) for p in candidates))

    have = [k for k in KNOWN if os.environ.get(k)]
    if verbose:
        for k in KNOWN:
            print(f"  {k:<15} {'set' if os.environ.get(k) else 'MISSING'}")
    return have


def require(key: str, why: str = "") -> str:
    """Return a secret, prompting with getpass if the .env did not supply it."""
    val = os.environ.get(key)
    if not val:
        from getpass import getpass

        val = getpass(f"{key}{' for ' + why if why else ''}: ")
        os.environ[key] = val
    return val


def model_cache_on_drive(root: str | Path = "/content/drive/MyDrive/sar-transfer/hf_cache") -> Path | None:
    """Keep downloaded model weights on Drive instead of the runtime's disk.

    Every encoder here (DINOv3, C-RADIO, TerraMind, SARATR-X) is fetched through
    the Hugging Face hub, which caches under HF_HOME. Pointing that at Drive
    means each model is downloaded ONCE and reused across sessions. Must be
    called before transformers / huggingface_hub are imported. Returns the
    cache path, or None (with a message) if Drive is not mounted.
    """
    import os

    if not os.path.ismount("/content/drive"):
        print("Drive not mounted -- model cache stays on the runtime (re-downloaded next session)")
        return None
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(root)
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"   # Drive has no symlinks; hub copies instead
    have = sorted(d.name for d in (root / "hub").glob("models--*")) if (root / "hub").exists() else []
    print(f"model cache: {root} | cached: {have or 'nothing yet'}")
    return root
