"""verify_remote_code refuses anything but the reviewed copy, and the pins are in place."""

import hashlib
import re
from pathlib import Path

import huggingface_hub                       # the stub from conftest
import pytest

from sartransfer.models import remote_code_hashes
from sartransfer.models.encoders import ENCODERS, EncoderSpec, verify_remote_code

MODEL = "example/remote-model"
REV = "a" * 40
REVIEWED = {"model.py": "x = 1\n", "layers/attn.py": "y = 2\n"}


def _spec(revision=REV):
    return EncoderSpec(MODEL, "example", "radio", trust_remote_code=True, revision=revision)


def _write(d: Path, rel: str, text: str) -> str:
    p = d / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    """A fake HF snapshot holding the reviewed files, registered in the hash table."""
    d = tmp_path / "snapshot"
    hashes = {rel: _write(d, rel, text) for rel, text in REVIEWED.items()}
    monkeypatch.setitem(remote_code_hashes.REMOTE_CODE_HASHES, MODEL, hashes)
    calls = []

    def fake_snapshot_download(model_id, revision=None, allow_patterns=None, token=None):
        calls.append((model_id, revision, allow_patterns))
        return str(d)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download, raising=False)
    return d, calls


def test_exact_match_passes_and_requests_only_py_of_the_pin(snapshot):
    d, calls = snapshot
    assert verify_remote_code(_spec()) == d
    assert calls == [(MODEL, REV, ["*.py"])]


def test_changed_content_raises(snapshot):
    d, _ = snapshot
    (d / "model.py").write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="model.py"):
        verify_remote_code(_spec())


def test_missing_reviewed_file_raises(snapshot):
    d, _ = snapshot
    (d / "layers" / "attn.py").unlink()
    with pytest.raises(RuntimeError, match="layers/attn.py"):
        verify_remote_code(_spec())


@pytest.mark.parametrize("extra", ["extra.py", "layers/extra.py", "deep/er/extra.py"])
def test_extra_py_file_raises(snapshot, extra):
    d, _ = snapshot
    _write(d, extra, "import os\n")
    with pytest.raises(RuntimeError, match=re.escape(extra)):
        verify_remote_code(_spec())


def test_no_revision_raises_before_any_download(snapshot):
    _, calls = snapshot
    with pytest.raises(ValueError, match="pinned revision"):
        verify_remote_code(_spec(revision=None))
    assert calls == []


def test_cradio_pin_and_hash_table():
    spec = ENCODERS["cradio-v4-h"]
    assert spec.trust_remote_code
    assert re.fullmatch(r"[0-9a-f]{40}", spec.revision)
    table = remote_code_hashes.REMOTE_CODE_HASHES["nvidia/C-RADIOv4-H"]
    assert len(table) == 26
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in table.values())
    assert all(name.endswith(".py") for name in table)


def test_every_remote_code_encoder_is_pinned_and_hashed():
    remote = [s for s in ENCODERS.values() if s.trust_remote_code]
    assert remote, "no remote-code encoder found"
    for spec in remote:
        assert spec.revision, f"{spec.name} has no pinned revision"
        assert spec.model_id in remote_code_hashes.REMOTE_CODE_HASHES, f"{spec.name} has no hash table"
