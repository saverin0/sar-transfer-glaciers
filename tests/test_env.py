"""parse_env_file keeps clean entries, drops empties and refuses merged lines without leaking them."""

from pathlib import Path

import pytest

from sartransfer.env import parse_env_file

REPO = Path(__file__).resolve().parents[1]


def test_quotes_comments_and_empty_values(tmp_path):
    f = tmp_path / ".env"
    f.write_text(
        "# a comment\n"
        "\n"
        'HF_TOKEN="abc"\n'
        "OTHER='x y'  \n"
        "no_equals_sign\n"
        "EMPTY=\n"
        "  SPACED = v \n",
        encoding="utf-8",
    )
    assert parse_env_file(f) == {"HF_TOKEN": "abc", "OTHER": "x y", "SPACED": "v"}


def test_merged_line_is_skipped_without_printing_the_value(tmp_path, capsys):
    f = tmp_path / ".env"
    f.write_text("HF_TOKEN=hf_abcWANDB_API_KEY=xyz\n", encoding="utf-8")
    assert parse_env_file(f) == {}
    out = capsys.readouterr().out
    assert "SKIPPING HF_TOKEN" in out
    assert "hf_abc" not in out
    assert "xyz" not in out


@pytest.mark.parametrize("value", ["abc==", "QUJD=="])   # upper-case one reaches the merged-line regex
def test_base64_padding_is_kept(tmp_path, value):
    f = tmp_path / ".env"
    f.write_text(f"KEY={value}\n", encoding="utf-8")
    assert parse_env_file(f) == {"KEY": value}


def test_env_example_yields_nothing():
    assert parse_env_file(REPO / ".env.example") == {}
