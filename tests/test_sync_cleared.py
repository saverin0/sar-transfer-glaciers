"""Published notebooks carry no packed code. Every sync cell is the CLEARED stub."""

import json

from sartransfer.sync import BEGIN, CLEARED, find_notebooks


def _sync_cells(nb):
    return ["".join(c["source"]) for c in nb["cells"]
            if c["cell_type"] == "code" and BEGIN in "".join(c["source"])]


def test_every_sync_cell_is_the_cleared_stub():
    paths = find_notebooks([])
    assert paths, "no notebooks found"
    for p in paths:
        cells = _sync_cells(json.loads(p.read_text(encoding="utf-8")))
        assert cells, f"{p.name} has no sync cell"
        for src in cells:
            assert src == CLEARED, f"{p.name} holds packed code in its sync cell"
