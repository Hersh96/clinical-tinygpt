from pathlib import Path
import pytest

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def tiny_fixture_cfg():
    return {
        "seed": 0,
        "sources": [
            {"name": "fixture_notes", "kind": "jsonl", "path": str(FIX / "notes.jsonl"),
             "id_field": "idx", "text_field": "note", "summary_field": "summary",
             "license": "synthetic test fixture"}
        ],
    }
