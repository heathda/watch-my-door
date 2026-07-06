"""Shared pytest fixtures."""
import sys
from pathlib import Path

import pytest

# Repo root importable (pytest.ini also sets pythonpath; this is a safety net).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import db  # noqa: E402


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Point db at a throwaway file so tests never touch the real events.db."""
    path = tmp_path / "events.db"
    monkeypatch.setattr(db, "DB_FILE", path)
    return path


@pytest.fixture
def sample_image(tmp_path):
    """A real on-disk file (contents irrelevant; classify is mocked)."""
    p = tmp_path / "alert.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    return p
