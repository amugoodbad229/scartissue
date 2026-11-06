from pathlib import Path

import pytest

from experiments.modal_validation import REMOTE, source_hashes, verify_snapshot


def test_modal_remote_path_is_absolute_posix_even_from_windows():
    assert REMOTE.startswith("/") and chr(92) not in REMOTE


def test_remote_snapshot_detects_changed_and_added_source(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("config")
    (tmp_path / "uv.lock").write_text("lock")
    for folder in ("experiments", "tests", "artifacts"):
        (tmp_path / folder).mkdir()
    source = tmp_path / "experiments/example.py"
    source.write_text("original")
    expected = source_hashes(tmp_path)
    verify_snapshot(tmp_path, expected)
    source.write_text("changed")
    with pytest.raises(ValueError, match="snapshot mismatch"):
        verify_snapshot(tmp_path, expected)
    source.write_text("original")
    (tmp_path / "tests/extra.py").write_text("added")
    with pytest.raises(ValueError, match="extra.py"):
        verify_snapshot(tmp_path, expected)
