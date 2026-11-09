from pathlib import Path

import pytest

from experiments.modal_validation import PRIVATE_RESEARCH, REMOTE, source_hashes, verify_snapshot


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


def test_private_forms_are_excluded_without_excluding_public_research(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("config")
    (tmp_path / "uv.lock").write_text("lock")
    research = tmp_path / "_research"
    research.mkdir()
    (research / "HF-ARTICLE.md").write_text("public article")
    for name in ("SUBMISSION-FORM-20261106.md", "SUBMISSION-FORM-personal.md"):
        (research / name).write_text("private fixture")
    hashes = source_hashes(tmp_path)
    assert "_research/HF-ARTICLE.md" in hashes
    assert not any(Path(name).match(PRIVATE_RESEARCH) for name in hashes)
    (research / "SUBMISSION-FORM-20261106.md").write_text("edited private fixture")
    verify_snapshot(tmp_path, hashes)


@pytest.mark.skipif(not Path(REMOTE).is_dir(), reason="Requires the actual Modal upload")
def test_modal_upload_has_no_private_submission_forms():
    assert (Path(REMOTE) / "_research/HF-ARTICLE.md").is_file()
    assert not list((Path(REMOTE) / "_research").rglob(PRIVATE_RESEARCH))
