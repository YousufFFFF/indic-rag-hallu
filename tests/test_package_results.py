"""T44: a GPU session's results are packaged completely and nothing from earlier sessions."""

import importlib.util
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pr", ROOT / "scripts" / "package_results.py")
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)


def make_run(root: Path, run_id: str, started: str, status: str = "completed") -> None:
    d = root / "results" / "runs" / run_id
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"started_at": started, "status": status}), "utf-8")
    (d / "out.jsonl").write_text("{}\n", "utf-8")


@pytest.fixture
def repo(tmp_path):
    make_run(tmp_path, "old", "2026-11-24T09:00:00+00:00")
    make_run(tmp_path, "new", "2026-11-25T10:30:00+00:00")
    make_run(tmp_path, "crashed", "2026-11-25T11:00:00+00:00", status="failed")
    (tmp_path / "results" / "m0").mkdir()
    (tmp_path / "results" / "m0" / "pilot.json").write_text("{}", "utf-8")
    return tmp_path


def test_only_this_sessions_runs_and_the_extra_paths_are_packaged(repo, capsys):
    out = repo / "s.zip"
    since = datetime(2026, 11, 25, 10, 0, tzinfo=UTC)
    names = pr.package(out, since, ["results/m0"], root=repo)
    assert names == [
        "results/m0/pilot.json",
        "results/runs/crashed/meta.json",
        "results/runs/crashed/out.jsonl",
        "results/runs/new/meta.json",
        "results/runs/new/out.jsonl",
    ]
    assert sorted(zipfile.ZipFile(out).namelist()) == names  # unzip at the repo root
    assert "crashed: failed  <- check" in capsys.readouterr().out  # reported, not hidden


def test_since_must_carry_a_timezone(repo):
    with pytest.raises(ValueError, match="timezone"):
        pr.package(repo / "s.zip", datetime(2026, 11, 25), [], root=repo)


def test_paths_outside_the_repo_or_missing_are_refused(repo, tmp_path):
    since = datetime(2026, 11, 25, tzinfo=UTC)
    with pytest.raises(ValueError, match="outside"):
        pr.package(repo / "s.zip", since, ["../elsewhere"], root=repo)
    with pytest.raises(FileNotFoundError):
        pr.package(repo / "s.zip", since, ["results/nope"], root=repo)


def test_an_empty_session_is_an_error_not_an_empty_zip(repo):
    with pytest.raises(RuntimeError, match="nothing to package"):
        pr.package(repo / "s.zip", datetime(2027, 1, 1, tzinfo=UTC), [], root=repo)


def test_after_a_failed_command_missing_outputs_are_skipped_not_fatal(repo, capsys):
    """Codex on #33: a crash must still produce a zip with the failed run's folder."""
    since = datetime(2026, 11, 25, 10, 45, tzinfo=UTC)
    names = pr.package(
        repo / "s.zip", since, ["results/never_written"], root=repo, allow_missing=True
    )
    assert names == ["results/runs/crashed/meta.json", "results/runs/crashed/out.jsonl"]
    assert "results/never_written: not found, skipped" in capsys.readouterr().out
