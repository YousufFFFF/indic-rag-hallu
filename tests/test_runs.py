"""T05: run folders, provenance and seeding."""

import json
import random
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from indichallu import runs
from indichallu.runs import apply_overrides, config_hash, git_info, set_seed, start_run

CONFIG = {"seed": 13, "lang": "hi", "retrieval": {"k": 5, "embedder": "BAAI/bge-m3"}}


class RepoPath(type(Path())):
    """A Path that also carries a ``git(*args)`` helper for the throwaway repo."""


@pytest.fixture
def repo(tmp_path):
    """A throwaway git repo with one commit, so provenance tests don't depend on ours."""
    root = RepoPath(tmp_path / "repo")
    root.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q", "-b", "main")
    (root / "a.txt").write_text("one\n", encoding="utf-8")
    git("add", "a.txt")
    git("commit", "-q", "-m", "init")
    root.git = git
    return root


def _draws():
    return (random.random(), np.random.rand(), torch.rand(1).item())


# --- seeding ---------------------------------------------------------------------------


def test_set_seed_makes_all_rngs_reproducible():
    set_seed(123)
    first = _draws()
    set_seed(123)
    assert _draws() == first
    set_seed(124)
    assert _draws() != first


@pytest.mark.parametrize("bad", [-1, 1.5, "7", True, None])
def test_set_seed_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        set_seed(bad)


# --- config overrides ------------------------------------------------------------------


def test_overrides_parse_yaml_types_and_do_not_mutate_input():
    out = apply_overrides(CONFIG, ["retrieval.k=10", "seed=7", "lang=mr"])
    assert out["retrieval"]["k"] == 10 and out["seed"] == 7 and out["lang"] == "mr"
    assert CONFIG["retrieval"]["k"] == 5  # original untouched


def test_overrides_reject_unknown_keys_and_bad_syntax():
    with pytest.raises(KeyError):
        apply_overrides(CONFIG, ["retrieval.top_k=10"])  # typo
    with pytest.raises(KeyError):
        apply_overrides(CONFIG, ["nope.k=1"])
    with pytest.raises(ValueError):
        apply_overrides(CONFIG, ["retrieval.k"])


def test_config_hash_ignores_key_order():
    reordered = {"retrieval": {"embedder": "BAAI/bge-m3", "k": 5}, "lang": "hi", "seed": 13}
    assert config_hash(reordered) == config_hash(CONFIG)
    assert config_hash(apply_overrides(CONFIG, ["seed=14"])) != config_hash(CONFIG)


# --- git provenance --------------------------------------------------------------------


def test_git_info_reports_commit_and_dirty_state(repo):
    info = git_info(repo)
    assert info["commit"] == repo.git("rev-parse", "HEAD")
    assert re.fullmatch(r"[0-9a-f]{40}", info["commit"])
    assert info["branch"] == "main" and info["dirty"] is False
    (repo / "a.txt").write_text("changed\n", encoding="utf-8")
    assert git_info(repo)["dirty"] is True


def test_git_info_fails_outside_a_repo(tmp_path):
    with pytest.raises(RuntimeError):
        git_info(tmp_path)


# --- run folders -----------------------------------------------------------------------


def test_run_folder_records_everything(repo, tmp_path):
    results = tmp_path / "results"
    with start_run(CONFIG, name="m0-hi", results_root=results, repo=repo) as run:
        run.add_model("generator", "qwen3:4b", revision="1cfa9a720891", backend="ollama")
        run.path("out/predictions.jsonl").write_text("{}\n", encoding="utf-8")

    assert re.fullmatch(rf"\d{{8}}-\d{{6}}-m0-hi-{config_hash(CONFIG)}", run.id)
    assert run.dir == results / run.id
    assert yaml.safe_load((run.dir / "config.yaml").read_text(encoding="utf-8")) == CONFIG
    assert (run.dir / "out" / "predictions.jsonl").exists()
    assert not (run.dir / "git.diff").exists()  # clean tree

    meta = json.loads((run.dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "completed" and meta["finished_at"]
    assert meta["seed"] == 13
    assert meta["git"] == {
        "commit": repo.git("rev-parse", "HEAD"),
        "branch": "main",
        "dirty": False,
    }
    assert meta["models"]["generator"] == {
        "id": "qwen3:4b",
        "revision": "1cfa9a720891",
        "backend": "ollama",
    }
    env = meta["environment"]
    assert env["python"].startswith("3.11.")
    assert env["packages"]["torch"].startswith("2.15.0")
    assert isinstance(env["gpus"], list)


def test_start_run_seeds_from_config(repo, tmp_path):
    start_run(CONFIG, name="a", results_root=tmp_path / "r", repo=repo).finish()
    first = _draws()
    set_seed(999)
    start_run(CONFIG, name="b", results_root=tmp_path / "r", repo=repo).finish()
    assert _draws() == first


def test_dirty_tree_saves_diff_and_can_be_refused(repo, tmp_path):
    (repo / "a.txt").write_text("changed\n", encoding="utf-8")
    (repo / "new.txt").write_text("x\n", encoding="utf-8")
    run = start_run(CONFIG, name="dirty", results_root=tmp_path / "r", repo=repo)
    diff = (run.dir / "git.diff").read_text(encoding="utf-8")
    assert "+changed" in diff and "new.txt" in diff
    assert json.loads((run.dir / "meta.json").read_text())["git"]["dirty"] is True
    with pytest.raises(RuntimeError):
        start_run(CONFIG, name="strict", results_root=tmp_path / "r", repo=repo, require_clean=True)


def test_failed_run_is_marked_and_error_propagates(repo, tmp_path):
    with pytest.raises(ZeroDivisionError):
        with start_run(CONFIG, name="boom", results_root=tmp_path / "r", repo=repo) as run:
            1 / 0  # noqa: B018
    meta = json.loads((run.dir / "meta.json").read_text())
    assert meta["status"] == "failed" and "ZeroDivisionError" in meta["error"]


def test_existing_run_folder_is_never_overwritten(repo, tmp_path, monkeypatch):
    fixed = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed

    monkeypatch.setattr(runs, "datetime", FrozenDatetime)
    start_run(CONFIG, name="same", results_root=tmp_path / "r", repo=repo)
    with pytest.raises(FileExistsError):
        start_run(CONFIG, name="same", results_root=tmp_path / "r", repo=repo)


def test_config_without_seed_is_refused(repo, tmp_path):
    with pytest.raises(ValueError):
        start_run({"lang": "hi"}, name="x", results_root=tmp_path / "r", repo=repo)


@pytest.mark.parametrize("name", ["Has Caps", "with space", "", "-lead", "a/b"])
def test_bad_run_names_are_refused(repo, tmp_path, name):
    with pytest.raises(ValueError):
        start_run(CONFIG, name=name, results_root=tmp_path / "r", repo=repo)


def test_models_need_a_revision_and_unique_role(repo, tmp_path):
    run = start_run(CONFIG, name="m", results_root=tmp_path / "r", repo=repo)
    with pytest.raises(ValueError):
        run.add_model("generator", "qwen3:4b", revision="")
    run.add_model("generator", "qwen3:4b", revision="abc")
    with pytest.raises(ValueError):
        run.add_model("generator", "other", revision="def")


def test_run_path_cannot_escape_the_run_folder(repo, tmp_path):
    run = start_run(CONFIG, name="p", results_root=tmp_path / "r", repo=repo)
    with pytest.raises(ValueError):
        run.path("../outside.txt")


def test_real_repo_provenance_works_in_the_container():
    """Our own checkout must be readable by git inside Docker/CI (safe.directory)."""
    info = git_info()
    assert re.fullmatch(r"[0-9a-f]{40}", info["commit"])
