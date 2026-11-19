"""Run bookkeeping: seeding, config overrides, and one folder per run under results/runs/.

Every run writes ``results/runs/<run_id>/`` containing:

* ``config.yaml``  the fully resolved config (after command-line overrides)
* ``meta.json``    git commit + dirty flag, models and their revisions, seed, environment
                   (Python, packages, GPU), command line, start/end time and status
* ``git.diff``     only when the working tree had uncommitted changes

Typical use::

    cfg = apply_overrides(load_config("configs/experiments/m0.yaml"), ["retrieval.k=10"])
    with start_run(cfg, name="m0-hi") as run:
        run.add_model("generator", "qwen3:4b", revision="1cfa9a720891", backend="ollama")
        ...  # write outputs to run.path("predictions.jsonl")
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "runs"

# Packages whose exact versions are recorded in every run's metadata.
TRACKED_PACKAGES = (
    "indichallu",
    "numpy",
    "torch",
    "transformers",
    "datasets",
    "sentence-transformers",
    "faiss-cpu",
    "indic-nlp-library",
    "scikit-learn",
    "pydantic",
    "openai",
)

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")


# --- seeding -------------------------------------------------------------------------


MAX_SEED = 2**32 - 1  # NumPy's legacy RNG accepts 32-bit seeds only


def validate_seed(seed: object) -> int:
    """Return ``seed`` if it is an int in ``[0, 2**32 - 1]``, else raise ValueError."""
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be an int in [0, {MAX_SEED}], got {seed!r}")
    return seed


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA).

    ``deterministic`` is applied explicitly on every call, so a previous call can't leave
    the process in a different mode than the one recorded. With ``deterministic=True``,
    PyTorch runs in strict mode: an operation without a deterministic implementation
    raises instead of silently running a nondeterministic kernel. (Set
    ``CUBLAS_WORKSPACE_CONFIG`` before CUDA initialises; it is set here only as a default.)
    """
    validate_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # affects subprocesses only
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a pinned dependency
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


# --- config --------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: config must be a YAML mapping")
    return data


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Return a copy of ``config`` with ``dotted.key=value`` overrides applied.

    Values are parsed as YAML (``k=10`` -> int, ``flag=true`` -> bool, ``xs=[1,2]`` -> list).
    Only existing keys can be overridden, so a typo fails instead of being silently added.
    """
    out = copy.deepcopy(config)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ValueError(f"override {item!r} must look like 'dotted.key=value'")
        *parents, leaf = key.split(".")
        node: Any = out
        for i, part in enumerate(parents):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"override {item!r}: unknown key {'.'.join(parents[: i + 1])!r}")
            node = node[part]
        if not isinstance(node, dict) or leaf not in node:
            raise KeyError(f"override {item!r}: unknown key {key!r}")
        node[leaf] = yaml.safe_load(raw)
    return out


def config_hash(config: dict[str, Any]) -> str:
    """Short, order-independent hash of a resolved config."""
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


# --- provenance ----------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def git_info(repo: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Commit, branch and dirty state of ``repo``. Raises if it is not a git checkout."""
    repo = Path(repo)
    try:
        commit = _git(repo, "rev-parse", "HEAD").strip()
    except FileNotFoundError as e:
        raise RuntimeError("git is not installed; runs must record their commit") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"{repo} is not a git repository with commits: {e.stderr}") from e
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    status = _git(repo, "status", "--porcelain")
    return {"commit": commit, "branch": branch, "dirty": bool(status.strip())}


def git_diff(repo: str | Path = REPO_ROOT) -> str:
    """Uncommitted changes to tracked files, plus the list of untracked files."""
    repo = Path(repo)
    diff = _git(repo, "diff", "HEAD")
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard")
    if untracked.strip():
        diff += "\n# Untracked files (contents not recorded):\n"
        diff += "".join(f"#   {line}\n" for line in untracked.splitlines())
    return diff


def environment_info() -> dict[str, Any]:
    packages = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    gpus: list[str] = []
    cuda = None
    try:
        import torch

        if torch.cuda.is_available():
            cuda = torch.version.cuda
            gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:  # pragma: no cover
        pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda,
        "gpus": gpus,
    }


# --- the run -------------------------------------------------------------------------


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class Run:
    """A run folder. Use via :func:`start_run` as a context manager."""

    def __init__(self, run_id: str, directory: Path, meta: dict[str, Any]):
        self.id = run_id
        self.dir = directory
        self.meta = meta

    def path(self, name: str) -> Path:
        """Path for an output file inside the run folder (subfolders are created)."""
        p = self.dir / name
        if self.dir.resolve() not in p.resolve().parents:
            raise ValueError(f"{name!r} escapes the run directory")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def add_model(self, role: str, model_id: str, *, revision: str, **extra: Any) -> None:
        """Record a model used in this run. ``revision`` is mandatory (CLAUDE.md §2)."""
        if not role or not model_id:
            raise ValueError("role and model_id are required")
        if not revision or not str(revision).strip():
            raise ValueError(f"model {model_id!r} needs a pinned revision")
        if role in self.meta["models"]:
            raise ValueError(f"model role {role!r} already recorded")
        self.meta["models"][role] = {"id": model_id, "revision": str(revision), **extra}
        self._save()

    def finish(self, status: str = "completed") -> None:
        self.meta["status"] = status
        self.meta["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        self._save()

    def _save(self) -> None:
        _write_json_atomic(self.dir / "meta.json", self.meta)

    def __enter__(self) -> Run:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.finish("completed")
        else:
            self.meta["error"] = f"{exc_type.__name__}: {exc}"
            self.finish("failed")


def start_run(
    config: dict[str, Any],
    *,
    name: str,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    repo: str | Path = REPO_ROOT,
    require_clean: bool = False,
    deterministic: bool = False,
) -> Run:
    """Create ``<results_root>/<run_id>/``, seed everything and record provenance.

    The config must contain an integer ``seed``. ``require_clean=True`` refuses to start
    from a working tree with uncommitted changes (use it for reported experiments).
    """
    if not _NAME_RE.match(name):
        raise ValueError(f"run name {name!r} must be lowercase letters, digits and '-'")
    if "seed" not in config:
        raise ValueError("config must set 'seed' (CLAUDE.md §2: every run is seeded)")
    validate_seed(config["seed"])  # before anything touches the filesystem

    git = git_info(repo)
    if require_clean and git["dirty"]:
        raise RuntimeError("working tree has uncommitted changes; commit before this run")

    started = datetime.now(UTC)
    run_id = f"{started:%Y%m%d-%H%M%S}-{name}-{config_hash(config)}"
    directory = Path(results_root) / run_id
    directory.mkdir(parents=True, exist_ok=False)  # never overwrite an existing run

    set_seed(config["seed"], deterministic=deterministic)

    (directory / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    if git["dirty"]:
        (directory / "git.diff").write_text(git_diff(repo), encoding="utf-8")

    meta = {
        "run_id": run_id,
        "name": name,
        "status": "running",
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": None,
        "seed": config["seed"],
        "deterministic": deterministic,
        "config_hash": config_hash(config),
        "git": git,
        "models": {},
        "environment": environment_info(),
        "command": sys.argv,
    }
    run = Run(run_id, directory, meta)
    run._save()
    return run
