"""Checksum-verified downloads of pinned dataset files (DECISIONS D-016).

Every file we download has its revision, size and SHA-256 recorded in a config under
``configs/data/``. :func:`fetch` streams to a temp file while hashing, verifies size and
hash, and only then moves the file into place, so a partial or tampered download never
lands at the destination. A valid existing file is reused without touching the network.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "raw"
INDICQA_CONFIG = REPO_ROOT / "configs" / "data" / "indicqa.yaml"

USER_AGENT = "indichallu-downloader"  # generic; no personal details (CLAUDE.md privacy)
CHUNK = 1 << 16


class ChecksumError(RuntimeError):
    """A downloaded or cached file does not match its pinned size/hash."""


@dataclass(frozen=True)
class PinnedFile:
    path: str  # path inside the source repo
    size: int
    sha256: str


@dataclass(frozen=True)
class PinnedDataset:
    repo: str
    repo_type: str
    revision: str
    licence: str
    files: dict[str, PinnedFile]  # keyed by language code

    def url(self, lang: str) -> str:
        return hf_resolve_url(self.repo, self.files[lang].path, self.revision, self.repo_type)

    def local_path(self, lang: str, data_root: str | Path = DEFAULT_DATA_ROOT) -> Path:
        """``<data_root>/<repo name>/<revision[:12]>/<file name>``: versioned on disk."""
        name = self.repo.split("/")[-1].lower()
        return Path(data_root) / name / self.revision[:12] / Path(self.files[lang].path).name


def hf_resolve_url(repo: str, path: str, revision: str, repo_type: str = "dataset") -> str:
    prefix = {"dataset": "datasets/", "model": "", "space": "spaces/"}[repo_type]
    return f"https://huggingface.co/{prefix}{repo}/resolve/{revision}/{path}"


def load_pinned_dataset(config: str | Path = INDICQA_CONFIG) -> PinnedDataset:
    raw = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    if not re.fullmatch(r"[0-9a-f]{40}", str(raw["revision"])):
        raise ValueError(f"{config}: revision must be a full 40-char commit hash")
    files = {}
    for lang, f in raw["files"].items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(f["sha256"])):
            raise ValueError(f"{config}: {lang} sha256 must be 64 hex chars")
        if not isinstance(f["size"], int) or f["size"] <= 0:
            raise ValueError(f"{config}: {lang} size must be a positive int")
        files[lang] = PinnedFile(path=f["path"], size=f["size"], sha256=f["sha256"])
    return PinnedDataset(
        repo=raw["repo"],
        repo_type=raw.get("repo_type", "dataset"),
        revision=raw["revision"],
        licence=raw["licence"],
        files=files,
    )


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def is_valid(path: str | Path, *, sha256: str, size: int | None = None) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    if size is not None and path.stat().st_size != size:
        return False
    return sha256_file(path) == sha256


def verified_local_path(
    dataset: PinnedDataset, lang: str, data_root: str | Path = DEFAULT_DATA_ROOT
) -> Path:
    """The local copy of ``lang``'s file, checked against its pinned size and SHA-256.

    Use this, not :meth:`PinnedDataset.local_path`, before reading cached data: a file
    changed or corrupted after download must not be analysed as if it were the pinned
    revision. Raises FileNotFoundError if missing, :class:`ChecksumError` if it differs.
    """
    path = dataset.local_path(lang, data_root)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not downloaded (python scripts/download_indicqa.py)")
    pinned = dataset.files[lang]
    if not is_valid(path, sha256=pinned.sha256, size=pinned.size):
        raise ChecksumError(
            f"{path} does not match the pinned size/sha256 for {dataset.repo}@"
            f"{dataset.revision[:12]}; delete it and re-run scripts/download_indicqa.py"
        )
    return path


def fetch(
    url: str,
    dest: str | Path,
    *,
    sha256: str,
    size: int | None = None,
    timeout: float = 60.0,
) -> tuple[Path, bool]:
    """Download ``url`` to ``dest`` and verify it. Returns ``(dest, downloaded)``.

    If ``dest`` already exists with the right size and hash, nothing is downloaded. A
    mismatching existing file is replaced. On a size/hash mismatch of the new download,
    :class:`ChecksumError` is raised and ``dest`` is left untouched.
    """
    dest = Path(dest)
    if is_valid(dest, sha256=sha256, size=size):
        return dest, False
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".part")
    try:
        h = hashlib.sha256()
        n = 0
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as resp, os.fdopen(fd, "wb") as out:
            for block in iter(lambda: resp.read(CHUNK), b""):
                h.update(block)
                n += len(block)
                out.write(block)
        if size is not None and n != size:
            raise ChecksumError(f"{url}: got {n} bytes, expected {size}")
        if h.hexdigest() != sha256:
            raise ChecksumError(f"{url}: sha256 {h.hexdigest()} != pinned {sha256}")
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return dest, True
