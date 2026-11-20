"""T10: pinned, checksum-verified downloads (no network: a local HTTP server)."""

import hashlib
import http.server
import re
import threading
from functools import partial
from pathlib import Path

import pytest

from indichallu.data.download import (
    ChecksumError,
    fetch,
    hf_resolve_url,
    is_valid,
    load_pinned_dataset,
    sha256_file,
)
from indichallu.languages import load_languages

PAYLOAD = '{"version": "1", "data": ["आयु २१ वर्ष"]}'.encode()
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


class _CountingHandler(http.server.SimpleHTTPRequestHandler):
    requests: list[str] = []

    def do_GET(self):  # noqa: N802 (stdlib name)
        type(self).requests.append(self.path)
        super().do_GET()

    def log_message(self, *args):  # keep test output quiet
        pass


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "srv"
    root.mkdir()
    (root / "file.json").write_bytes(PAYLOAD)
    _CountingHandler.requests = []
    handler = partial(_CountingHandler, directory=str(root))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", _CountingHandler.requests
    httpd.shutdown()


def test_fetch_downloads_and_verifies(server, tmp_path):
    base, requests = server
    dest = tmp_path / "out" / "file.json"
    path, downloaded = fetch(f"{base}/file.json", dest, sha256=PAYLOAD_SHA, size=len(PAYLOAD))
    assert downloaded and path == dest and dest.read_bytes() == PAYLOAD
    assert len(requests) == 1
    assert list(dest.parent.iterdir()) == [dest]  # no leftover temp file


def test_valid_existing_file_is_not_downloaded_again(server, tmp_path):
    base, requests = server
    dest = tmp_path / "file.json"
    dest.write_bytes(PAYLOAD)
    _, downloaded = fetch(f"{base}/file.json", dest, sha256=PAYLOAD_SHA, size=len(PAYLOAD))
    assert not downloaded and requests == []


def test_corrupt_existing_file_is_replaced(server, tmp_path):
    base, requests = server
    dest = tmp_path / "file.json"
    dest.write_bytes(b"truncated")
    _, downloaded = fetch(f"{base}/file.json", dest, sha256=PAYLOAD_SHA, size=len(PAYLOAD))
    assert downloaded and dest.read_bytes() == PAYLOAD and len(requests) == 1


@pytest.mark.parametrize(
    ("sha", "size"),
    [("0" * 64, len(PAYLOAD)), (PAYLOAD_SHA, len(PAYLOAD) + 1)],
    ids=["wrong-hash", "wrong-size"],
)
def test_mismatch_raises_and_leaves_destination_untouched(server, tmp_path, sha, size):
    base, _ = server
    dest = tmp_path / "out" / "file.json"
    with pytest.raises(ChecksumError):
        fetch(f"{base}/file.json", dest, sha256=sha, size=size)
    assert not dest.exists()
    assert list(dest.parent.iterdir()) == []  # temp file cleaned up


def test_http_error_leaves_nothing_behind(server, tmp_path):
    base, _ = server
    dest = tmp_path / "out" / "missing.json"
    with pytest.raises(OSError):  # urllib.error.HTTPError (404) is an OSError
        fetch(f"{base}/missing.json", dest, sha256=PAYLOAD_SHA)
    assert list(dest.parent.iterdir()) == []


def test_hash_helpers(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(PAYLOAD)
    assert sha256_file(p) == PAYLOAD_SHA
    assert is_valid(p, sha256=PAYLOAD_SHA, size=len(PAYLOAD))
    assert not is_valid(p, sha256=PAYLOAD_SHA, size=1)
    assert not is_valid(tmp_path / "nope", sha256=PAYLOAD_SHA)


# --- the pinned IndicQA config --------------------------------------------------------


def test_indicqa_config_pins_every_configured_language():
    ds = load_pinned_dataset()
    assert ds.repo == "ai4bharat/IndicQA" and ds.licence == "CC-BY-4.0"
    assert re.fullmatch(r"[0-9a-f]{40}", ds.revision)
    # same revision as recorded in docs/RESOURCES.md
    resources = (Path(__file__).resolve().parents[2] / "docs" / "RESOURCES.md").read_text(
        encoding="utf-8"
    )
    assert "`ai4bharat/IndicQA` | " in resources and f"`{ds.revision[:12]}`" in resources
    assert set(ds.files) == set(load_languages())
    for lang, f in ds.files.items():
        assert f.path == f"data/indicqa.{lang}.json"
        assert re.fullmatch(r"[0-9a-f]{64}", f.sha256) and f.size > 0
    assert len({f.sha256 for f in ds.files.values()}) == len(ds.files)  # no copy-paste dupes


def test_urls_and_local_paths_are_pinned():
    ds = load_pinned_dataset()
    assert ds.url("hi") == (
        f"https://huggingface.co/datasets/ai4bharat/IndicQA/resolve/{ds.revision}"
        "/data/indicqa.hi.json"
    )
    assert ds.local_path("hi", "/d") == Path(f"/d/indicqa/{ds.revision[:12]}/indicqa.hi.json")
    assert hf_resolve_url("org/m", "w.bin", "abc", "model").startswith(
        "https://huggingface.co/org/m/"
    )


def test_bad_configs_are_rejected(tmp_path):
    good = (Path(__file__).resolve().parents[2] / "configs" / "data" / "indicqa.yaml").read_text(
        encoding="utf-8"
    )
    ds = load_pinned_dataset()
    for bad in (
        good.replace(ds.revision, ds.revision[:12]),  # short revision
        good.replace(ds.files["hi"].sha256, "abc"),  # malformed hash
    ):
        p = tmp_path / "c.yaml"
        p.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError):
            load_pinned_dataset(p)


@pytest.mark.network
def test_real_hindi_download_matches_pin(tmp_path):
    """Opt-in (pytest -m network): fetch the real Hindi file and verify its pinned hash."""
    ds = load_pinned_dataset()
    f = ds.files["hi"]
    path, _ = fetch(ds.url("hi"), ds.local_path("hi", tmp_path), sha256=f.sha256, size=f.size)
    assert sha256_file(path) == f.sha256
