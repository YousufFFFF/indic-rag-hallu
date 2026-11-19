# Hallucination Detection in RAG for Low-Resource Indic Languages

A claim-level framework that checks whether an LLM answer in an Indian language is actually supported by the passages retrieved for it, with span-level attribution.

> Work in progress — Stage I review complete.

## Development

Everything runs in Docker; no local Python install is needed. Requires Docker Desktop (with Compose v2).

```sh
docker compose build                 # build the dev image (Python 3.11, CPU PyTorch, pinned deps)
docker compose run --rm dev pytest   # run the test suite
docker compose run --rm dev ruff check .
```

- Source is bind-mounted, so code edits take effect without rebuilding. Rebuild after changing dependencies.
- Dependencies: direct pins live in `pyproject.toml`; the full frozen set is `requirements-lock.txt`. After changing a pin, run `sh scripts/lock_deps.sh` and rebuild.
- Secrets (for example `HF_TOKEN`) go in a git-ignored `.env` file, which the container loads automatically if present.
- The generator runs in [Ollama](https://ollama.com) on the host; the container reaches it at `http://host.docker.internal:11434/v1`.
