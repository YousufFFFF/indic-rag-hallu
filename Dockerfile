# syntax=docker/dockerfile:1
# Development / CPU test image. GPU runs happen on Colab/Kaggle (see docs/DECISIONS.md D-001, D-002).
# Base image pinned by digest: python:3.11-slim, Python 3.11.18.
FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface

WORKDIR /workspace

# 0) git: every run records the commit it ran from (src/indichallu/runs.py).
#    The repo is bind-mounted with a different owner, so mark it safe for git.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/* \
 && git config --system --add safe.directory /workspace

# 1) Fully pinned dependency set (CPU build of torch from the PyTorch index).
COPY requirements-lock.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
      -r requirements-lock.txt

# 2) The package itself, editable, so the bind-mounted source is what runs.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps -e .

CMD ["pytest"]
