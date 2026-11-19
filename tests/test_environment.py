"""T01 smoke tests: the package installs and the pinned environment is what we expect."""

import sys

import indichallu


def test_python_is_3_11():
    assert sys.version_info[:2] == (3, 11)


def test_package_imports_with_version():
    assert indichallu.__version__ == "0.0.1"


def test_core_dependencies_import():
    import faiss  # noqa: F401
    import indicnlp  # noqa: F401
    import numpy  # noqa: F401
    import sentence_transformers  # noqa: F401
    import sklearn  # noqa: F401
    import torch
    import transformers  # noqa: F401

    # Docker image uses the CPU build of torch.
    assert torch.__version__.startswith("2.15.0")
