"""Shared pytest fixtures and lightweight stubs for heavy optional dependencies.

``generate_hidden_text`` imports ``torch``, ``diffusers`` and ``controlnet_aux``
at module scope. Those packages are large (and require a GPU to be useful), so
for unit testing we register minimal stand-ins in ``sys.modules`` *before* the
module under test is imported. Only the small, pure-Python helpers exercised by
the tests are run for real; the heavy machine-learning code paths are mocked.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest


def _install_stub(name):
    """Register a MagicMock-backed stub module under ``name`` if it is missing."""
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    module.__dict__["__getattr__"] = lambda attr: MagicMock(name=f"{name}.{attr}")
    sys.modules[name] = module
    return module


# Install stubs for the heavy dependencies that are not needed for unit tests.
def _ensure_heavy_stubs():
    try:
        import torch  # noqa: F401
    except ImportError:
        torch_stub = _install_stub("torch")
        torch_stub.float16 = "float16"
        torch_stub.float32 = "float32"
        cuda = types.SimpleNamespace(is_available=lambda: False)
        torch_stub.cuda = cuda
        torch_stub.Generator = MagicMock(name="torch.Generator")

    try:
        import diffusers  # noqa: F401
    except ImportError:
        diffusers_stub = _install_stub("diffusers")
        diffusers_stub.StableDiffusionControlNetPipeline = MagicMock(
            name="StableDiffusionControlNetPipeline"
        )
        diffusers_stub.ControlNetModel = MagicMock(name="ControlNetModel")

    try:
        import controlnet_aux  # noqa: F401
    except ImportError:
        controlnet_stub = _install_stub("controlnet_aux")
        controlnet_stub.CannyDetector = MagicMock(name="CannyDetector")


_ensure_heavy_stubs()


@pytest.fixture
def generate_module():
    """Import and return the module under test after stubs are installed."""
    _ensure_heavy_stubs()
    import generate_hidden_text

    return generate_hidden_text
