"""Stubs so the tests import sartransfer without torch, numpy or huggingface_hub.

Everything tested here is pure Python (hash checks, .env parsing, source scans).
`sartransfer.models.encoders` evaluates `torch.bfloat16` and
`@torch.inference_mode()` at import time, so the torch stub answers any
attribute with a callable that also works as a decorator.
"""

import sys
import types


class _Anything:
    """Stands in for any torch attribute."""

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and callable(args[0]):
            return args[0]                   # @torch.inference_mode() leaves the function alone
        return self

    def __getattr__(self, name):
        return self


class _TorchStub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Anything()


sys.modules["torch"] = _TorchStub("torch")
sys.modules["numpy"] = types.ModuleType("numpy")
sys.modules["huggingface_hub"] = types.ModuleType("huggingface_hub")
