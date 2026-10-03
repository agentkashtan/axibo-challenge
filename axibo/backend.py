"""Single place that decides which Genesis backend and torch device to use.

The project runs on two machines - an Apple M4 (Metal / MPS) and a Linux box with an RTX 4090 (CUDA) - and every
script used to hardcode one of them, so each file copied between the two needed a sed. Both are picked here
instead, by detection, with an env override for the cases detection gets wrong:

    AXIBO_BACKEND=cuda|metal|cpu     # Genesis backend
    AXIBO_DEVICE=cuda|mps|cpu        # torch device (defaults to match the backend)
"""

import os
import platform


def gs_backend():
    """The Genesis backend enum for this machine (import genesis lazily: gs.init must run before scene code)."""
    import genesis as gs

    name = os.environ.get("AXIBO_BACKEND")
    if name:
        return getattr(gs, name)
    if _cuda_available():
        return gs.cuda
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return gs.metal
    return gs.cpu


def torch_device() -> str:
    """The torch device string for this machine: "cuda", "mps" or "cpu"."""
    name = os.environ.get("AXIBO_DEVICE")
    if name:
        return name
    if _cuda_available():
        return "cuda"
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False
