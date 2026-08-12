"""
JIT loader for the fused CPU TensorFold kernel.

Usage:

    from tensorfold.fused_backend import fused_available, fused_forward

    if fused_available():
        y = fused_forward(x, U, V, bias)

The extension is compiled lazily, on first use, with
torch.utils.cpp_extension.load(). Compilation is attempted at most once
per process; if it fails, fused_available() returns False for the rest
of the process and callers should fall back to the pure-PyTorch path.
"""

import platform
import sys
import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch

_ext: Optional[Any] = None
_load_attempted = False
_load_error: Optional[Exception] = None

_CSRC_DIR = Path(__file__).resolve().parent / "csrc"

# Set to True after a failed build if the failure looks like the
# classic "cl.exe not on PATH" MSVC problem, so we can give a
# specific, actionable message instead of a generic one.
_looks_like_missing_msvc_env = False


def _compiler_flags() -> Tuple[List[str], List[str]]:
    """
    Return (extra_cflags, extra_ldflags) for the current toolchain.

    GCC/Clang (Linux, macOS, MinGW) use "-flag" syntax.
    MSVC (cl.exe, the default on Windows) uses "/flag" syntax and has
    no equivalent to -march=native — it auto-vectorizes reasonably at
    /O2 without needing one, so we just omit it there. -march=native
    matters most for the ARM64 CI build anyway (GCC/Clang on Linux),
    not for local Windows iteration.
    """
    if sys.platform == "win32":
        # /openmp enables the #pragma omp used inside ATen's
        # parallel_for header when libtorch's threading backend is
        # OpenMP-based. Harmless no-op if it isn't.
        return (["/O2", "/openmp"], [])

    # GCC/Clang path (Linux ARM64 CI, Linux x86_64, macOS).
    extra_cflags = ["-O3", "-fopenmp", "-march=native"]
    extra_ldflags = ["-fopenmp"]
    return extra_cflags, extra_ldflags


def _build_extension() -> None:
    global _ext, _load_attempted, _load_error, _looks_like_missing_msvc_env

    if _load_attempted:
        return
    _load_attempted = True

    try:
        from torch.utils.cpp_extension import load

        extra_cflags, extra_ldflags = _compiler_flags()

        _ext = load(
            name="tensorfold_fused",
            sources=[str(_CSRC_DIR / "fused_linear.cpp")],
            extra_cflags=extra_cflags,
            extra_ldflags=extra_ldflags,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 - any build/import failure
        _load_error = exc

        msg = str(exc).lower()
        _looks_like_missing_msvc_env = sys.platform == "win32" and (
            "cl.exe" in msg
            or "cl : command line error" in msg
            or "microsoft visual c++" in msg
            or "vcvarsall" in msg
            or "'where', 'cl'" in msg
            or "winerror 2" in msg
            or ("error(s) building extension" in msg and "cl" in msg)
        )

        if _looks_like_missing_msvc_env:
            warnings.warn(
                "TensorFold fused CPU kernel could not be built: "
                "cl.exe was not found or the MSVC environment isn't "
                "initialized. Run this from the 'x64 Native Tools "
                "Command Prompt for VS' (or run vcvarsall.bat x64 "
                "first), then re-run — the same requirement as any "
                "MSVC-based extension build. If you switched terminals "
                "since a previous failed attempt, also clear the "
                "stale build cache: delete "
                "%LOCALAPPDATA%\\torch_extensions\\tensorfold_fused "
                "before rebuilding. "
                "Falling back to the pure-PyTorch two-matmul path in "
                "the meantime — this does not affect correctness, "
                "only local speed. Note the ARM64 CI build uses "
                "GCC/Clang and is unaffected by this.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                "TensorFold fused CPU kernel could not be built "
                f"({exc.__class__.__name__}: {exc}). "
                "Falling back to the pure-PyTorch two-matmul path. "
                "This does not affect correctness, only speed.",
                RuntimeWarning,
                stacklevel=2,
            )


def fused_available() -> bool:
    """Return True if the fused kernel is built and usable."""
    _build_extension()
    return _ext is not None


def fused_forward(
    x: torch.Tensor,
    U: torch.Tensor,
    V: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute (x @ U) @ V + bias using the fused CPU kernel.

    Raises RuntimeError if the extension is unavailable; callers
    should check fused_available() first (TensorFoldLinear does this
    automatically and falls back transparently).
    """
    _build_extension()
    if _ext is None:
        raise RuntimeError(
            f"Fused extension unavailable: {_load_error!r}"
        )
    return _ext.fused_forward(x, U, V, bias)


def platform_summary() -> str:
    """Small helper for logging/benchmark headers."""
    return (
        f"machine={platform.machine()} "
        f"system={platform.system()} "
        f"fused_available={fused_available()}"
    )
