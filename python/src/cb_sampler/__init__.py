"""Python API for the continuous-binomial posterior sampler."""
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
import secrets
from typing import Any

import numpy as np

from . import _backend

__version__ = "0.1.0a1"

__all__ = ["CbStream", "draw_streams", "sample", "__version__"]

UINT64_MOD = 1 << 64


def _coerce_uint64(value: int | None, *, name: str, default: int = 0) -> int:
    if value is None:
        return default
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0 or value >= UINT64_MOD:
        raise ValueError(f"{name} must be in [0, 2**64)")
    return value


def _validate_stats(chi: float, nu: float) -> tuple[float, float]:
    chi = float(chi)
    nu = float(nu)
    if not np.isfinite(chi) or not np.isfinite(nu) or nu < 0.0 or chi < 0.0 or chi > nu:
        raise ValueError("Require finite sufficient statistics 0 <= chi <= nu")
    return chi, nu


@dataclass(frozen=True)
class Peek:
    chi: float
    nu: float
    regime: int
    sigma: float


class CbStream:
    """Owned sampler for the closed continuous-binomial conjugate family.

    ``(chi, nu) == (0, 0)`` is the startup prior.  With ``nu > 0``, the
    boundary states ``chi == 0`` and ``chi == nu`` are the weak-limit point
    masses at zero and one; interior states use the ordinary sampler.
    """

    def __init__(
        self,
        chi: float,
        nu: float,
        seed: int | None = None,
        stream_idx: int = 0,
        buf_size: int = 256,
    ) -> None:
        chi, nu = _validate_stats(chi, nu)
        seed_u64 = secrets.randbits(64) if seed is None else _coerce_uint64(seed, name="seed")
        stream_idx_u64 = _coerce_uint64(stream_idx, name="stream_idx")
        if not isinstance(buf_size, (int, np.integer)):
            raise TypeError("buf_size must be an integer")
        if int(buf_size) < 0:
            raise ValueError("buf_size must be nonnegative")
        self._backend = _backend.BackendStream(chi, nu, seed_u64, stream_idx_u64, int(buf_size))
        self._closed = False

    def __enter__(self) -> "CbStream":
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("CB stream is closed")

    def close(self) -> None:
        if not self._closed:
            self._backend.close()
            self._closed = True

    def draw(self, n: int = 1) -> np.ndarray:
        self._require_open()
        if not isinstance(n, (int, np.integer)):
            raise TypeError("draw count must be an integer")
        n = int(n)
        if n <= 0:
            raise ValueError("draw count must be a positive integer")
        return np.asarray(self._backend.draw(n), dtype=np.float64)

    def update(self, x: float) -> None:
        self._require_open()
        self._backend.update(float(x))

    def update_batch(self, x: Any) -> None:
        self._require_open()
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError("update_batch expects a one-dimensional observation array")
        self._backend.update_batch(arr.tolist())

    def set_stats(self, chi: float, nu: float) -> None:
        """Replace sufficient statistics while preserving the RNG stream."""

        self._require_open()
        chi, nu = _validate_stats(chi, nu)
        self._backend.set_stats(chi, nu)

    def peek(self) -> Peek:
        self._require_open()
        raw = self._backend.peek()
        return Peek(
            chi=float(raw["chi"]),
            nu=float(raw["nu"]),
            regime=int(raw["regime"]),
            sigma=float(raw["sigma"]),
        )

    def diagnostics(self) -> dict[str, int | bool]:
        self._require_open()
        return dict(self._backend.diagnostics())


def draw_streams(streams: Any) -> np.ndarray:
    """Draw once from every live stream in one C extension call."""

    streams = list(streams)
    if not streams:
        raise ValueError("streams must be nonempty")
    backends = []
    for stream in streams:
        if not isinstance(stream, CbStream):
            raise TypeError("all items must be CbStream instances")
        stream._require_open()
        backends.append(stream._backend)
    return np.asarray(_backend.draw_streams_c(backends), dtype=np.float64)


def sample(
    chi: float,
    nu: float,
    size: int | tuple[int, ...] | None = None,
    seed: int | None = None,
) -> np.ndarray | np.float64:
    """Draw samples from one fresh stream.

    With ``size=None`` this returns a NumPy scalar.  Integer and tuple sizes
    return ``float64`` arrays with the requested shape.
    """

    if size is None:
        shape: tuple[int, ...] = ()
        n = 1
    elif isinstance(size, (int, np.integer)):
        if int(size) <= 0:
            raise ValueError("size must be positive")
        shape = (int(size),)
        n = int(size)
    else:
        shape = tuple(int(dim) for dim in size)
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError("size dimensions must be positive")
        n = int(np.prod(shape))

    with CbStream(chi, nu, seed=seed) as stream:
        draws = stream.draw(n)
    if size is None:
        return np.float64(draws[0])
    return draws.reshape(shape)
