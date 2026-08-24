"""Numerical helpers for Python validation and examples."""
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

_BFT_MU = np.array(
    [
        -2.22077497298413459315e-02,
        -1.33068859543940699180e-03,
        -1.12776786425130941260e-02,
    ],
    dtype=np.float64,
)
_BFT_R = np.array(
    [
        7.70792205297306590173e-03,
        2.08031967001529281835e-02,
        1.31555479135406720032e-02,
    ],
    dtype=np.float64,
)
_BFT_D1 = np.array(
    [
        2.0 / 24.0,
        -4.0 / 2880.0,
        6.0 / 181440.0,
        -8.0 / 9676800.0,
        10.0 / 479001600.0,
        -12.0 * 691.0 / 15692092416000.0,
        14.0 * 7.0 / 7322976460800.0,
    ],
    dtype=np.float64,
)
_BFT_D2 = np.array(
    [
        2.0 / 24.0,
        -12.0 / 2880.0,
        30.0 / 181440.0,
        -56.0 / 9676800.0,
        90.0 / 479001600.0,
        -132.0 * 691.0 / 15692092416000.0,
        182.0 * 7.0 / 7322976460800.0,
    ],
    dtype=np.float64,
)


def bft(eta: np.ndarray | float) -> np.ndarray | float:
    """Evaluate B(eta) = log((exp(eta)-1)/eta)."""

    scalar = np.isscalar(eta)
    eta_arr = np.asarray(eta, dtype=np.float64)
    out = np.empty_like(eta_arr, dtype=np.float64)
    small = np.abs(eta_arr) < 1.0
    if np.any(small):
        u = eta_arr[small] * eta_arr[small]
        g = (
            _BFT_R[0] / (1.0 - _BFT_MU[0] * u)
            + _BFT_R[1] / (1.0 - _BFT_MU[1] * u)
            + _BFT_R[2] / (1.0 - _BFT_MU[2] * u)
        )
        out[small] = u * g + 0.5 * eta_arr[small]
    pos = (~small) & (eta_arr > 0.0)
    if np.any(pos):
        q = np.exp(-eta_arr[pos])
        out[pos] = eta_arr[pos] + np.log1p(-q) - np.log(eta_arr[pos])
    neg = (~small) & (~pos)
    if np.any(neg):
        out[neg] = np.log(np.expm1(eta_arr[neg]) / eta_arr[neg])
    return float(out) if scalar else out


def bft_d1(eta: float) -> float:
    if -1.0 < eta < 1.0:
        u = eta * eta
        q = _BFT_D1[-1]
        for coeff in _BFT_D1[-2::-1]:
            q = q * u + coeff
        return eta * q + 0.5
    if eta > 0.0:
        q = np.exp(-eta)
        return 1.0 / (1.0 - q) - 1.0 / eta
    em1 = np.expm1(eta)
    return (em1 + 1.0) / em1 - 1.0 / eta


def bft_d2(eta: float) -> float:
    if -1.0 < eta < 1.0:
        u = eta * eta
        r = _BFT_D2[-1]
        for coeff in _BFT_D2[-2::-1]:
            r = r * u + coeff
        return float(r)
    if eta > 0.0:
        q = np.exp(-eta)
        om = 1.0 - q
        return 1.0 / (eta * eta) - q / (om * om)
    em1 = np.expm1(eta)
    e = em1 + 1.0
    return 1.0 / (eta * eta) - e / (em1 * em1)


def cb_mode(chi: float, nu: float) -> float:
    if nu <= 0.0:
        raise ValueError("nu must be positive")
    xbar = min(max(chi / nu, 1e-12), 1.0 - 1e-12)
    eta = np.log(xbar / (1.0 - xbar))
    for _ in range(40):
        step = (bft_d1(float(eta)) - xbar) / bft_d2(float(eta))
        eta -= step
        if abs(step) < 1e-14 * (1.0 + abs(eta)):
            break
    return float(eta)


def sigmoid(eta: np.ndarray | float) -> np.ndarray | float:
    eta_arr = np.asarray(eta, dtype=np.float64)
    out = np.empty_like(eta_arr)
    pos = eta_arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-eta_arr[pos]))
    exp_eta = np.exp(eta_arr[~pos])
    out[~pos] = exp_eta / (1.0 + exp_eta)
    return float(out) if np.isscalar(eta) else out
