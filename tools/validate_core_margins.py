#!/usr/bin/env python3
"""Validate transported-core margin constants on a modest deterministic grid.

This is an offline audit tool for the constants used by cb_core.c.  It checks:

  1. The deflated standardized core S(u) is below every scanned in-range
     standardized target shape G_q(u).
  2. The anchor W safety bound covers scanned W_q values.

It intentionally uses only the Python standard library so it can run anywhere
MATLAB/MEX development does.  The scan is not exhaustive proof; it is the
house-style numerical gate described in TRANSPORTED_CORE_SPEC.md.
"""

from __future__ import annotations

import math


CORE_K = 5
U_MAX = 8.0
BUILD_N = 257
SHAPE_MARGIN = 1e-3
W_SAFETY = 1e-3
MIN_NU = 150.0
RANGE_ROOM_FRAC = 0.5
RANGE_MAX = 256.0


def bft(eta: float) -> float:
    if -1.0 < eta < 1.0:
        u = eta * eta
        return 0.5 * eta + u / 24.0 - u * u / 2880.0 + u * u * u / 181440.0
    if eta > 0.0:
        q = math.exp(-eta)
        return eta + math.log1p(-q) - math.log(eta)
    return math.log(math.expm1(eta) / eta)


def bft_d1(eta: float) -> float:
    if -1.0 < eta < 1.0:
        u = eta * eta
        return 0.5 + eta * (1.0 / 12.0 - u / 720.0 + u * u / 30240.0)
    if eta > 0.0:
        q = math.exp(-eta)
        return 1.0 / (1.0 - q) - 1.0 / eta
    em1 = math.expm1(eta)
    return (em1 + 1.0) / em1 - 1.0 / eta


def bft_d2(eta: float) -> float:
    if -1.0 < eta < 1.0:
        u = eta * eta
        return 1.0 / 12.0 - u / 240.0 + u * u / 6048.0
    if eta > 0.0:
        q = math.exp(-eta)
        om = 1.0 - q
        return 1.0 / (eta * eta) - q / (om * om)
    em1 = math.expm1(eta)
    return 1.0 / (eta * eta) - (em1 + 1.0) / (em1 * em1)


def mode(chi: float, nu: float) -> float:
    x = min(max(chi / nu, 1e-12), 1.0 - 1e-12)
    eta = math.log(x / (1.0 - x))
    for _ in range(60):
        step = (bft_d1(eta) - x) / bft_d2(eta)
        eta -= step
        if abs(step) < 1e-14 * (1.0 + abs(eta)):
            break
    return eta


def logf(chi: float, nu: float, eta: float) -> float:
    return chi * eta - nu * bft(eta)


def ghat(chi: float, nu: float, eta_star: float, sig: float, u: float) -> float:
    return logf(chi, nu, eta_star + sig * u) - logf(chi, nu, eta_star)


def logsumexp(vals: list[float]) -> float:
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


def simpson_log(vals: list[float], width: float) -> float:
    logs = []
    n = len(vals)
    for i, v in enumerate(vals):
        w = 1.0 if i == 0 or i == n - 1 else (4.0 if i & 1 else 2.0)
        logs.append(v + math.log(w))
    return math.log(width / 3.0) + logsumexp(logs)


def build(theta: float, nu0: float):
    chi0 = theta * nu0
    room = nu0 - 2.0 * chi0
    r = min(RANGE_ROOM_FRAC * room, RANGE_MAX)
    nu1 = nu0 + r
    zlo = chi0 / nu1
    zhi = (chi0 + r) / nu1
    anchors = []
    eta0 = mode(chi0, nu0)
    sig0 = 1.0 / math.sqrt(nu0 * bft_d2(eta0))
    anchors.append((chi0, nu0, eta0, sig0))
    for a in range(CORE_K - 1):
        t = a / (CORE_K - 2)
        z = zlo + t * (zhi - zlo)
        chi = z * nu1
        eta = mode(chi, nu1)
        sig = 1.0 / math.sqrt(nu1 * bft_d2(eta))
        anchors.append((chi, nu1, eta, sig))
    return chi0, nu0, r, anchors


def main() -> int:
    us = [-U_MAX + 2.0 * U_MAX * i / (BUILD_N - 1) for i in range(BUILD_N)]
    du = us[1] - us[0]
    max_shape_violation = -math.inf
    max_w_excess = -math.inf
    worst_shape = None
    worst_w = None

    for theta in (0.02, 0.05, 0.10, 0.20, 0.40):
        for nu0 in (MIN_NU, 300.0, 1000.0, 10000.0):
            chi0, nu_origin, r, anchors = build(theta, nu0)
            if r <= 0.0:
                continue

            rows = []
            logWs = []
            for chi, nu, eta, sig in anchors:
                vals = [ghat(chi, nu, eta, sig, u) for u in us]
                rows.append(vals)
                logWs.append(simpson_log(vals, du))
            core = [min(row[i] for row in rows) - SHAPE_MARGIN for i in range(BUILD_N)]
            logW_bound = max(logWs) + W_SAFETY

            candidates = []
            for dnu_frac in (0.0, 0.25, 0.5, 0.75, 1.0):
                dnu = r * dnu_frac
                for edge in (0.0, 0.5, 1.0):
                    dchi = dnu * edge
                    candidates.append((chi0 + dchi, nu_origin + dnu))
            for a in range(CORE_K):
                t = a / (CORE_K - 1)
                candidates.append((chi0 + r * t, nu_origin + r))

            for chi, nu in candidates:
                eta = mode(chi, nu)
                sig = 1.0 / math.sqrt(nu * bft_d2(eta))
                vals = [ghat(chi, nu, eta, sig, u) for u in us]
                for u, s_val, g_val in zip(us, core, vals):
                    viol = s_val - g_val
                    if viol > max_shape_violation:
                        max_shape_violation = viol
                        worst_shape = (theta, nu0, r, chi / nu, u, viol)
                lw = simpson_log(vals, du)
                excess = lw - logW_bound
                if excess > max_w_excess:
                    max_w_excess = excess
                    worst_w = (theta, nu0, r, chi / nu, excess)

    print(f"max shape violation: {max_shape_violation:.3e} at {worst_shape}")
    print(f"max W excess:         {max_w_excess:.3e} at {worst_w}")
    if max_shape_violation > 0.0 or max_w_excess > 0.0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
