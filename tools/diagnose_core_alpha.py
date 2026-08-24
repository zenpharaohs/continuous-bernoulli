#!/usr/bin/env python3
"""Decompose transported-core alpha losses on the validation grid.

This is a development diagnostic, not a proof.  It mirrors
validate_core_margins.py and reports how much alpha is lost to the common
shape itself, the deflation margin, and the W safety term.
"""

from __future__ import annotations

import math

from validate_core_margins import (
    BUILD_N,
    CORE_K,
    MIN_NU,
    RANGE_MAX,
    RANGE_ROOM_FRAC,
    SHAPE_MARGIN,
    U_MAX,
    W_SAFETY,
    build,
    ghat,
    logsumexp,
    mode,
    bft_d2,
    simpson_log,
)


def scan_candidates(chi0: float, nu0: float, r: float):
    for dnu_frac in (0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0):
        dnu = r * dnu_frac
        for edge in (0.0, 0.25, 0.5, 0.75, 1.0):
            yield chi0 + dnu * edge, nu0 + dnu
    for a in range(2 * CORE_K + 1):
        t = a / (2 * CORE_K)
        yield chi0 + r * t, nu0 + r


def alpha_row(theta: float, nu0: float):
    chi0, _, r, anchors = build(theta, nu0)
    if r <= 0.0:
        return None

    us = [-U_MAX + 2.0 * U_MAX * i / (BUILD_N - 1) for i in range(BUILD_N)]
    du = us[1] - us[0]

    rows = []
    log_ws = []
    for chi, nu, eta, sig in anchors:
        vals = [ghat(chi, nu, eta, sig, u) for u in us]
        rows.append(vals)
        log_ws.append(simpson_log(vals, du))

    raw_core = [min(row[i] for row in rows) for i in range(BUILD_N)]
    deflated_core = [v - SHAPE_MARGIN for v in raw_core]
    log_zs_raw = simpson_log(raw_core, du)
    log_zs_deflated = simpson_log(deflated_core, du)
    log_w_anchor = max(log_ws)

    log_w_scan = -math.inf
    worst_z = None
    max_shape_gap = -math.inf
    for chi, nu in scan_candidates(chi0, nu0, r):
        eta = mode(chi, nu)
        sig = 1.0 / math.sqrt(nu * bft_d2(eta))
        vals = [ghat(chi, nu, eta, sig, u) for u in us]
        lw = simpson_log(vals, du)
        if lw > log_w_scan:
            log_w_scan = lw
            worst_z = chi / nu
        for s_val, g_val in zip(deflated_core, vals):
            max_shape_gap = max(max_shape_gap, s_val - g_val)

    return {
        "theta": theta,
        "nu": nu0,
        "R": r,
        "raw": math.exp(log_zs_raw - log_w_anchor),
        "margin": math.exp(log_zs_deflated - log_w_anchor),
        "scan": math.exp(log_zs_deflated - log_w_scan),
        "hat": math.exp(log_zs_deflated - log_w_anchor - W_SAFETY),
        "need_w": log_w_scan - log_w_anchor,
        "worst_z": worst_z,
        "shape_gap": max_shape_gap,
    }


def main() -> int:
    print(
        "theta    nu      R       raw     -shape   scanW   alpha_hat  "
        "needW      shapeGap   worst_z"
    )
    for theta in (0.01, 0.02, 0.05, 0.10, 0.20, 0.40):
        for nu0 in (MIN_NU, 300.0, 1000.0, 10000.0):
            row = alpha_row(theta, nu0)
            if row is None:
                continue
            print(
                f"{row['theta']:5.2f} {row['nu']:7.0f} {row['R']:7.1f} "
                f"{row['raw']:7.4f} {row['margin']:7.4f} "
                f"{row['scan']:7.4f} {row['hat']:9.4f} "
                f"{row['need_w']:9.2e} {row['shape_gap']:10.2e} "
                f"{row['worst_z']:7.4f}"
            )
        print()

    print(f"shape margin log tax: {SHAPE_MARGIN:g}")
    print(f"W safety log tax:     {W_SAFETY:g}  factor={math.exp(-W_SAFETY):.6f}")
    print(f"range constants: room_frac={RANGE_ROOM_FRAC:g}, range_max={RANGE_MAX:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
