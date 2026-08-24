"""Python-native validation for the continuous-binomial sampler."""
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import platform
import subprocess
import time

import numpy as np

from . import CbStream
from ._math import bft, cb_mode


@dataclass(frozen=True)
class ValidationCase:
    chi: float
    nu: float
    label: str


@dataclass(frozen=True)
class CaseResult:
    case: ValidationCase
    regime: int
    sigma: float
    ks: float
    ad: float
    cvm: float
    leg_snr: float
    d_eff: int
    passed: bool
    draw_rate: float


CASES = [
    ValidationCase(0.25, 5.0, "ARS   Z=0.05 nu=5"),
    ValidationCase(0.50, 10.0, "ARS   Z=0.05 nu=10"),
    ValidationCase(2.00, 20.0, "ARS   Z=0.10 nu=20"),
    ValidationCase(4.00, 10.0, "ARS   Z=0.40 nu=10"),
    ValidationCase(5.00, 10.0, "ARS   Z=0.50 nu=10"),
    ValidationCase(6.00, 20.0, "ARS   Z=0.30 nu=20"),
    ValidationCase(50.0, 100.0, "ARS   Z=0.50 nu=100"),
    ValidationCase(250.0, 500.0, "CF    Z=0.50 nu=500 routes ARS"),
]
REGIME_NAMES = {0: "prior", 1: "Gamma", 2: "CF", 3: "ARS", 4: "point"}
BLOCK_LO = np.array([1, 9, 17, 33, 65])
BLOCK_HI = np.array([8, 16, 32, 64, 128])


def _eta_grid(chi: float, nu: float, n_grid: int) -> tuple[np.ndarray, np.ndarray]:
    eta_star = cb_mode(chi, nu)
    f_star = chi * eta_star - nu * bft(eta_star)
    step = 0.5
    threshold = np.log(1e-12)
    e_left = eta_star
    while chi * e_left - nu * bft(e_left) - f_star > threshold:
        e_left -= step
    e_right = eta_star
    while chi * e_right - nu * bft(e_right) - f_star > threshold:
        e_right += step
    e_left -= 2.0
    e_right += 2.0
    eta = np.linspace(e_left, e_right, n_grid)
    logp = chi * eta - nu * bft(eta)
    p = np.exp(logp - np.max(logp))
    mass = 0.5 * (p[:-1] + p[1:]) * np.diff(eta)
    cdf = np.concatenate(([0.0], np.cumsum(mass)))
    cdf /= cdf[-1]
    cdf[-1] = 1.0
    return eta, cdf


def _pit(theta: np.ndarray, eta_grid: np.ndarray, cdf_grid: np.ndarray) -> np.ndarray:
    theta = np.clip(theta, np.finfo(float).tiny, 1.0 - np.finfo(float).eps)
    eta = np.log(theta / (1.0 - theta))
    return np.clip(np.interp(eta, eta_grid, cdf_grid), 0.0, 1.0)


def _legendre_signal(u: np.ndarray, d_max: int = 128, snr_stop: float = 2.0) -> tuple[float, int]:
    n = u.size
    x = 2.0 * u - 1.0
    sums = np.zeros(d_max)
    pkm2 = np.ones_like(x)
    pkm1 = x
    sums[0] = np.sqrt(3.0) * np.sum(pkm1)
    for k in range(2, d_max + 1):
        pk = ((2 * k - 1) * x * pkm1 - (k - 1) * pkm2) / k
        sums[k - 1] = np.sqrt(2 * k + 1) * np.sum(pk)
        pkm2 = pkm1
        pkm1 = pk
    coeff = sums / n
    block_snr = []
    block_chi2 = []
    for lo, hi in zip(BLOCK_LO, BLOCK_HI):
        if hi > d_max:
            break
        size = hi - lo + 1
        energy = float(np.sum(coeff[lo - 1 : hi] ** 2))
        block_snr.append((n * energy - size) / np.sqrt(2.0 * size))
        block_chi2.append(energy - size / n)
    significant = [idx for idx, value in enumerate(block_snr) if abs(value) > snr_stop]
    if not significant:
        return 0.0, 0
    last = significant[-1]
    d_eff = int(BLOCK_HI[last])
    chi2_leg = max(float(np.sum(block_chi2[: last + 1])), 0.0)
    sigma_h2 = np.sqrt(max(d_eff, 1)) / (2.0 * np.sqrt(2.0) * n)
    leg_snr = chi2_leg / (sigma_h2 * 8.0)
    return float(leg_snr), d_eff


def validate_case(case: ValidationCase, n: int, seed: int, grid_size: int) -> CaseResult:
    eta_grid, cdf_grid = _eta_grid(case.chi, case.nu, grid_size)
    start = time.perf_counter()
    with CbStream(case.chi, case.nu, seed=seed, buf_size=4096) as stream:
        state = stream.peek()
        theta = stream.draw(n)
    elapsed = max(time.perf_counter() - start, 1e-12)
    u = np.sort(_pit(theta, eta_grid, cdf_grid))
    idx = np.arange(1, n + 1, dtype=np.float64)
    ks = float(np.max(np.abs(idx / n - u)))
    u_ad = np.clip(u, 1e-300, 1.0 - 1e-15)
    ad = float(-n - np.mean((2.0 * idx - 1.0) * (np.log(u_ad) + np.log(1.0 - u_ad[::-1]))))
    cvm = float(1.0 / (12.0 * n) + np.sum((u - (2.0 * idx - 1.0) / (2.0 * n)) ** 2))
    leg_snr, d_eff = _legendre_signal(u)
    ks_fail = ks > 1.63 / np.sqrt(n)
    ad_fail = ad > 3.857
    cvm_fail = cvm > 0.743
    passed = abs(leg_snr) <= 3.0
    if not passed and not (ks_fail or ad_fail or cvm_fail):
        passed = False
    return CaseResult(
        case=case,
        regime=state.regime,
        sigma=state.sigma,
        ks=ks,
        ad=ad,
        cvm=cvm,
        leg_snr=leg_snr,
        d_eff=d_eff,
        passed=passed,
        draw_rate=n / elapsed,
    )


def run_validation(mode: str = "quick", n: int | None = None, grid_size: int = 12000) -> list[CaseResult]:
    if n is None:
        n = {"smoke": 20_000, "quick": 100_000, "standard": 1_000_000}[mode]
    print()
    print("=" * 72)
    print(f"  CB Sampler: Python Validation [{mode.upper()}]")
    print("=" * 72)
    print(f"N per case: {n:,}")
    print("Tests: KS/AD/CvM warning lights; adaptive Legendre chi^2 arbiter")
    print()
    print(f"{'Case':32s} {'Reg':6s} {'sigma':>8s} {'KS':>8s} {'AD':>8s} {'CvM':>8s} {'LegSNR':>8s} {'deff':>5s}")
    print("-" * 94)
    results = []
    for idx, case in enumerate(CASES, start=1):
        result = validate_case(case, n=n, seed=idx * 31 + 7, grid_size=grid_size)
        results.append(result)
        print(
            f"{case.label:32s} {REGIME_NAMES.get(result.regime, '?'):6s} "
            f"{result.sigma:8.3g} {result.ks:8.4g} {result.ad:8.3g} "
            f"{result.cvm:8.3g} {result.leg_snr:+8.2f} {result.d_eff:5d}"
        )
    print()
    if all(r.passed for r in results):
        print("RESULT: PYTHON VALIDATION PASSED")
        print("All adaptive Legendre signals are quiet: |LegSNR| <= 3.")
    else:
        print("RESULT: PYTHON VALIDATION NEEDS REVIEW")
        print("At least one case has |LegSNR| > 3.")
    rates = np.array([r.draw_rate for r in results])
    print(f"Sampler throughput median: {np.median(rates) / 1e6:.1f} M/s")
    return results


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def write_json_report(path: str, results: list[CaseResult], mode: str, n: int | None, grid_size: int) -> None:
    payload = {
        "schema": "cb_sampler.python_validation.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "n_requested": n,
        "grid_size": grid_size,
        "platform": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "git_commit": _git_commit(),
        "passed": all(result.passed for result in results),
        "results": [asdict(result) for result in results],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "quick", "standard"], default="quick")
    parser.add_argument("--n", type=int, default=None, help="override samples per case")
    parser.add_argument("--grid-size", type=int, default=12000)
    parser.add_argument("--json-out", default=None, help="write a machine-readable validation artifact")
    args = parser.parse_args(argv)
    results = run_validation(mode=args.mode, n=args.n, grid_size=args.grid_size)
    if args.json_out is not None:
        write_json_report(args.json_out, results, args.mode, args.n, args.grid_size)
        print(f"Wrote validation artifact: {args.json_out}")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
