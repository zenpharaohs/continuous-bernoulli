"""Small Python examples for the continuous-binomial sampler."""
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse

import numpy as np

from . import CbStream, sample


def posterior_demo(chi: float = 6.0, nu: float = 20.0, n: int = 10_000, seed: int = 1) -> dict[str, float]:
    draws = sample(chi, nu, size=n, seed=seed)
    summary = {
        "chi": float(chi),
        "nu": float(nu),
        "mean_draw": float(np.mean(draws)),
        "std_draw": float(np.std(draws)),
        "q05": float(np.quantile(draws, 0.05)),
        "q50": float(np.quantile(draws, 0.50)),
        "q95": float(np.quantile(draws, 0.95)),
    }
    print("Continuous-binomial posterior draw summary")
    for key, value in summary.items():
        print(f"  {key:>9s}: {value:.6g}")
    return summary


def bandit_quick(k: int = 8, t: int = 2000, trials: int = 5, seed: int = 137) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    theta_true = np.array([0.05, 0.10, 0.18, 0.25, 0.33, 0.42, 0.50, 0.62], dtype=np.float64)[:k]
    theta_opt = float(np.min(theta_true))
    regret_cb = np.zeros((trials, t))
    regret_beta = np.zeros((trials, t))
    for trial in range(trials):
        rewards = rng.random((k, t)) < theta_true[:, None]
        chi_cb = np.zeros(k)
        nu_cb = np.zeros(k)
        chi_beta = np.zeros(k)
        nu_beta = np.zeros(k)
        for step in range(t):
            draws_cb = np.empty(k)
            for arm in range(k):
                draws_cb[arm] = sample(
                    chi_cb[arm],
                    nu_cb[arm],
                    seed=trial * 100_000 + step * k + arm + 1,
                )
            arm_cb = int(np.argmin(draws_cb))
            r_cb = float(rewards[arm_cb, step])
            chi_cb[arm_cb] += r_cb
            nu_cb[arm_cb] += 1.0
            regret_cb[trial, step] = theta_true[arm_cb] - theta_opt

            draws_beta = rng.beta(chi_beta + 0.5, nu_beta - chi_beta + 0.5)
            arm_beta = int(np.argmin(draws_beta))
            r_beta = float(rewards[arm_beta, step])
            chi_beta[arm_beta] += r_beta
            nu_beta[arm_beta] += 1.0
            regret_beta[trial, step] = theta_true[arm_beta] - theta_opt
    summary = {
        "cb_regret": float(np.mean(np.sum(regret_cb, axis=1))),
        "beta_regret": float(np.mean(np.sum(regret_beta, axis=1))),
    }
    summary["beta_over_cb"] = summary["beta_regret"] / max(summary["cb_regret"], 1e-12)
    print(f"Quick bandit example: K={k}, T={t}, trials={trials}")
    print(f"  CB exact posterior regret: {summary['cb_regret']:.3g}")
    print(f"  Beta fallback regret:      {summary['beta_regret']:.3g}")
    print(f"  Beta / CB:                 {summary['beta_over_cb']:.3g}")
    return summary


def streaming_demo(seed: int = 1) -> None:
    with CbStream(3.0, 10.0, seed=seed, stream_idx=0) as stream:
        print("Initial:", stream.peek())
        print("Draws:", stream.draw(5))
        stream.update(0.75)
        stream.update_batch(np.array([0.25, 0.5]))
        print("Updated:", stream.peek())
        print("Diagnostics:", stream.diagnostics())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", choices=["posterior", "bandit", "stream"], nargs="?", default="stream")
    args = parser.parse_args(argv)
    if args.example == "posterior":
        posterior_demo()
    elif args.example == "bandit":
        bandit_quick()
    else:
        streaming_demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
