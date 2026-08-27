# continuous-bernoulli

Continuous Bernoulli conjugate posterior sampler for Thompson sampling bandits.

[![Python CI](https://github.com/zenpharaohs/continuous-bernoulli/actions/workflows/python-ci.yml/badge.svg)](https://github.com/zenpharaohs/continuous-bernoulli/actions/workflows/python-ci.yml)

## Python quick start (no MATLAB required)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install cb-sampler
cb-example       # posterior draw summary and a small CB-vs-Beta bandit demo
cb-validate      # validation suite
```

The Python package compiles the same C sampler backend used by the MATLAB
interface. To work from a clone instead, use
`python -m pip install -e ./python` from the repository root. The release
source archive carries a synchronized copy of the authoritative C core so it
also builds without a repository checkout.

## MATLAB quick start

```matlab
cd continuous-bernoulli
build_cb_stream          % compiles MEX, sets up path

% Create a stream for arm with chi=3, nu=10 (Z=0.30)
s = cb_stream(3.0, 10.0, 'seed', uint64(1));

% Thompson sampling loop
for t = 1:1000
    theta_draw = s.draw(1);     % posterior draw
    % ... select arm, observe reward x ...
    s.update(x);                % update sufficient statistics
end
s.delete();

% Validation
cb_validate
```

On Apple Silicon Macs, build with MATLAB R2026a and Xcode Clang by running
`build_cb_stream` from the repository root.  On Linux, the same command with
gcc installed produces a `.mexa64`.  The generated MEX binary is
platform-specific and intentionally ignored by Git.

## Structure

```
continuous-bernoulli/
  build_cb_stream.m     build script (run first)
  cb_validate.m         validation suite
  src/
    c/
      cb_core.c         C backend (all sampling regimes)
      cb_stream_mex.c   MEX gateway
      include/
        cb_bft.h        B(eta) and derivatives (inline)
        cb_rng.h        RNG primitives (xorshift64*, ziggurats)
    matlab/
      cb_stream.m       MATLAB wrapper for C stream
      cb_mode.m         Newton mode solver
      cb_ars.m          ARS reference sampler
      cb_sample.m       Pure-MATLAB reference (all regimes)
      bft_*.m           B function family
      leg_*.m           Legendre validation utilities
  python/
    src/cb_sampler/     CPython binding over the same cb_core.c
  examples/
    beta_vs_cb_bandit.m Beta approximation vs CB in Thompson sampling
    bandit_quick.m      Small Thompson sampling demonstration
    bandit_likelihood.m Likelihood diagnostics
    bandit_diagnostics.m Bandit run diagnostics
    cb_posterior_demo.m Posterior visualization
    core_stream_benchmark.m Transported-core throughput benchmark
  tests/
    cb_smoke_test.m     Smoke test (< 5 sec)
```

## Sampler regimes

| Regime | Condition | Notes |
|--------|-----------|-------|
| Prior  | chi = nu = 0 | Explicit startup prior |
| Point  | nu > 0 and chi in {0, nu} | Closed-family limit delta_0 or delta_1 |
| ARS    | default (nu < 50) | Exact, always correct |
| Gamma  | nu >= 50, extreme Z | Fast approximation, validated |
| CF     | nu >= 50, sigma < 0.20, symmetric | Fast approximation, validated |

The point-mass regime is an explicit weak closure of the proper posterior
family.  Boundary sufficient statistics are returned deterministically and
are never passed to ARS.  A later observation at the opposite endpoint moves
the aggregate statistics back into the ordinary continuous family.

## Known limitations

- **ARS Z=0.10 nu=20** (and similar asymmetric cases with sigma < 2.5): ziggurat
  equal-area strips use a symmetric tail cutoff, giving a small bias
  (H^2 ~ 1.7e-5) for posteriors where one side decays much more slowly.
  Harmless for bandit regret; documented here as a known limitation.

- **High-Z output precision**: samples with Z > 0.5 are computed as 1 - phi
  where phi is sampled from the reflected distribution.  Near theta = 1,
  double precision has limited resolution (eps/2 ~ 1.1e-16).  This is an
  IEEE 754 constraint, not a sampler error.  The sampler fights as hard as
  double arithmetic allows.

- **CF missing kappa_4**: the Cornish-Fisher path (very narrow posteriors,
  nu >= 50, symmetric) has a small detectable error at N = 1G (H^2 ~ 1e-6)
  from a missing fourth-order term. Deferred.

## Throughput (ThinkPad P16s, single thread)

| Regime | Sampler only |
|--------|-------------|
| ARS    | 9-17 M/s    |
| CF     | 10-16 M/s   |

Full validation pipeline (draw + PIT + Legendre d=128): ~3-4 M/s.

## Validation Policy

`cb_validate.m` uses streaming shifted-Legendre diagnostics for large-N PIT
validation and optional fitted-Beta smoothed-spacing Hellinger certification
for near-null sample-scale estimates:

```matlab
cb_validate_hq_cv = true;
cb_validate_mode = 'thorough';
cb_validate
```

See [Validation](docs/validation.md) for how this mirrors the sibling
`streaming-pit-validate` and `hellinger-qualify` packages.

## Authors

Andrew P. Mullhaupt, Stony Brook University AMS/QF, 2026.
MIT License.
