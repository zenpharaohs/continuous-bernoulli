# Faster exact ARS hull placement

The initial point solver stopped Newton after three steps, then restarted a
bounded bracketed solve from a Gaussian distance guess. In moderately skewed
states this discarded an almost-converged point. For `(chi,nu)=(.5,5)`, each
right-hand drop-2 and drop-5 point cost 11 density evaluations.

The solver now allows six bounded Newton steps. When the reflected mode is
below -8, its asymmetric tail-geometry root supplies a seed, which the full
posterior density checks and refines. In the tail coordinate `y=log(eta/mode)`,
the seed solves `y-expm1(y)=-drop/nu`; this accounts for unequal left and right
distances from the mode. The existing direct tail placement below
-32, bounded fallback, tangent construction, acceptance test, adaptive insertion,
rejection budget, and stream invalidation policy remain in place. Placement
need only approximate the requested density drop: all tangents still use the
full posterior. There is no change to the target distribution or input statistics.

## Controlled setup and batch measurements

Apple Silicon, Clang `-O3`, nine trials with rotating variant order; median
counter-free times, with counters in separate builds. Baseline `a5f277d`.
Each batch starts with a fresh exact ARS hull. Constructor, mode/scale solve,
and Gamma/core construction are outside the measured interval. These figures
measure the changed hull placement path, rather than total public stream setup.

| chi, nu | Seven-point setup before → after (µs) | Reduction | Fresh 64-draw batch reduction | Placement density evaluations before → after |
|---|---:|---:|---:|---:|
| .5, 5 | 1.250 → .715 | 43% | 14% | 34 → 10 |
| .5, 10 | 1.092 → .605 | 45% | 14% | 27 → 6 |
| 2, 20 | .844 → .784 | 7% | 3% | 16 → 9 |
| .16, 5 | 1.600 → .629 | 61% | 23% | 38 → 6 |

The six-step-only ablation recovers part of the benefit: setup at `(.5,5)`
is .840 µs, versus .715 µs with the improved seed. At `(.16,5)` it is 1.023 µs.
The full change also reduces fresh single-draw time by 44% at `(.5,5)` and
54% at `(.16,5)`, where a three-point hull is used.

Proposal counts show no material loss: for fresh 64-draw batches at `(.5,5)`
they are 1.041241 → 1.041289 per output; at `(.16,5)` they are
1.041898 → 1.041998. Across the four skewed states and tested sizes, the
largest increase is .000536 proposals per output (about .05%). These are
changes in deterministic proposal trajectories, not distribution errors.

Central and extreme-endpoint placement has unchanged evaluation counts.
Their timings show small variation without a consistent improvement. Large
100000-draw batches amortize setup and show mixed changes in elapsed time
per output (about -2.5% to +3.3% across all eight states); this is a setup optimization.
Raw minimum/maximum times and all variants are in `batches.csv`.

## Correctness and numerical sensitivity

- 32 Python tests passed after rebuilding the extension.
- All 96 independent-CDF distribution cases passed at 100000 draws each.
  Added moderate-skew families `(.5,5)`, `(.5,10)`, `(2,20)` and modes near
  -8 and -32; each includes representable reflections, exact ARS,
  core/remainder where available, and deliberately weak-hull adaptive rescue.
- Live buffers survive covered statistics updates; 3000 replacements,
  hull reuse/invalidation, and bounded numerical-failure checks passed.
- Independent 80-digit analysis examined 440 hulls across 110 states,
  including both sides of mode -8 and -32 at six values of nu. No upper
  envelope deficit or lower squeeze excess was found. Maximum normalized
  CDF mass error was 3.8e-16. Parameters include gaps down to 5e-151 and
  nu up to 1e6, short/full initial hulls and adaptively refined hulls.
- The deterministic 1395-draw SPONG replay completed with the updated C library.
- Production C compiled with `-Wall -Wextra` without warnings; packaged and
  repository C sources match byte for byte.

MATLAB/MEX validation: awaiting the user's run of the expanded 96-case suite.

## Reproduce

```bash
python3 tools/exact_ars/profile_setup.py --output /tmp/cb-hull-profile
python3 tools/exact_ars/analyze.py --sensitivity-only --output /tmp/cb-hull-audit
python3 tests/exact_endpoint_regression.py --n 100000
```

For the actual MATLAB interface:

```matlab
clear cb_stream_mex cb_stream_eta_mex;
build_cb_stream;
cb_validate_mode = 'quick';
cb_validate;
```
