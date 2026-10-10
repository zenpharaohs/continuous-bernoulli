# Exact correction for the CF phase

The old CF phase reduced to a normal at symmetry, missing the posterior's
quartic term. At `(chi,nu)=(250,500)` its population squared Hellinger error
was 1.197e-7, detected by the user's 100-million-draw MATLAB validation.

CF now uses exact rejection correction with the existing fast normal RNG.
A curvature-bounded normal envelope covers the center, and two posterior
tangents provide exponential tails. A cheap lower quadratic squeeze handles
almost every accepted output; ambiguous proposals use the full posterior.
This corrects every posterior term. The public regime code remains 2.

Unsuitable CF setup selects exact ARS before drawing. ARS adaptive insertion,
core/remainder transport and endpoint paths remain in place. Numerical failure
propagates through both buffered and large draws; no selected CF sample is
replaced with a fallback draw. The input distribution is unchanged.

CF reflection is applied in eta before output conversion. The old CF used the
reflected low-Z mode without reflecting its output; paired regression cases
now check both direct and buffered paths. CF theta buffers invalidate on
statistics replacement. Python enables exact CF; `CB_ENABLE_CF=0` now provides
the intended ARS comparison build.

## Envelope and correction

Let `f(eta)=chi_eff*eta-nu*B(eta)`, computed mode `m`, residual slope `d=f'(m)`,
and radius `r=4.5*sigma`. B'' is even and decreases with absolute eta:
`sinh(t)^3 > t^3*cosh(t)` for t>0, from the positive power-series difference
starting at t^7, proves the sign of the third derivative. Curvature extrema
on the central interval are therefore at the points farthest from and nearest
to zero. Their products with nu give `k_min,k_max`. Taylor's remainder yields

```
d*delta - k_max*delta^2/2 <= f(m+delta)-f(m)
                         <= d*delta - k_min*delta^2/2.
```

The upper quadratic is a normal with shift `d/k_min` and variance `1/k_min`.
Concavity provides tangent envelopes beyond each boundary. Their masses are
`exp(f(edge)-f(m)+padding)/abs(f'(edge))`. The central component uses its
untruncated normal mass; central proposals outside the interval reject the
entire mixture attempt. This avoids a normal-CDF or inverse-CDF evaluation
while preserving component weights. Rejection by `exp(f-relative-upper)`
then gives exactly the posterior density.

The squeeze uses `1-x <= exp(-x)`, with
`x=(k_max-k_min)*delta^2/2+2*padding`, sharing the acceptance uniform with the
full-density test. Zero uniforms are redrawn before logarithms. The residual
mode slope is retained in both quadratics. Curvatures receive a 1e-10 relative
allowance and log densities receive a floating-point allowance. Below |eta|=1,
the first omitted B'' series term is at most 5.1e-12, versus about 7.9e-12 of
curvature allowance at that boundary; direct evaluations have a larger margin.

Setup requires central component mass >= .9, curvature ratio <= 1.5,
absolute shift < .1*r, and padding <= .01. Integrating the lower quadratic
bounds acceptance above .6 (a conservative product bound exceeds .65).
The proposal budget is `65536+64*n`: ample for a certified envelope, while
reporting numerical failure promptly instead of permitting a million
rejections per output. Unsuitable setup routes to exact ARS.

## Benchmarks

Apple Silicon, Clang -O3, nine rotating-order trials; counter-free timings,
separate diagnostic builds. Constructors are outside the clock. Warm batches
reuse setup; cold batches replace statistics each time. All figures are
median nanoseconds per output. The biased old CF is included only to measure
speed retained, not as an exact alternative.

Direct warm 100000-draw batches at symmetry:

| nu | Old CF (approximate) | Corrected CF (exact) | ARS (exact) |
|---:|---:|---:|---:|
| 301 | 15.24 | 18.93 | 41.34 |
| 500 | 15.27 | 18.56 | 41.40 |
| 1000 | 15.22 | 18.22 | 41.47 |
| 5000 | 15.27 | 18.17 | 41.20 |
| 1e+06 | 15.20 | 18.22 | 41.39 |

At nu=500, corrected CF retains about 82% of old CF throughput and is about
2.2 times faster than direct ARS. Statistics setup takes around .07 microseconds
versus .03–.04 for old CF. Cold batches still benefit.

The public stream comparison includes ARS buffers and transported cores.
At `(250,500)`:

| Batch size | Warm CF | Warm ARS | Replaced-stats CF | Replaced-stats ARS |
|---:|---:|---:|---:|---:|
| 1 | 25.00 | 43.00 | 104.20 | 239.40 |
| 64 | 19.33 | 39.30 | 30.04 | 61.40 |
| 100000 | 19.06 | 41.62 | 19.18 | 37.33 |

The public CF advantage is roughly 1.8–2.2 times for warm draws. These are
platform measurements, not guarantees for other compilers or workloads.
The CSV includes all seven states, five variants, sizes, counters and ranges.

## Test evidence

- 34 Python tests passed, including population-density, envelope and live
  CF buffer replacement regressions.
- Thirteen CF states passed independent CDF checks at 100000 latent draws:
  symmetric nu=301/500/1000/5000/1e6, both reflections, modes around the B
  evaluator's |eta|=1 boundary, and endpoint-adjacent modes near +/-100.
- Independent 80-digit checks found no envelope or squeeze violations.
  Reconstructing mixture weights, the squeeze and full acceptance gives
  maximum population squared Hellinger discrepancy below 6e-19, versus
  the old symmetric normal's 1.197e-7.
- Reference CDF grid differences are below 6.8e-8. Independent log-concavity
  bounds put omitted reference-tail probability below exp(-579).
- Rare exponential components were exercised in 2 million CF draws.
  Buffered and large requests propagate deliberate budget failures;
  direct/buffered reflection pairs match exactly in latent eta.
- All 96 endpoint cases passed at 100000 draws, including adaptive rescue,
  repeated statistics replacement, live buffers and remainder errors.
- The ARS numerical audit still passes 440 hulls across 110 states.
- Production C compiles with -Wall -Wextra without warnings; repository
  and packaged core sources match byte for byte.

MATLAB/MEX qualification passed on 10 October 2026 after fresh Xcode Clang
builds. The full quick suite passed all 23 ordinary cases at 100000 draws
per case, all 96 endpoint cases, and all nine exact CF cases. Reflection,
statistics replacement with live buffers, adaptive rescue and bounded-error
checks passed directly through MEX.

The previously failing symmetric CF case `(chi,nu)=(250,500)` also passed
focused thorough qualification at **100 million draws**: KS*sqrt(N)=1.132,
AD=1.001, CvM=0.2076, Legendre SNR=0.0 and effective degree=0. The previous
uncorrected result had AD=5.343 and Legendre SNR=+7.9. No excess Legendre
Hellinger discrepancy was detected; this is a statistical result, not a claim
of zero population distance. Sampler throughput was 48.3 million draws/s
(21.4 million/s including PIT and Legendre processing).

The endpoint and nine-case CF qualifications passed again during that run.
This reran case 19 at 100 million draws, not the full ordinary thorough suite.
The user-provided console results are summarized in
`docs/evidence/exact-cf/matlab-validation.json`.

## Reproduce

```bash
python3 tools/exact_cf/benchmark.py --output /tmp/cb-cf-bench
python3 tools/exact_cf/qualify.py --output /tmp/cb-cf-qualification
```

Qualification needs NumPy; benchmarking uses standard Python, Clang and Git.

```matlab
clear cb_stream_mex cb_stream_eta_mex cb_validate_case_idx;
build_cb_stream;
cb_validate_mode='quick'; cb_validate;
cb_validate_case_idx=19;
cb_validate_mode='thorough'; cb_validate;
clear cb_validate_case_idx;
```
