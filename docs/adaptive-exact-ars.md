# Adaptive exact ARS

The exact posterior has log density `f(eta) = chi*eta - nu*B(eta)`.
Concavity gives an upper hull from supporting tangents and a lower squeeze from
chords. Refining the hull changes the proposal; it leaves the target density
and rejection test unchanged. This change addresses inefficient envelopes before
the existing finite proposal limit is reached.

## Construction and intervention

Initial points follow approximate log-density drops from the mode rather than
fixed multiples of sigma. This accommodates asymmetric endpoint geometry without
placing an unnecessarily distant point across eta=0. For an adaptive fill target
of at most four samples, use the mode and drop-2 points on either side. Otherwise,
use seven points: the mode and drops 0.5, 2 and 5 on either side. The tail placement
uses a cheap solve in `log(-eta/mode)`; elsewhere safeguarded Newton steps locate
the drops. Placement tolerance affects efficiency only. Target evaluations and
acceptance continue to use the full continuous-Bernoulli log density.

A rejection supplies a candidate tangent at the rejected eta. Plain ARS inserts
it after four consecutive rejections, or after at least eight window rejections
with a rejection fraction above 1/8 when at least 256 draws remain or have accrued
since construction. The window resets on a successful insertion. The hull stops
at 16 points. This policy avoids rebuilding already efficient hulls after every
occasional rejection. Duplicate points, deep-tail points more than 50 log-density
units below the mode, and candidates with invalid geometry are skipped. A failed
insertion leaves the previous hull intact.

Each insertion recomputes intersections, segment integrals and an alias table.
The alias table keeps segment selection constant-time as the hull grows. Refined
hulls survive both refills and large batches; statistics changes invalidate them.
For remainder sampling, insertion tightens an upper bound on the **full** posterior
only when that bound has slack. The selected residual `f - alpha*core`, subtraction
constant and mixture weight remain fixed. A small residual mass by itself cannot
be repaired by inserting full-posterior tangents.

Every proposal is accepted against the current target or its lower squeeze. The
proposal chosen for the next draw may depend on previous rejections without
changing that conditional accepted distribution. This is the exact ARS argument;
there is no approximate fallback or clipping of input statistics. An invalid
initial hull or exhausted finite proposal budget still reports an error.

## Numerical changes and sensitivity

Tangent intercepts use the cancellation-free expression
`nu*(eta*B'(eta)-B(eta))`, with a tail identity. Near zero, evaluation separates
`B(eta)=eta/2+B_centered(eta)` so `chi-nu/2` is formed before multiplying by eta.
The slope and intercept use corresponding centered formulas. The numerical audit
found that uncentered near-zero calculations at large nu could put a tangent
slightly below the target. The centered evaluations and a common upward allowance
`64*DBL_EPSILON*(1 + nu + max(abs(intercepts)))` remove the defects in the audit.
The same allowance is subtracted from chord squeezes. Common padding preserves
intersections and normalized proposal shape, and is included in the rejection
ratio. It is a floating-point allowance, not a target-density modification.

Finite-segment exponential inversions are anchored on the higher-density end,
using `expm1(-abs(slope)*width)`; they avoid positive exponential overflow. Tail
inversion and log acceptance redraw a zero RNG uniform so they cannot produce an
infinite eta from an otherwise valid hull. A deterministic RNG regression forces
that zero value and checks the redraw.

The independent audit uses Python Decimal at 80 digits and analytic partition
identities, rather than the sampler's approximations or mode solver. It checks
86 parameter states and four hull configurations per state (short/full,
initial/refined), for 344 hulls. The sweep covers nu from 0.5 to 1e6,
chi/nu down to 1e-150, central states, and upper endpoints one representable step
inside nu. For each tangent, concavity locates the global maximum of target minus
line; additional checks cover intersections, adjacent representable values and
chord bounds. It also reconstructs the alias probabilities.

Results:

- No upper-envelope deficit or lower-squeeze excess was observed.
- Maximum alias probability discrepancy: 2.50e-16.
- Maximum log-density evaluation error: 7.04e-8, at nu=1e6 and chi=1e-144;
  the allowance there is 4.91e-6. Maximum tangent intercept error: 5.36e-8.
- Maximum relative slope error: 7.30e-13, without an envelope violation.
- The approximate initial drop differs from its request by at most 0.047 log
  units in the sweep. This affects only proposal placement.

This is a numerical audit of the stated range, rather than an interval-arithmetic
proof for every representable input. Small/tail terms omitted in the independent
reference are below 1e-80 over the audit range.

One-ULP input perturbations expose intrinsic endpoint conditioning. At
`nu=1e6, chi=nextafter(nu,0)`, decreasing chi by one more representable step doubles
the endpoint gap, halves the latent scale and shifts the mode by 500 original
standard deviations. This is a real change in the requested distribution.
At tiny lower endpoints the relative gap perturbation is of order machine epsilon.
Keeping the representable chi and nu intact is essential; clipping would conceal
this sensitivity and request a different distribution.

## Setup cost versus saved proposals

Measurements use Clang `-O3` on this machine, against commit `79bad31`, with three
trial medians. Timing builds have no counters; proposal counts come from separate
instrumented runs with the same seed. These are exact-ARS envelope costs, excluding
stream construction, mode solving and core setup. A new hull is built for every
benchmark batch. Actual streams additionally amortize construction by reusing it.

A successful insertion including hull copy and all refreshed tables costs
0.31–0.36 microseconds across the seven benchmark states. A rejected proposal in
the deliberately weak envelope costs about 30–35 nanoseconds, giving an approximate
break-even of ten saved proposals per insertion.

For 100000 draws from a deliberately weak **valid** two-tangent envelope:

| State | Fixed proposals/draw | Adaptive proposals/draw | Insertions | Fixed time | Adaptive time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reported tiny nu=7 | 123.40 | 1.068 | 5.0 | 408.6 ms | 5.35 ms |
| Reported nu=594 | 580.52 | 1.067 | 5.3 | 1771.7 ms | 4.69 ms |
| Central chi=400, nu=1000 | 581.76 | 1.059 | 6.7 | 1914.1 ms | 5.01 ms |

These weak hulls are injected by the diagnostic harness to exercise intervention;
they are not the normal new initialization. Five to seven insertions save millions
of proposals, decisively repaying setup.

For ordinary initial hulls, the tradeoff is smaller:

| State / batch | Old setup | New setup | Old total | New total | Old/new proposals per draw |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tiny nu=7 / 1 draw | 0.276 us | 0.188 us | 0.374 us | 0.240 us | 1.042 / 1.203 |
| Reported nu=594 / 1 draw | 0.268 us | 0.176 us | 0.371 us | 0.256 us | 1.038 / 1.213 |
| chi=0.5, nu=5 / 1 draw | 0.230 us | 0.611 us | 0.337 us | 0.699 us | 1.044 / 1.190 |
| Tiny nu=7 / 100000 draws | 0.274 us | 0.491 us | 4.57 ms | 4.93 ms | 1.038 / 1.042 |
| Central chi=400, nu=1000 / 100000 draws | 0.113 us | 0.432 us | 3.99 ms | 4.56 ms | 1.102 / 1.044 |

The small hull saves setup in endpoint replacement workloads despite accepting
less efficiently. Root placement costs more in the moderate skew case. Central
large-batch initialization saves about 5850 proposals per 100000 draws, but the
current complete implementation still takes more wall time than the baseline.
This change improves rescue behavior and endpoint construction, with measurable
ordinary-path overhead; it is not a universal throughput improvement. Adaptive
refinement is deliberately selective to limit that cost. Raw results include
batch sizes 1, 32, 64, 1024 and 100000 and the no-adaptation ablation.

## Regression evidence and reproduction

- Python: 32 tests pass, including a short independent Decimal audit.
- Python endpoint qualification: all 66 cases pass with 100000 draws per case.
- User-run MATLAB quick validation: the actual production and diagnostic MEX
  builds pass all 66 endpoint cases in 4.05 seconds and all 23 ordinary cases.
  Maximum endpoint KS is 0.00648 versus the 0.00949 limit.
- Both suites cover reflections, live buffers, repeated statistics replacement,
  weak-envelope rescue, gaps down to 2e-150 and nu=1e6. Independent CDF quadrature
  uses two resolutions and log-concavity bounds on omitted tails.
- The original 1395-row SPONG replay completes with a freshly compiled exact
  library. Production C compiles cleanly with `-Wall -Wextra`.

Run the complete benchmark and sensitivity audit (standard Python library and
Clang, macOS/Linux):

```sh
python3 tools/exact_ars/analyze.py --output /tmp/cb-ars-analysis
```

Use `--sensitivity-only` to avoid requiring historical Git sources;
`--quick` selects eight sensitivity states for the regular regression test.
Use `--cc` to select another compatible C compiler. Run endpoint qualification
with NumPy using `python3 tests/exact_endpoint_regression.py --n 100000`.
MATLAB uses `build_cb_stream; cb_validate_mode='quick'; cb_validate;` independently
of Python.

[Raw benchmark, numerical sweep, metadata and validation logs](evidence/adaptive-exact-ars/)
record the evidence for this change.
