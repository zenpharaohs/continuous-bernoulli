# Reducing ordinary exact-ARS overhead

This follow-up profiles the implementation in commit `ad17c90` and improves its
ordinary batch path while retaining adaptive rescue and endpoint behavior.

## Profile and changes

The proposal helper remained an out-of-line call in Clang's generated assembly.
It passed eta/envelope results through stack slots and prevented useful register
reuse in the draw loop. More significantly, seven-point hulls paid for alias
selection: a uniform column, probability lookup, alternate-index lookup, and then
loads from the chosen segment. A short cumulative scan performs better at this
size on this machine. Chord interpolation also recomputed a slope division for
every proposal although its inputs remain fixed until the hull changes.

The changes are:

- Inline the proposal helper and small target evaluator explicitly, using
  portable compiler-specific inline declarations.
- Cache normalized cumulative segment masses during hull construction. Hulls
  with up to seven points select by a short cumulative scan; larger refined
  hulls continue to use alias selection. The prefix sums are monotone, skip
  zero-mass segments and end at one.
- Cache each chord slope during construction/refinement. A nonfinite cached
  slope uses the existing weighted interpolation. Statistics replacement
  invalidates all geometry as before.

The supporting tangents, segment density and target rejection test are unchanged.
Cumulative selection samples the same normalized segment masses as alias
selection. A cached chord evaluates the same lower line; the floating-point
allowance remains in its squeeze test. No reciprocal approximation, alternate
acceptance variate or approximate fallback is introduced. Refinement still acts
on rejection, with the same trigger and sixteen-point cap. Random sequences for
a fixed seed change because the two segment-selection methods partition the
uniform differently.

Isolated operation timings help locate the cost, but are not additive because
complete draws have different branch and register interactions. For the reported
tiny nu=7 state, proposal generation costs 28.25 ns before and 17.91 ns afterward;
for chi=400, nu=1000 it costs 25.77 ns before and 18.82 ns afterward. The isolated
squeeze falls from 1.46 to 1.23 ns and from 1.50 to 1.34 ns respectively. The RNG,
log acceptance uniform and target evaluation timings remain broadly similar.

## Controlled complete-batch measurements

Clang `-O3`, on this machine, nine trial medians with a rotating variant order.
Each large trial draws one million samples in ten batches, constructing a fresh
hull for each batch. Timed builds have no counters; separately instrumented
builds measure proposals. Constructor, mode solver and core construction are
outside this exact-ARS benchmark. Both versions use the same target parameters
and stream seed. Costs include setup, actual accepted draws and sigmoid output.

| State | Before ns/draw | After ns/draw | Reduction | Before/after proposals per draw |
| --- | ---: | ---: | ---: | ---: |
| Reported tiny nu=7 | 45.15 | 32.70 | 27.6% | 1.0419 / 1.0417 |
| Tiny nu=2 | 44.78 | 33.04 | 26.2% | 1.0391 / 1.0384 |
| chi=0.5, nu=5 | 45.45 | 34.45 | 24.2% | 1.0406 / 1.0405 |
| Reported nu=594 | 42.37 | 33.51 | 20.9% | 1.0433 / 1.0425 |
| chi=4, nu=10 | 46.27 | 37.14 | 19.7% | 1.0419 / 1.0419 |
| chi=400, nu=1000 | 44.23 | 36.35 | 17.8% | 1.0434 / 1.0422 |

The improvement comes from lower proposal cost, not relaxed acceptance or a
substantial change in rejection rate. Large-batch throughput is about 27–31
million exact-ARS draws per second in this benchmark. These are measured C loop
rates, rather than a claim about end-to-end application throughput.

For cold 64-draw batches the time reduction is 13.1–22.0%. Setup cost remains
approximately unchanged: about 0.4–0.55 us in five states and 1.2 us for
chi=0.5, nu=5. That moderate-skew root placement remains a noticeable setup cost
in small batches. Reusing a hull amortizes it in actual streams.

Performance ablations preserve the target and rejection test:

| 100000-draw state | Optimized | Alias at every size | Uncached chords | Compiler's inline choice |
| --- | ---: | ---: | ---: | ---: |
| Reported tiny nu=7 | 32.70 ns | 42.08 ns | 32.82 ns | 35.73 ns |
| Reported nu=594 | 33.51 ns | 38.45 ns | 36.48 ns | 36.12 ns |
| chi=400, nu=1000 | 36.35 ns | 41.70 ns | 36.99 ns | 38.86 ns |

Short-hull selection accounts for the largest gain. Cached chords and explicit
inlining contribute smaller, workload-dependent improvements. Ablation timing
variability is retained in the raw CSV; sub-nanosecond differences should not be
read as universal wins.

## Rescue and correctness checks

The deliberately weak valid two-tangent benchmark continues to refine during
actual sampling. After optimization its proposal rates are 1.054–1.062 per draw,
with about five to six inserted tangents per batch. Rescue batch times improve
by 11–20% across the six states. Larger refined hulls still exercise alias
selection; this optimization does not disable intervention.

The independent 80-digit Decimal audit checks 344 hulls over 86 states, including
chi/nu=1e-150, nu=1e6, one-ULP upper-endpoint gaps, reflections and both initial
and refined hulls. It now checks the actual cached chord coefficients at both
endpoints of every chord. Concavity then bounds the whole chord. It independently
reconstructs both cumulative and alias segment masses, as well as checking the
upper tangents globally. No upper-envelope or lower-squeeze violation occurs;
maximum cumulative-mass discrepancy is 3.80e-16.

The Python suite passes all 32 tests, including a shortened independent audit.
The original 1395-row SPONG replay also completes with a freshly compiled exact library.
All 66 endpoint cases pass at 100000 draws each. A deterministic regression
forces nonfinite chord caches and compares the retained interpolation with the
cached path using identical RNG states. Live-buffer replacement, reflection,
refinement reuse and finite failure limits remain covered.

The user-run MATLAB/MEX quick suite passes all 66 endpoint cases at 100000
samples per case in 4.62 seconds, with maximum endpoint KS=0.00655 against a
0.00949 limit. All 23 ordinary KS/AD/CvM and Legendre cases pass. The reported
median ARS throughput is 23.7 M/s, compared with 20.4 M/s in the preceding run;
these application runs are corroborating evidence, rather than the paired
controlled comparison above. Production C compiles cleanly with `-Wall -Wextra`.

## Reproduce

```sh
python3 tools/exact_ars/profile.py --output /tmp/cb-ordinary-profile
python3 tools/exact_ars/analyze.py --sensitivity-only --output /tmp/cb-ordinary-audit
python3 tests/exact_endpoint_regression.py --n 100000
```

The profile needs the fixed `ad17c90` Git source, standard Python and a compatible
POSIX C compiler. `--sizes 1,32,64,1024,100000` extends the batch sweep;
`--trials` controls repeat count. `--cc` selects another compiler. The sensitivity
audit uses only standard Python and the compiler; endpoint tests also use NumPy.
MATLAB qualification is self-contained and tests the actual MEX interface.

[Raw timing, operation profiles, numerical audit and validation logs](evidence/ordinary-ars-overhead/).
