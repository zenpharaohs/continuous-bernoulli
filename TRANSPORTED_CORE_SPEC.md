# Transported Common-Core Sampler — Implementation Spec

Status: implemented in `src/c/cb_core.c` (2026-07-05). Supersedes the
corner-min common hull in the older tree (`build_hull_common` /
`current_common_alpha`). The correctness fixes in that tree (accept-ratio
clamps, buffer/zig invalidation on update, smoke-test additions) are kept; the
corner-min core construction and the per-update alpha quadrature are replaced.

Implementation notes from the performance pass:

- The stream buffer carries standardized `u` values only while the buffer kind
  is `CB_BUF_CORE_U`; Gamma/CF/prior buffers continue to carry theta samples.
  This tag prevents stale theta buffers from being transported as core draws
  after regime flapping.
- `alpha_hat` is build-time constant. The validation grid showed the previous
  `CB_CORE_W_SAFETY=0.05` was almost entirely unused, so the implemented
  constant is `1e-3`, validated by `tools/validate_core_margins.py`.
- The core sampler uses a 128-strip rectangle/alias table. On the development
  machine, direct standardized-core generation is about 95M draws/sec; `u` to
  theta transport is not the bottleneck.
- `determine_regime()` uses a warm-started Newton solve from the previous
  mode when the reflection branch is unchanged. This is exact Newton
  refinement, not an approximation shortcut, and removes most update-side
  cost in the Thompson hot path.
- Adaptive contraction was explored and deliberately left out. The main
  fixed-extreme cost is now alpha/remainder policy, not common-core draw
  mechanics.

Background (session findings, verified numerically on a
theta x nu x R grid, theta in [0.02,0.4], nu in {100,1000,1e4}, R up to 256):

- The committed corner-min core fights *location* drift of the posterior
  (mode moves R*sigma^2 against width sigma) and collapses for extreme arms:
  alpha ~ 1e-20 at theta=0.02, nu=1000, R=1.
- Recentering each shape on its own mode and rescaling by its own sigma
  ("affine transport") makes the standardized shape a near-pivot:
  alpha >= 0.98 over the entire grid, out to the reflection room
  nu*(1-2*zbar). Raw shape gaps on the validation grid are about 1e-5
  nats, covered by the 1e-3 deflation margin.
- alpha laws: untranslated horizon ~ theta*sqrt(nu) pulls; translate-only
  ~ theta*nu; translate+scale ~ room (not a statistical limit).
- The superseded `current_common_alpha` path ran a 2048-pt quadrature per
  parameter change: ~100x the cost of the exact hull rebuild (~20 B-evals)
  it was meant to avoid. The implemented design has zero quadrature on the
  per-update path.

## 1. Summary

Replace the corner-min common core with a **standardized (pivot) core**:

- The core is a distribution over the standardized variable
  `u`, built once per reuse range from K anchor shapes, certified as a
  pointwise log-domain lower bound (after a validated margin deflation)
  for every in-range posterior's standardized shape.
- The stream buffer stores **u-draws** (dimensionless), not thetas.
- Per query (chi, nu): transport `eta = eta_star_q + sigma_q * u`
  (one fused multiply-add; both factors already computed by
  `determine_regime`), then `theta = sigmoid(eta)`, reflect if needed.
- The mixture weight `alpha_hat` is a **build-time constant** (the sigma_q
  dependence cancels; see 2.4). Per-update hot-path cost: zero.
- The remainder branch (prob `1 - alpha_hat`, ~1-2% in the target regime)
  builds the exact hull at the current parameters and computes Z_q by a
  short standardized quadrature. Exactness of the overall scheme does not
  depend on alpha_hat being tight, only on `alpha_hat <= alpha_max` (2.4).

Correctness invariants (non-negotiable):
- I1. Every accept test clamps the log accept ratio to <= 0.
- I2. The core is used only while `(chi_eff, nu)` lies in the certified
  range (triangle) of the current build, on the same reflection branch.
- I3. `alpha_hat <= alpha_max(q)` for every in-range q (certified by
  construction + validated constants; see 2.4 and the validation script).
- I4. The remainder density uses the exact same `alpha_hat` and the exact
  same deflated core shape `S` as the split, with Z_q computed at the
  current parameters.

## 2. Mathematical specification

### 2.1 Objects and notation

All in the low-Z chart (`chi_eff = reflect ? nu-chi : chi`, `zbar <= 1/2`,
mode `eta* <= 0`). Unnormalized log-target:

    g_q(eta) = chi*eta - nu*B(eta),        q = (chi, nu)

Build origin `(chi0, nu0)`; certified range = triangle

    T(R) = { (chi0+dchi, nu0+dnu) : 0 <= dchi <= dnu <= R },
    R <= room = nu0 - 2*chi0        (keeps the branch un-reflected)

Standardized shape of q (recentered on its own mode, rescaled by its own
sigma; peak value 0 at u=0):

    Ghat_q(u) = g_q(eta*_q + sigma_q*u) - g_q(eta*_q)
              = -nu * D_B(eta*_q + sigma_q*u, eta*_q)

with `D_B` the Bregman divergence of B, `sigma_q = 1/sqrt(nu*B''(eta*_q))`.

### 2.2 Core definition

Anchors: one point at the range origin plus K-1 points on the far edge of
the triangle (`dnu = R`),
`zbar_a = linspace(chi0/(nu0+R), (chi0+R)/(nu0+R), K-1)`. For each anchor:
mode `eta*_a`, `sigma_a`, and shape `Ghat_a(u)` per 2.1.

Core shape, truncated to `|u| <= U_MAX` and deflated by the validated
margin:

    S(u) = min_a Ghat_a(u) - CB_CORE_SHAPE_MARGIN,   |u| <= U_MAX
    S(u) = -inf otherwise

Core density: `c(u) = exp(S(u)) / Z_S`, `Z_S = integral of exp(S)` over
the window (Simpson, build time). `S` is concave (min of concave
functions), so the existing piecewise-exponential hull machinery
(`cb_hull_t`, tangent construction) applies directly in u-space with unit
sigma: hull points at fixed offsets {-3,-1.5,0,1.5,3}, slope at a point
taken from the active (argmin) anchor — a tangent of the active concave
branch dominates the min everywhere, so the envelope is valid.

Transported core at query q: sample `u ~ c`, output
`eta = eta*_q + sigma_q*u`. Its density in eta is
`c((eta-eta*_q)/sigma_q)/sigma_q`.

### 2.3 Validity and certification

Requirement: for every q in T(R),

    S(u) <= Ghat_q(u)   for all u.          (V)

(V) implies the transported, alpha-weighted core is pointwise below the
target, so the remainder is a nonnegative density.

Certification strategy (two layers, house style — validated constants
with a documented offline script, like CB_SIGMA_CF):

- Runtime: origin + K-1 far-edge anchors as above. (For the translate-only family we
  have a proof that the far edge suffices — constant-zbar rays increase
  nu*D_B monotonically — and that in the B''' > 0 zone, i.e. when the
  whole evaluation window stays at eta < 0, the single far corner is the
  exact infimum. For the scaled family the ray argument does NOT carry
  over; the origin anchor is needed for the finite-nu standardized tails,
  and the residual is covered by the margin plus a runtime coarse scan.)
- Offline: `tools/validate_core_margins.py` (new) scans the usage grid
  (theta in [0.01, 0.49], nu in [CB_CORE_MIN_NU, 1e6] log-spaced, R up to
  CB_CORE_RANGE_MAX, fine z-grid over each triangle's boundary AND a
  coarse interior grid) and asserts

      max over grid of [ min_a Ghat_a(u) - Ghat_q(u) ] < CB_CORE_SHAPE_MARGIN.

  This script gates any change to K, U_MAX, RANGE_MAX, or the anchor
  placement. Measured raw violations are about 1e-5 nats vs margin 1e-3.

### 2.4 Mixture split and alpha_hat

Exact per-query weight of the transported core inside the target:

    alpha_max(q) = Z_S / W_q,    W_q = integral of exp(Ghat_q(u)) du

(The sigma_q and peak_q factors cancel between the transported-core mass
`sigma_q * Z_S * exp(peak_q)` and `Z_q = sigma_q * W_q * exp(peak_q)`.
This is what makes a constant split weight possible.)

Any `alpha_hat <= min over T(R) of alpha_max(q)` yields a valid mixture
`f_q = alpha_hat * c_transported + (1-alpha_hat) * r_q` with `r_q >= 0`.
We use the build-time constant

    alpha_hat = Z_S / ( max_a W_a * exp(CB_CORE_W_SAFETY) )

where `W_a` are the anchor values of W — free at build time, since the
same K x N_quad matrix of `Ghat_a` evaluations yields both `S` (column
min) and every `W_a` (row integral). CB_CORE_W_SAFETY covers the
between/off-anchor variation of W over the triangle and is validated by
the same offline script (assert `max_T W_q <= max_a W_a * exp(safety)`).

Gate: if `alpha_hat < CB_CORE_MIN_ALPHA`, do not install the core; fall
back to exact-rebuild-per-update (current fallback behavior).

### 2.5 Remainder

With `u_q(eta) = (eta - eta*_q)/sigma_q`, the remainder is proportional to

    r_q(eta) ∝ g_q(eta) - exp( log(alpha_hat) + logZ_q
                               - log(sigma_q) - log(Z_S) + S(u_q(eta)) )

`logZ_q` is computed exactly ONCE per remainder event at the current
parameters: Simpson, CB_CORE_QUAD_N points over
`eta*_q ± CB_CORE_QUAD_WIDTH * sigma_q` (documented tail bound; 12 sigma
=> relative truncation ~1e-31). Sampling: proposals from the exact hull
at q (`build_hull_exact`), accept test
`logdiffexp(g_q(eta), log_subtrahend(eta)) - log_env`, clamped <= 0 (I1).
`S(u)` outside the window is -inf and logdiffexp degrades gracefully to
plain g_q. I3 guarantees the subtrahend never exceeds g_q.

## 3. Data structure changes (`cb_stream_t`)

Removed from the legacy corner-min implementation: `hull_common`,
`hull_chi0`, `hull_nu0`, `hull_range_updates`,
`core_logZ`, `alpha_chi`, `alpha_nu`, `alpha`, `alpha_valid`,
`common_core_log_f`, `logdiffexp` stays (used by remainder),
`current_common_alpha`, `build_hull_common`, `build_corner_pieces`,
`piece_*`, `unique_sorted_boundaries`, `cmp_double`,
CB_ARS_MAX_HULL back to 9 (u-space hull is m=5).

Added:

    /* Standardized common core (certified pivot cache) */
    int    core_valid;         /* 1 = core installed and range-certified   */
    int    core_reflect;       /* reflection branch at build               */
    double core_chi0, core_nu0;/* range origin (low-Z chart)               */
    double core_R;             /* certified update horizon (pulls)         */
    double core_zA[CB_CORE_K]; /* anchor zbars (origin + far edge)         */
    double core_eta[CB_CORE_K];/* anchor modes                             */
    double core_sig[CB_CORE_K];/* anchor sigmas                            */
    double core_nu1;           /* nu0 + R (anchor nu)                      */
    cb_hull_t core_hull;       /* piecewise-exp envelope over u, m=5       */
    double core_logZS;         /* log integral of exp(S) over |u|<=U_MAX   */
    double core_alpha_hat;     /* build-time constant split weight          */

Buffer semantics: `buf[]` now holds **u-draws from the core** while
`regime == ARS && core_valid`. (Gamma/CF/prior refills still store theta
as today; those regimes drain unchanged.) The sign-bit reflect encoding
is dropped for ARS core draws — reflection is applied at drain time from
the live `s->reflect`.

## 4. Algorithms

### 4.1 build_core (called from ARS refill when !core_valid)

    room = nu - 2*chi_eff
    if (nu < CB_CORE_MIN_NU || room < CB_CORE_MIN_ROOM) -> no core; return
    R = min(CB_CORE_RANGE_ROOM_FRAC * room, CB_CORE_RANGE_MAX)
    nu1 = nu + R;  zA = [chi_eff/nu, linspace(chi_eff/nu1, (chi_eff+R)/nu1, K-1)]
    for each anchor: eta*_a (cb_mode), sigma_a, row of Ghat_a on the
        Simpson u-grid (CB_CORE_BUILD_N points, |u| <= U_MAX)
    S = column-min - CB_CORE_SHAPE_MARGIN
    runtime coarse scan over origin/interior/far-edge candidates; if any
        scanned Ghat_q falls below S, do not install the core
    core_logZS = Simpson(exp(S));  W_a = Simpson(exp(Ghat_a)) per row
    alpha_hat = exp(core_logZS - max_a log W_a - CB_CORE_W_SAFETY)
    if (alpha_hat < CB_CORE_MIN_ALPHA) -> no core; return
    build core_hull over S (offsets {-3,-1.5,0,1.5,3}, tangents from the
        active anchor); install fields; hull_rebuilds++

Cost: K x CB_CORE_BUILD_N B-evals (~5*257 ~ 1.3k) + m=5 hull points.
Amortized over the horizon (>= CB_CORE_MIN_ROOM/2 pulls) => a few
B-evals per pull.

### 4.2 core_covers_current (replaces hull_covers_current)

    core_valid && reflect == core_reflect
    && dnu  = nu - core_nu0     in [0, core_R]
    && dchi = chi_eff - core_chi0 in [0, dnu]     (eps-tolerant)

### 4.3 update / update_batch

    chi += x; nu += 1;  rem_valid = 0;  determine_regime(s)
    if (regime == ARS && core_covers_current(s))
        keep buffer (u-draws remain valid — transport is applied at drain)
    else
        core_valid = 0; buf_pos = buf_filled = 0

No quadrature, no alpha work, no hull work. (determine_regime already
recomputes eta_star and sigma — the transport parameters.)

### 4.4 refill (ARS branch of do_refill)

    if (!core_valid) build_core(s)
    if (core_valid)  draw_core_batch(s, n, s->buf)   /* u-draws */
    else             draw_ars_batch_exact(s, n, s->buf) /* theta, as today,
                                                       hull_valid=0 after */

draw_core_batch: proposals from core_hull (u-space), accept iff
`log(rand) <= S(u) - env(u)` (clamped; S eval = K anchor B-evals, or the
active-anchor short-circuit: evaluate anchors in order, early-out when
the running min already fails the test).

### 4.5 draw (cb_stream_draw, ARS + core_valid branch)

    take  = min(n - drawn, buf_filled - buf_pos)
    k_rem = binom_wait2_rem(rng, take, 1 - core_alpha_hat)
    for (take - k_rem) core draws:
        u = buf[buf_pos++]
        eta = eta_star + sigma*u            /* live q's transport */
        th = sigmoid(eta), clamp to (0,1), th = reflect ? 1-th : th
    if (k_rem): draw_remainder_batch(k_rem) per 2.5 (build_hull_exact at
        current q, one CB_CORE_QUAD_N quadrature for logZ_q, rejection
        with the logdiffexp subtrahend), reflect applied, then place the
        `k_rem` remainder samples into a uniformly chosen subset of the
        `take` output positions and fill the complement from the core
        buffer. Full Fisher-Yates is unnecessary; only the remainder labels
        need random positions. If `k_rem == 0`, do no placement/shuffle work.

Large-request bypass (`n > buf_capacity`): unchanged
(`draw_current_batch` -> exact hull, buffer reset).

## 5. Constants (all documented + validation-script-gated)

    CB_CORE_K            5      anchors: origin + K-1 on the far edge
                                (origin anchor REQUIRED: smallest nu =>
                                fattest standardized tails; far-edge-only
                                violates at the origin by up to ~20 nats
                                at large R/nu — measured 2026-07-04)
    CB_CORE_U_MAX        8.0    core support half-width (standardized)
    CB_CORE_BUILD_N      257    Simpson points at build
    CB_CORE_SHAPE_MARGIN 1e-3   nats; raw scan violations ~1e-5 (2 orders headroom)
    CB_CORE_W_SAFETY     1e-3   log-space W envelope; raw excess ~2.5e-4 — do not shave
    CB_CORE_MIN_ALPHA    0.5    install gate
    CB_CORE_MIN_NU       150.0  below this, exact rebuilds are cheap enough
    CB_CORE_MIN_ROOM     128.0  amortization gate (build cost / baseline)
    CB_CORE_RANGE_ROOM_FRAC 0.5
    CB_CORE_RANGE_MAX    256.0
    CB_CORE_QUAD_N       257    remainder logZ_q Simpson points
    CB_CORE_QUAD_WIDTH   12.0   sigmas; truncation ~1e-31 relative

New script `tools/validate_core_margins.py` (numpy; run with
DLA_MLX venv) asserting 2.3 and 2.4 bounds over the usage grid; its
output table is pasted into the constants' comment block, matching the
existing style for CB_SIGMA_CF / CB_GAMMA_SQRTN_MAX.

## 6. MEX / MATLAB API

- `cb_stream_mex('core_stats', ptr)` -> `[alpha_hat, core_draws,
  remainder_draws, rebuilds, core_active]`; counters added to
  `cb_stream_t` (u64 core_draws, remainder_draws).
- `CbStreamSampler.core_stats()` wrapper; `rebuilds()` kept.
- No changes to create/draw/update signatures.

## 7. Validation and benchmarks

Implemented checks:

- `tests/cb_smoke_test.m` keeps the existing regime/range/reflection tests
  and adds:
  - update/draw KS regression for the stale-cache bug class,
  - large-request bypass sanity,
  - transported-core engagement under update/draw cycling.
- `tools/validate_core_margins.py` is the offline numerical gate for
  shape and W-safety margins.
- `tools/diagnose_core_alpha.py` decomposes alpha into raw overlap, shape
  margin, scanned W variation, and final `alpha_hat`; this caught the
  overconservative `CB_CORE_W_SAFETY=0.05` tax.
- `tools/build_core_bench.sh` builds the C-only benchmark
  (`-DCB_CORE_BENCH`) used to separate core generation, draw mechanics,
  update-only cost, update/draw cost, and remainder cost.
- `examples/core_stream_benchmark.m` is the MATLAB-facing benchmark with
  MEX overhead included.

Representative C-only development numbers on the implementation machine:

- direct standardized-core generation: about 95M draws/sec,
- `u -> theta` transport: about 600M draws/sec,
- update-only after warm-started mode solve: about 30M updates/sec,
- extreme Thompson update/draw: about 6M iterations/sec,
- interior ARS update/draw: about 21M iterations/sec,
- CF update/draw: about 31M iterations/sec.

## 8. Future work

The remaining fixed-extreme slowdown is alpha/remainder policy, not the
mechanical common-core draw path. Useful future knobs are:

- adaptive buffer size policy,
- alpha/range policy for stationary or slowly drifting extreme arms,
- optional remainder sampler memoization when repeated requests arrive at
  unchanged parameters,
- stronger offline certification grids if `CB_CORE_K`, `CB_CORE_U_MAX`,
  `CB_CORE_RANGE_MAX`, or anchor placement changes.

## 10. Generalization note (the "lazy core" pattern)

The construction is not CB-specific. Ingredients: (i) a log-concave
exponential-family posterior with cheap mode/curvature, (ii) an update
cone with a monotone structure (here: constant-zbar rays proving the
far-edge reduction), (iii) an affine transport under which standardized
shapes are near-pivotal (Bernstein-von Mises finite-nu version — here
exact in the Gamma-tail limit, where D_B(xi; eta*) = phi(xi/eta*) is a
pure scale family). Any stream sampler meeting (i)-(iii) — Gamma rates,
Gaussian location, Beta-Bernoulli in the natural parameter — admits the
same certified pivot cache: buffer standardized draws once, transport
per query, pay exact-cost only on the (binomially chosen) remainder.
Worth a short note in the repo docs once Phase 1 lands.
