# Exact sampler endpoint regression

The Gamma precision guard for reflected posteriors jumped to `ars_regime`
below the current mode/curvature calculation. Fresh streams used zero-initialized
mode and scale; streams updated through `cb_stream_set_stats` reused stale
values. This was a control-flow failure, not unsupported endpoint observations.

Instrumenting the SPONG replay with seed 1729 and a 4096-element buffer showed
at nu=289: effective chi=5.8772231454895518e-6, stored mode=-8.51693e7,
current mode=-4.91729e7, stored sigma=1.2167e7. At nu=425 the current mode
was -3.97885e7 while the stored mode was still -8.51693e7. These inconsistent
coordinates invalidated the core/remainder density decomposition. The exact
stall draw depends on buffer capacity and RNG consumption; SPONG's 64-element
buffer stalled near draw 599.

The guard now enters ARS before computing the current mode and scale. A separate
numerical correction removes the absolute 1e-8 zero-slope approximation in
exponential segment weights and inversions. Near an endpoint, a tiny slope
multiplied by a huge interval width need not be small. Only exactly zero slopes
use uniform inversion. Log subtraction and segment mass differences use
`expm1` to preserve small positive differences.

Remainder rejection allows at most one million proposals per requested sample.
Exhaustion propagates `-1` from `cb_stream_draw`; it does not return a sample from
a different density. Input statistics remain unmodified. Existing floating
point output rounding still applies: extremely small sigmoid draws underflow,
and reflected draws may be indistinguishable from one at double precision.

## Verification

Run `python3 tests/exact_endpoint_regression.py` with NumPy, SciPy, and a C
compiler. The regression compiles an unmodified API build and a test-only build
that records latent eta before sigmoid. It has a 60-second subprocess deadline.
It tests 3000 successive replacements, the two reported states, opposite
endpoints, reflected states, and replacements while buffers are live. If the
external SPONG CSV exists, it also replays the entire trace. A deliberately
invalid remainder subtraction verifies bounded failure.

Independent Python evaluation and SciPy quadrature supply the reference CDF;
they do not use the C mode solver, B function, or core normalization. There are
24 distribution checks, with 16000 draws each: direct exact ARS and the
core/remainder path where eligible, on both sides, for gaps 1.874462168e-5
at nu=594, 2.846340396e-5 at nu=258, 1e-10 at nu=2, .02 at nu=2, .01 at
nu=49, and 2 at nu=200. Capturing eta prevents rounded output endpoints from
concealing latent bias. Observed KS distances were 0.00478–0.01014 against a
0.018 threshold; all checks completed in 1.63 seconds on the development Mac.
The production low-endpoint Gamma approximation is an existing separate regime;
these numerical tests explicitly exercise the exact path on both sides.

The SPONG command using `CB_CORE_LIBRARY` set to the rebuilt shared library
completed all 1395 trace rows. The Python API suite also includes a portable
15-second timeout regression requiring no SciPy or compiler at test time.
