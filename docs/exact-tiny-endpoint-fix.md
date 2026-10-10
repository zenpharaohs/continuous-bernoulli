# Exact sampling at tiny interior statistics

SPONG reported a second stall on commit `3fa5188`, after replacing stream
statistics with `chi = 7 * 3.8758505860895066e-32`, `nu = 7`. A child-process
replay confirms that `set_stats` returns but the next draw exceeds five seconds.
This is a different failure from the stale mode/scale issue fixed previously.

## Cause and correction

The exact ARS hull places tangent points around the mode using its local scale.
For this state the mode is about `-2.58e31`, the scale about `9.75e30`, and the
rightmost point about `8.33e30`. Computing the tangent intercept as
`f(eta) - eta*f'(eta)` subtracts terms of order `1e31`. Rounding produces an
intercept around `2e15`, instead of the correct value around `491`. The resulting
rightmost segment dominates the proposal weights and has effectively zero
acceptance probability.

There is also a scale-dependent intersection error: distinct slopes of order
`1e-31` were treated as equal by an absolute `1e-14` cutoff. Correcting that
cutoff alone does not cure the cancellation above.

The intercept is now evaluated directly as `nu * (eta*B'(eta) - B(eta))`.
The expression is even in eta. For `a = abs(eta) > 32`, it is computed as

```
nu * (log(a) - 1 + a*q/(1-q) - log1p(-q)), q = exp(-a).
```

This retains the exponential correction without subtracting large linear
terms. Hull intersections, segment weights, and proposal envelope evaluations
use these stable intercepts. Exact hull slope differences use a zero test
rather than an absolute cutoff. The density and input statistics are unchanged.

Plain ARS now has a finite proposal cap, as the remainder sampler already did.
A failed envelope returns an error through stream refills and large requests;
no alternate sample is returned. Constructor prefill failures release the
stream and return NULL.

## Regression evidence

Both regular Python and self-contained MATLAB/MEX qualification now include
three additional tiny lower-tail states: the reported nu=7 state, `chi=7e-20`
at nu=7, and `chi=2e-32` at nu=2. They exercise direct ARS, the core-availability
path, and single draws from stream buffers. Independent density quadrature
checks the latent eta distribution before sigmoid underflow. A reflected
interior state at these gaps cannot be encoded in double precision; existing
representable reflected states remain covered.

The direct stream tests exercise constructor prefill, repeated replacements
among these states and live core states, and request sizes 1, 7, and 65 with a
64-element buffer. A focused Python regression has a five-second subprocess
deadline. Fault injection checks both ARS proposal exhaustion and error
propagation through single, buffered-batch, and larger-than-buffer requests.

On this machine:

- The original minimal replay exceeds five seconds; the rebuilt C stream draw
  returns in about four microseconds, and SPONG's bank call in about ten.
- All 35 Python latent cases pass at 100000 draws each; maximum KS is 0.00447,
  below the 0.00949 limit.
- MATLAB quick validation rebuilds both actual MEX binaries and passes all 35
  endpoint cases in 3.12 seconds, plus all 23 ordinary cases. Maximum endpoint
  KS is 0.00448, below 0.00949.
- The C smoke test and all 31 Python tests pass.
