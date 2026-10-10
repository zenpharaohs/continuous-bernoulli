# Validation

`cb_validate.m` uses two PIT-based validation tracks.

## Streaming Legendre

The primary large-N diagnostic is the streaming shifted-Legendre validator. It
updates Legendre coefficient sums directly from PIT samples with fixed memory,
then reports adaptive block SNR and the second-order Hellinger proxy. This is
the right tool for `N=1e8` to `N=1e9` runs because it does not store or sort the
PIT samples.

This algorithm has been extracted into the sibling Python package
`streaming-pit-validate`. The MATLAB implementation in `cb_validate.m` remains
in place so the current validation suite is self-contained.

## Smoothed-Spacing Hellinger

The optional Hellinger certification path sorts PIT samples and computes a
smoothed-spacing estimate. With

```matlab
cb_validate_hq_cv = true;
cb_validate_mode = 'thorough';
cb_validate
```

`cb_validate.m` also applies a fitted-Beta control variate and reports
classifier sample scales:

- `H2_smooth`: deterministic smoothed-spacing squared Hellinger estimate.
- `H2_CV`: fitted-Beta control-variate estimate.
- `no_CV`: iid sample count below which no classifier is certified to have
  total error below `1/2`.
- `some_CV`: iid sample count after which some classifier is certified to have
  total error below `1/2`.

This policy mirrors the sibling Python package `hellinger-qualify`: use the
streaming Legendre track for cheap large-N detection, and use fitted-Beta
smoothed-spacing Hellinger for near-null certification when sorting the PIT
sample is acceptable.

## Exact endpoint qualification

Both `cb-validate` (all modes) and `cb_validate.m` automatically include endpoint
qualification. They cover the reported SPONG states, gaps down to 2e-32,
both reflections, and exact ARS/core/remainder sampling. Two distribution cases
retain a live core buffer across statistics replacement. Both suites also
check 3000 replacements and explicit failure of invalid remainder subtraction.
Three tiny lower-tail states include the reported nu=7 stall and exercise both
batched ARS and single draws from live stream buffers. Their reflected interior
counterparts cannot be represented in double precision; the existing reflected
cases use representable gaps.

The shared C diagnostic flag `CB_VALIDATE_ETA` disables Gamma/CF approximations
and returns eta before sigmoid rounding through the usual stream, reflection,
and buffer path. Ordinary builds retain their theta output and regime policy.
This prevents endpoint rounding from hiding a wrong latent distribution.
All 35 latent cases use a two-sided KS limit of 3/sqrt(N), plus reference-grid
error. The independently evaluated density CDF must agree within 5e-7 on
65537- and 131073-point grids. Each suite uses at most 100000 samples per
endpoint case, even for larger ordinary PIT runs.

### MATLAB

`cb_validate_endpoints.m` is self-contained: it requires MATLAB and the same
configured compiler used by `build_cb_stream`, with no Python or extra toolbox.
It tests `draw`, `set_stats`, `peek`, `update`, and `update_batch` through the
loaded production MEX. It then builds `cb_stream_eta_mex` from the same MEX
source with `CB_VALIDATE_ETA`, samples through that gateway, and computes its
reference CDF in MATLAB. Live-buffer diagnostics verify that a covered buffer
is actually retained. The suite also tests both single and batched MEX errors
on an invalid remainder and verifies interruption by the rejection budget.

Validation sets an optional five-second CPU budget for each MEX call. The
exact rejection loops check the budget every 4096 iterations; it can interrupt a
stuck MEX loop even though MATLAB timers cannot run during the call. The budget
is disabled for ordinary use and restored after validation. MATLAB also checks
a 60-second elapsed-time limit between calls, including diagnostic compilation;
this elapsed-time check cannot preempt the compiler itself. A failure aborts
validation. Results remain in `endpoint_results` after `cb_validate`.

After rebuilding the production MEX, run the focused suite with:

```matlab
endpoint_results = cb_validate_endpoints(16000);
```

Or run the normal suite:

```matlab
build_cb_stream
cb_validate_mode = 'quick';
cb_validate
```

### Python

Python qualification compiles an unmodified API build and an eta diagnostic
build in temporary storage. Its reference CDF uses NumPy, without SciPy.
The entire worker has a 60-second subprocess deadline, including compilation.
This qualifies the packaged C source; ordinary Python PIT checks exercise the
loaded extension. It requires a C compiler (`cc`, `clang`, or `gcc`).
Endpoint results are included in JSON schema `cb_sampler.python_validation.v2`
with `kind="endpoint_eta"`, sample count, path, KS limit, and reference-grid
error. To run just the Python section from the repository:

```bash
python3 tests/exact_endpoint_regression.py --n 16000
```
