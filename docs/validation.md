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

Both `cb-validate` (all modes) and `cb_validate.m` now run the same endpoint
qualification suite automatically. It checks the two reported SPONG states,
smaller gaps down to 1e-10, both reflections, and exact ARS/core/remainder
sampling. Two cases retain a live core buffer across statistics replacement
and test the subsequent draw distribution. It also checks 3000 replacements,
regime transitions, and explicit failure of an invalid remainder subtraction.

The C source is compiled twice in temporary storage: an unmodified API build
for stream checks, and an instrumented build that returns eta before sigmoid
for distribution checks. This qualifies the packaged C source; ordinary Python
and MATLAB PIT checks continue to exercise their loaded backend. Endpoint
rounding therefore cannot hide a wrong latent distribution. The independent
NumPy density integral must agree within 5e-7 on 65537- and 131073-point grids.
All 26 latent cases use a two-sided KS limit of 3/sqrt(N), plus reference-grid
error. This conservative limit accounts for testing multiple cases. The entire
endpoint worker, including compilation and rejection checks, has a 60-second
deadline; a timeout or build failure fails validation.

This section requires NumPy and a C compiler (`cc`, `clang`, or `gcc`), but not
SciPy. It uses at most 100000 samples per endpoint case, even for larger
ordinary PIT runs. MATLAB uses `python3` by default; set `cb_validate_python`
to an interpreter path if needed. Endpoint results are included in Python's
JSON report, now schema `cb_sampler.python_validation.v2`, with
`kind="endpoint_eta"`, sample count, path, KS limit, and reference-grid error.

To run just this section from the repository:

```bash
python3 tests/exact_endpoint_regression.py --n 16000
```
