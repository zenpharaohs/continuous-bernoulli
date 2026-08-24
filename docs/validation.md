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
