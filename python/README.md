# cb-sampler Python package

Python bindings for the validated continuous-binomial C sampler backend in
this repository.  Python users do not need MATLAB to use or validate the
sampler: the package includes its own validation harness and examples.

`draw_streams(streams)` draws once from every independent `CbStream` through a
single C-extension call.  It is intended for Thompson rounds with many live
posteriors, where separate Python calls would dominate the actual sampler.

`stream.set_stats(chi, nu)` replaces, rather than increments, a stream's
sufficient statistics while preserving its RNG state.  This supports rested
descent-bandit pseudo-posteriors such as `(chi,nu)=(N*Z,N)`, where `Z` is the
current sample-and-hold loss and `N` is the arm's selection count.

The API implements the closed posterior family.  `(chi,nu)=(0,0)` is the
explicit startup prior.  For `nu > 0`, `chi=0` and `chi=nu` are the weak-limit
point masses at zero and one and therefore draw those endpoints
deterministically.  All other valid states satisfy `0 < chi < nu` and use the
ordinary sampling regimes.  Observations may lie anywhere in `[0,1]`.

## Validate

```bash
python -m cb_sampler.validation --mode smoke
python -m cb_sampler.validation --mode quick
python -m cb_sampler.validation --mode standard --json-out cb_validation.json
```

Editable installs also provide:

```bash
cb-validate --mode quick
cb-validate --mode standard --json-out cb_validation.json
cb-example stream
cb-example posterior
cb-example bandit
```

The validation harness computes an exact eta-grid CDF in Python, applies the
PIT transform to C-backed sampler draws, reports KS/AD/CvM warning statistics,
and uses adaptive Legendre chi-square as the decision signal.  The JSON artifact
records platform metadata, the git commit when available, every case result,
and the overall pass/fail verdict.
