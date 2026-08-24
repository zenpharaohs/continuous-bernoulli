# License and algorithm provenance

The repository is distributed under the MIT License; see [`LICENSE`](../LICENSE)
for the complete license terms. This note is explanatory and does not modify
those terms.

The implementations in this repository are original code informed by
published mathematical and numerical methods, including:

- Marsaglia, Tsang, and Wang's exact Kolmogorov-Smirnov distribution method;
- Soderlind's PI digital-filter step-size controller;
- the Dormand-Prince adaptive Runge-Kutta tableau; and
- Gilks and Wild's adaptive rejection sampling method.

The Python package's runtime dependency is NumPy, which is distributed under
the BSD 3-Clause License. Build and test dependencies are development tools and
are not bundled with this package.
