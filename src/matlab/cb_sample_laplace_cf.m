function theta = cb_sample_laplace_cf(chi, nu, n)
% CB_SAMPLE_LAPLACE_CF  Cornish-Fisher corrected Laplace sampler.
%
%   theta = cb_sample_laplace_cf(chi, nu, n)
%
%   Draws n samples from the CB conjugate posterior using the
%   Cornish-Fisher transformation of a standard normal.
%
%   Algorithm:
%     1. Find mode eta* and curvature via Newton (already fast)
%     2. Compute sigma = 1/sqrt(nu*B''(eta*))
%     3. Compute kappa3 = B'''(eta*) / B''(eta*)^(3/2)  [standardized skewness]
%     4. Draw u ~ N(0,1)
%     5. eta = eta* + sigma*u - (kappa3/(6*sqrt(nu)))*sigma*(u^2-1)
%     6. Return theta = sigmoid(eta)
%
%   The CF transformation maps N(0,1) to approximately the posterior
%   distribution.  The induced density of this transformation (via
%   change of variables) agrees with the true posterior to O(1/nu)
%   in total variation.  Validated by PIT testing in cb_cf_test.m.
%
%   Speed: ~20-100M samp/s vs ~3M for ARS.  Appropriate for large-nu
%   moderate-Z_bar arms where ARS is the bottleneck.
%
%   NOT appropriate outside the range where PIT tests pass.
%   Use cb_sample for automatic triage.
%
%   See also: cb_sample, cb_sample_gamma, cb_cf_test

eta_star = cb_mode(chi, nu);
[~, ~, bpp] = bft_all(eta_star);
sigma = 1.0 / sqrt(nu * bpp);

% kappa3 via analytic B''' (bft_d3): B''=R(eta^2) => B'''=2*eta*R'(eta^2)
% Accuracy ~2*eps vs ~4e-14 for 5-point FD; also faster (no extra bft_all calls).
bppp   = bft_d3(eta_star);
kappa3 = bppp / bpp^(1.5);

u   = randn(n, 1);
eta = eta_star + sigma * u ...
    - (kappa3 / (6*sqrt(nu))) * sigma * (u.^2 - 1);

theta = 1.0 ./ (1.0 + exp(-eta));
theta = max(realmin, min(1-eps, theta));
end
