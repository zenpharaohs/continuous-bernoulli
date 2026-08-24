function [a, b] = cb_to_beta_num(chi_p, nu)
%CB_TO_BETA_NUM  Beta(a,b) moment-matched to CB posterior (chi_p, nu).
%
%   Moments computed by MATLAB integral() over (-Inf, Inf) in eta-space.
%   No grid, no march-outward, no scalar bft_b loop.
%   Degenerate cases: Jeffreys posterior Beta(chi+0.5, nu-chi+0.5).

    if nu <= 0 || chi_p <= 0 || chi_p >= nu
        a = chi_p + 0.5;  b = nu - chi_p + 0.5;  return;
    end

    eta_s    = cb_mode(chi_p, nu);
    log_peak = chi_p*eta_s - nu*bft_b(eta_s);

    p_s  = @(eta) exp(chi_p*eta - nu*bft_b(eta) - log_peak);
    sig  = @(eta) 1 ./ (1 + exp(-eta));
    opts = {'AbsTol',1e-10,'RelTol',1e-8,'ArrayValued',false};

    Z_q = integral(p_s,                       -Inf, Inf, opts{:});
    m1  = integral(@(e) sig(e)   .* p_s(e),   -Inf, Inf, opts{:}) / Z_q;
    m2  = integral(@(e) sig(e).^2 .* p_s(e),  -Inf, Inf, opts{:}) / Z_q;
    v   = max(m2 - m1^2, 0);

    v = min(v, m1*(1-m1) * 0.9999);
    v = max(v, 1e-9);
    c = max(m1*(1-m1)/v - 1, 1e-4);
    a = m1 * c;
    b = (1-m1) * c;
end
