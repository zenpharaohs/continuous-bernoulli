function eta_star = cb_mode(chi, nu)
% CB_MODE  Newton solver for mode of CB conjugate posterior.
%
%   eta_star = cb_mode(chi, nu)
%
%   The mode satisfies f'(eta*) = 0 where f(eta) = chi*eta - nu*B(eta).
%   Equivalently:  B'(eta*) = chi/nu = Z_bar  (empirical mean of losses).
%
%   Newton iteration:
%     eta_{n+1} = eta_n - (B'(eta_n) - Z_bar) / B''(eta_n)
%   converges quadratically from warm start eta_0 = logit(Z_bar).
%   Typically 2-4 iterations.
%
%   Inputs:
%     chi  = sum of observed losses = N_k * Z_k   (scalar, >= 0)
%     nu   = number of pulls = N_k                (scalar, > 0)
%
%   Requirements:
%     chi/nu must be in (0,1).  This is guaranteed when Z_k in (0,1),
%     i.e. when arm losses are genuine CB samples.  chi/nu = 0 or 1
%     correspond to degenerate cases (all losses 0 or all losses 1).
%
%   See also: cb_ars, cb_sample, bft_all

% --- Input validation ---
if nu <= 0
    error('cb_mode:badNu', 'nu must be > 0 (got nu=%.6g)', nu);
end
xbar = chi / nu;
if xbar <= 0 || xbar >= 1
    error('cb_mode:badXbar', ...
        ['chi/nu = %.6g is outside (0,1).\n' ...
         'chi = N_k*Z_k and nu = N_k must give Z_bar in (0,1).\n' ...
         'Got chi=%.6g, nu=%.6g.'], xbar, chi, nu);
end

% --- Newton iteration ---
eta = log(xbar / (1-xbar));   % logit warm start (nearly exact for extreme xbar)

for iter = 1:40
    [~, bp, bpp] = bft_all(eta);
    res  = bp - xbar;             % residual: zero at mode
    step = res / bpp;             % Newton step (bpp = B'' > 0 always)
    eta  = eta - step;
    if abs(step) < 1e-14 * (1 + abs(eta))
        break
    end
end

if iter == 40
    warning('cb_mode:noConverge', ...
        'Newton did not converge in 40 iterations (residual=%.3e)', abs(res));
end

eta_star = eta;
end
