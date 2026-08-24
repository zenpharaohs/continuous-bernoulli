function [a_opt, b_opt, rho, H2] = cb_to_beta_hellinger_mu(chi, nu, varargin)
% CB_TO_BETA_HELLINGER_MU  Best Beta(a,b) for Thompson sampling with CB rewards.
%
%   [a_opt, b_opt, rho, H2] = cb_to_beta_hellinger_mu(chi, nu)
%
%   Finds Beta(a_opt, b_opt) minimizing the Hellinger distance between the
%   CB oracle's Thompson sampling loss proxy distribution and Beta(a,b) used
%   directly as a loss proxy:
%
%       H^2_mu = 1 - integral_0^1 sqrt(p_CB_mu(m) * betapdf(m,a,b)) dm
%
%   where p_CB_mu(m) is the push-forward of the CB posterior through
%   theta -> mu = B'(logit(theta)).
%
%   CONTEXT
%   -------
%   In CB Thompson sampling, the oracle draws theta ~ CB posterior and uses
%   mu = B'(logit(theta)) as the loss proxy (selection statistic).
%   A Beta Thompson sampler draws p ~ Beta(a,b) and uses p directly.
%   This function finds the Beta(a,b) whose samples best approximate the
%   oracle's loss proxy distribution in Hellinger distance.
%
%   This is DIFFERENT from cb_to_beta_hellinger, which minimizes H^2 between
%   the CB posterior and Beta(a,b) as distributions over the arm parameter
%   theta.  The theta-space Hellinger-optimal Beta concentrates near theta*
%   (the arm parameter, << mu for small mu), making it a poor loss proxy.
%   The mu-space Hellinger-optimal Beta concentrates near mu (the expected
%   loss), making it a useful Thompson sampler.
%
%   FORMULA
%   -------
%   On the eta grid (change of variables m = B'(eta), dm = B''(eta) deta):
%
%     rho_mu = integral sqrt(p_CB_mu * betapdf) dm
%            = integral sqrt(exp(lp_eta - log_Z_CB) * B'' * betapdf(B'(eta),a,b)) deta
%
%   Log integrand:
%     l_k = (lp_eta_k + log(B''_k) - lmax_eff)/2
%           + (a-1)/2 * log(mu_k) + (b-1)/2 * log(1-mu_k)
%   where lmax_eff = max(lp_eta + log(B'')).
%
%   Key difference from theta-space: B''(eta) * betapdf(mu, a, b) uses (a-1)
%   and (b-1) exponents (no J*betapdf algebraic simplification).
%   Gradient formula is structurally identical: a/2*(S1/S0 + psi(a+b)-psi(a)).
%
%   NORMALIZATION
%   -------------
%   Z_f0_mu = Z_CB * exp(-lmax_eff) = sum(exp(lp_eta - lmax_eff)) * deta
%   log(rho_mu) = lmax_g + log(S0) - 0.5*log(Z_f0_mu) - 0.5*betaln(a,b)
%   (same formula as theta-space, different Z_f0 and different log_g)
%
%   GRADIENT (exact on fixed grid, envelope theorem)
%   -------------------------------------------------
%   With g_k = exp(l_k - lmax_g) and weighted sums S0, S1, S2, S11, S22, S12:
%
%     d(log rho_mu)/d(log a) = a/2 * (S1/S0 + psi(a+b) - psi(a))
%     d(log rho_mu)/d(log b) = b/2 * (S2/S0 + psi_diff_stable(a,b))
%
%   Identical structure to cb_to_beta_hellinger (log_mu replaces log_sig).
%
%   INITIALIZATION
%   --------------
%   The mu-space optimal Beta concentrates near mu = chi/nu with variance
%   B''(eta*)/nu (the CB posterior push-forward variance).  Matching these
%   two moments gives concentration s = mu*(1-mu)*nu/B''(eta*) - 1.
%
%   Starting points:
%     1. Push-forward moment-match: mean=mu, var=B''(eta*)/nu  [best start]
%     2. Bernoulli-Beta Beta(chi+1, nu-chi+1): correct mean, var~mu(1-mu)/nu
%     3. Data-level MM cb_to_beta_data_mm: correct mean, var=B''(eta*) [too wide]
%     4. Boundary-matched: Beta(mu*nu, (1-mu)*nu)
%
%   REFLECTION FOR Z > 0.5
%   -----------------------
%   By CB symmetry, p_CB_mu(m|chi,nu) = p_CB_mu(1-m|nu-chi,nu), and
%   Beta(a,b)(1-m) = Beta(b,a)(m), so:
%     H^2_mu(CB(chi,nu), Beta(a,b)) = H^2_mu(CB(nu-chi,nu), Beta(b,a))
%   For Z > 0.5 we solve the reflected (low-Z) problem and swap (a,b).
%
%   OPTIMIZER: DOPRI5 with Soderlind PI controller
%   Same as cb_to_beta_hellinger -- see that file for details.
%
%   NUMERICAL STABILITY: same three cancellation hazards as theta-space
%   (betaln_stable, psi_diff_stable, psi1_diff_stable).
%   Additional guard: log(mu_k) and log(1-mu_k) clamped to [-700, -1e-15]
%   at grid boundaries where B'(eta) approaches 0 or 1.
%
%   INPUTS
%     chi    sum of observed losses (>= 0)
%     nu     pull count (> 0)
%
%   OPTIONS
%     'N_grid'   eta-grid points (default 2000)
%     'max_iter' max steps (default 200)
%     'tol'      gradient-norm tolerance (default 1e-9)
%     'verbose'  print trace (default false)
%
%   OUTPUTS
%     a_opt, b_opt   Hellinger-optimal Beta parameters (in mu-space)
%     rho            Hellinger affinity in [0,1]
%     H2             1 - rho
%
% See also: cb_to_beta_hellinger (theta-space posterior approximation)
%
% MIT License.  Andrew Mullhaupt, Stony Brook University AMS/QF, 2026.

%% Parse options
p = inputParser;
addRequired(p,  'chi');
addRequired(p,  'nu');
addParameter(p, 'N_grid',   2000,  @(x) isscalar(x) && x >= 100);
addParameter(p, 'max_iter', 200,   @(x) isscalar(x) && x >= 1);
addParameter(p, 'tol',      1e-9,  @(x) isscalar(x) && x > 0);
addParameter(p, 'verbose',  false, @(x) islogical(x)||(isnumeric(x)&&isscalar(x)));
parse(p, chi, nu, varargin{:});
N        = p.Results.N_grid;
MAX_ITER = p.Results.max_iter;
TOL      = p.Results.tol;
verb     = logical(p.Results.verbose);

%% Degenerate cases
if nu <= 0 || chi <= 0 || chi >= nu
    a_opt = max(chi + 0.5, 1e-4);
    b_opt = max(nu - chi + 0.5, 1e-4);
    rho = NaN;  H2 = NaN;
    return;
end

%% Reflection for Z > 0.5
% p_CB_mu(m|chi,nu) = p_CB_mu(1-m|nu-chi,nu)  and  Beta(a,b)(1-m) = Beta(b,a)(m)
% => H^2_mu(CB(chi,nu),Beta(a,b)) = H^2_mu(CB(nu-chi,nu),Beta(b,a))
reflected = (chi / nu > 0.5);
if reflected
    chi = nu - chi;
end

%% Stable special functions (identical to cb_to_beta_hellinger)
ASYM = 1e8;

    function bl = betaln_stable(a, b)
        if b > ASYM && b > a
            bl = gammaln(a) - (a*log(b) + a*(a-1)/(2*b));
        elseif a > ASYM && a > b
            bl = gammaln(b) - (b*log(a) + b*(b-1)/(2*a));
        else
            bl = betaln(a, b);
        end
    end

    function ps = psi_stable(x)
        if x > ASYM
            ps = log(x) - 1/(2*x) - 1/(12*x^2);
        else
            ps = psi(x);
        end
    end

    function d = psi_diff_stable(a, b)
        if b > ASYM && b > 10*a
            t = a/b;
            d = t*(1 - t/2 + t^2/6 - t^3/12 + t^4/20 - t^5/30);
        else
            d = psi(a + b) - psi(b);
        end
    end

    function d = psi1_diff_stable(a, b)
        if b > ASYM && b > 10*a
            t = a/b;
            d = -(t/b)*(1 - t + t^2 - t^3 + t^4);
        else
            d = psi(1, a + b) - psi(1, b);
        end
    end

%% Build fixed eta-grid
eta_star = cb_mode(chi, nu);
[~, ~, bpp_star] = bft_all(eta_star);
mu_star  = chi / nu;
sig_eta  = 1.0 / sqrt(max(nu * bpp_star, 1e-6));

f_peak  = chi * eta_star - nu * bft_b(eta_star);
LOG_THR = 35;
step    = max(sig_eta, 0.5);

eL = eta_star;
while chi*eL - nu*bft_b(eL) > f_peak - LOG_THR,  eL = eL - step;  end
eR = eta_star;
while chi*eR - nu*bft_b(eR) > f_peak - LOG_THR,  eR = eR + step;  end
eL = eL - step;  eR = eR + step;

eta  = linspace(eL, eR, N)';
deta = eta(2) - eta(1);

lp_eta = chi*eta - nu*bft_b(eta);   % log unnormalized CB posterior

%% B'(eta) and B''(eta) on the grid
% mu_grid = B'(eta) = mean value parameter (loss proxy axis)
% bpp_grid = B''(eta) = variance function
mu_grid  = zeros(N, 1);
bpp_grid = zeros(N, 1);
for k = 1:N
    [~, mu_grid(k), bpp_grid(k)] = bft_all(eta(k));
end

% Log of mu and 1-mu, clamped to avoid -Inf at boundaries
LOG_EPS  = -700;
log_mu   = max(log(max(mu_grid,   1e-300)), LOG_EPS);
log_1mmu = max(log(max(1-mu_grid, 1e-300)), LOG_EPS);
log_bpp  = log(max(bpp_grid, 1e-300));

%% Mu-space grid quantities
% lp_eff = lp_eta + log(B'') is the "effective" log unnormalized density
% whose half-value defines the mu-space f0 (analogue of log_f0 in theta-space)
lp_eff    = lp_eta + log_bpp;
lmax_eff  = max(lp_eff);

% f0_mu: the square root of the unnormalized CB push-forward integrand
% log_f0_mu = (lp_eta + log_bpp - lmax_eff) / 2
log_f0_mu = (lp_eff - lmax_eff) / 2;

% Z_f0_mu: the normalization denominator for the rho formula
% Z_f0_mu = Z_CB * exp(-lmax_eff) = sum(exp(lp_eta - lmax_eff)) * deta
% Note: uses lp_eta (not lp_eff) -- see derivation in function header.
Z_f0_mu     = sum(exp(lp_eta - lmax_eff)) * deta;
log_Z_f0_mu = log(Z_f0_mu);

%% Core: rho, gradient, and Hessian at ab = (log a, log b)
    function [rv, gv, Hv] = rho_grad_hess(ab)
        alpha = exp(ab(1));
        beta  = exp(ab(2));

        % Integrand weights
        % log_g = log_f0_mu + (a-1)/2*log_mu + (b-1)/2*log_1mmu
        % [uses (a-1),(b-1) because B''*betapdf doesn't simplify to a/2,b/2]
        log_g  = log_f0_mu + ((alpha-1)/2)*log_mu + ((beta-1)/2)*log_1mmu;
        lmax_g = max(log_g);
        g      = exp(log_g - lmax_g);

        % Weighted sums with g as weight
        S0  = sum(g)                         * deta;
        S1  = sum(g .* log_mu)               * deta;   % log mu
        S2  = sum(g .* log_1mmu)             * deta;   % log(1-mu)
        S11 = sum(g .* log_mu.^2)            * deta;
        S22 = sum(g .* log_1mmu.^2)          * deta;
        S12 = sum(g .* log_mu .* log_1mmu)   * deta;

        % log rho_mu:  same formula as theta-space with Z_f0_mu
        log_rv = lmax_g + log(S0) - 0.5*log_Z_f0_mu - 0.5*betaln_stable(alpha, beta);
        rv = exp(log_rv);

        if nargout < 2,  return;  end

        % Gradient in (log alpha, log beta) -- identical to theta-space formula
        diff_a = psi_stable(alpha+beta) - psi(alpha);
        diff_b = psi_diff_stable(alpha, beta);

        ga = alpha * rv/2 * (S1/S0 + diff_a);
        gb = beta  * rv/2 * (S2/S0 + diff_b);
        gv = [ga; gb];

        if nargout < 3,  return;  end

        % Hessian of log rho in (log alpha, log beta) -- same structure
        V11 = S11/S0 - (S1/S0)^2;
        V22 = S22/S0 - (S2/S0)^2;
        V12 = S12/S0 - (S1/S0)*(S2/S0);

        p1_a   = psi(1, alpha);
        p1_ab  = psi(1, alpha + beta);
        dp1_b  = psi1_diff_stable(alpha, beta);

        H_aa = ga + alpha^2/4 * V11 + alpha^2/2 * (p1_ab - p1_a);
        H_bb = gb + beta^2/4  * V22 + beta^2/2  * dp1_b;
        H_ab = alpha*beta/4   * V12 + alpha*beta/2 * p1_ab;
        Hv   = [H_aa, H_ab; H_ab, H_bb];
    end

%% Initialization: four starting points
% Best starting point in mu-space: match mean=mu and variance=B''(eta*)/nu,
% i.e., the CB push-forward posterior variance.
% s = mu*(1-mu) / (B''(eta*)/nu) - 1 = mu*(1-mu)*nu/B''(eta*) - 1
s_post  = max(mu_star*(1-mu_star)*nu/max(bpp_star, 1e-10) - 1, 0.01);
a_post  = max(mu_star * s_post,       0.01);
b_post  = max((1-mu_star) * s_post,   0.01);

% Data-level MM: matches E[X] and Var[X] -- correct mean but too wide
try
    [a_dl0, b_dl0] = cb_to_beta_data_mm(chi, nu);
    if ~isfinite(a_dl0)||~isfinite(b_dl0)||a_dl0<=0||b_dl0<=0
        a_dl0 = chi+1;  b_dl0 = nu-chi+1;
    end
catch
    a_dl0 = chi+1;  b_dl0 = nu-chi+1;
end

starts = [a_post,          b_post;           % push-forward moment-match [best]
           chi + 1,         nu - chi + 1;     % Bernoulli-Beta
           a_dl0,           b_dl0;            % data-level MM
           mu_star*nu,      (1-mu_star)*nu];  % Beta(mu*nu, (1-mu)*nu)

% Guard: all starts must be positive
starts = max(starts, 1e-3);

rho_s = zeros(size(starts,1), 1);
for s = 1:size(starts, 1)
    rho_s(s) = rho_grad_hess(log(starts(s,:)'));
end
[rho_best_start, ibest] = max(rho_s);
ab = log(starts(ibest,:)');
[rho_val, grad] = rho_grad_hess(ab);

if verb
    fprintf('cb_to_beta_hellinger_mu: start %d (a=%.4g b=%.4g) rho=%.6f\n', ...
        ibest, exp(ab(1)), exp(ab(2)), rho_val);
end

%% DOPRI5 with Soderlind PI controller (identical to cb_to_beta_hellinger)
dp_a21 = 1/5;
dp_a31 = 3/40;       dp_a32 = 9/40;
dp_a41 = 44/45;      dp_a42 = -56/15;      dp_a43 = 32/9;
dp_a51 = 19372/6561; dp_a52 = -25360/2187; dp_a53 = 64448/6561; dp_a54 = -212/729;
dp_a61 = 9017/3168;  dp_a62 = -355/33;     dp_a63 = 46732/5247;
dp_a64 = 49/176;     dp_a65 = -5103/18656;
dp_b1 = 35/384;   dp_b3 = 500/1113;  dp_b4 = 125/192;
dp_b5 = -2187/6784; dp_b6 = 11/84;
dp_e1 =  71/57600;    dp_e3 = -71/16695;   dp_e4 =  71/1920;
dp_e5 = -17253/339200; dp_e6 = 22/525;     dp_e7 = -1/40;

RKF_TOL  = max(TOL * 100, 1e-7);
H_MAX    = 4.0;
H_MIN    = 1e-13;
h        = 0.5;
PI_K1    = 0.7 / 5;
PI_K2    = 0.4 / 5;
err_prev = RKF_TOL;

k1 = grad;

for iter = 1:MAX_ITER
    gnorm = norm(k1);
    if gnorm < TOL
        if verb
            fprintf('cb_to_beta_hellinger_mu: converged iter=%d |g|=%.2e rho=%.10f\n', ...
                iter, gnorm, rho_val);
        end
        break;
    end

    [~, k2] = rho_grad_hess(ab + h*dp_a21*k1);
    [~, k3] = rho_grad_hess(ab + h*(dp_a31*k1 + dp_a32*k2));
    [~, k4] = rho_grad_hess(ab + h*(dp_a41*k1 + dp_a42*k2 + dp_a43*k3));
    [~, k5] = rho_grad_hess(ab + h*(dp_a51*k1 + dp_a52*k2 + dp_a53*k3 + dp_a54*k4));
    [~, k6] = rho_grad_hess(ab + h*(dp_a61*k1 + dp_a62*k2 + dp_a63*k3 + dp_a64*k4 + dp_a65*k5));

    ab_try = ab + h*(dp_b1*k1 + dp_b3*k3 + dp_b4*k4 + dp_b5*k5 + dp_b6*k6);
    [rho_try, k7] = rho_grad_hess(ab_try);

    err_vec = h*(dp_e1*k1 + dp_e3*k3 + dp_e4*k4 + dp_e5*k5 + dp_e6*k6 + dp_e7*k7);
    err     = norm(err_vec) / max(norm(ab_try - ab), 1e-10);

    err_safe = max(err, 1e-300);
    scale = 0.9 * (RKF_TOL/err_safe)^PI_K1 * (err_prev/RKF_TOL)^PI_K2;
    scale = max(0.1, min(5.0, scale));

    if err <= RKF_TOL && rho_try >= rho_val
        ab       = ab_try;
        rho_val  = rho_try;
        k1       = k7;
        err_prev = err_safe;
        h        = min(h*scale, H_MAX);

        if verb && mod(iter,25)==0
            fprintf('  iter %3d: rho=%.10f  |g|=%.2e  a=%.5g b=%.4g  h=%.2e\n', ...
                iter, rho_val, gnorm, exp(ab(1)), exp(ab(2)), h);
        end
    else
        h = max(h*scale, H_MIN);
        if h <= H_MIN
            if verb
                fprintf('cb_to_beta_hellinger_mu: h->H_MIN |g|=%.2e\n', gnorm);
            end
            break;
        end
    end
end

% Post-optimization guarantee: never return worse than best start
if rho_val < rho_best_start - 1e-10
    if verb
        fprintf('cb_to_beta_hellinger_mu: regressed; returning best start\n');
    end
    ab      = log(starts(ibest,:)');
    rho_val = rho_best_start;
end

a_opt = exp(ab(1));
b_opt = exp(ab(2));
rho   = max(0.0, min(1.0, rho_val));
H2    = max(0.0, 1.0 - rho);

if reflected
    [a_opt, b_opt] = deal(b_opt, a_opt);
end

if verb
    fprintf('cb_to_beta_hellinger_mu: a=%.6g  b=%.6g  rho=%.8f  H2=%.3e\n', ...
        a_opt, b_opt, rho, H2);
end

end  % cb_to_beta_hellinger_mu
