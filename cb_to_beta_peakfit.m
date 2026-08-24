


function [alpha, beta, info] = cb_to_beta_peakfit(chi, nu, varargin)
%CB_TO_BETA_PEAKFIT  Fast surrogate fit of Beta(a,b) to a CB posterior.
%
%   [alpha, beta, info] = cb_to_beta_peakfit(chi, nu)
%   [alpha, beta, info] = cb_to_beta_peakfit(chi, nu, 'Name', value, ...)
%
% Fits a Beta(a,b) proxy to the continuous-Bernoulli posterior
%
%     p(eta | chi, nu) propto exp( chi*eta - nu*B(eta) )
%
% by minimizing the peak of the transported CB density.  Equivalently,
% it minimizes the surrogate objective
%
%     Phi(a,b) = log max_x [ p_CB(x) / p_Beta(x; a,b) ]
%
% where p_CB(x) is the CB posterior density in x-space, up to a multiplicative
% constant independent of (a,b).  This is the cheap "peak-flattening" stage,
% not the final high-resolution Hellinger refinement.
%
% INPUTS
%   chi, nu     posterior sufficient statistics, with 0 < chi < nu
%
% NAME-VALUE OPTIONS
%   'alpha0'        initial alpha         (default: chi + 1)
%   'beta0'         initial beta          (default: nu - chi + 1)
%   'MaxIter'       outer BFGS iterations (default: 40)
%   'TolGrad'       gradient tolerance    (default: 1e-8)
%   'TolStep'       step tolerance        (default: 1e-10)
%   'EtaPad'        eta half-width pad    (default: 10)
%   'EtaCap'        minimum eta half-cap  (default: 40)
%   'Display'       true/false            (default: false)
%
% OUTPUTS
%   alpha, beta    fitted Beta parameters
%   info           diagnostic struct
%
% NOTES
%   1) This routine uses the surrogate "minimize transported peak height".
%      It does NOT yet do the final Hellinger-affinity refinement.
%   2) The CB normalizing constant is omitted from the objective because it
%      is independent of (alpha,beta); the minimizer is unchanged.
%   3) For extreme cases the optimum may lie near the boundary of parameter
%      space; this routine clips log-parameters to a broad finite box.
%
% DEPENDENCIES
%   Requires: bft_all.m, bft_b.m, cb_mode.m
%
% MIT License. Andrew Mullhaupt / OpenAI draft helper, 2026.

% -------------------- Parse inputs --------------------
p = inputParser;
p.addParameter('alpha0', [], @(x) isempty(x) || (isscalar(x) && x > 0));
p.addParameter('beta0',  [], @(x) isempty(x) || (isscalar(x) && x > 0));
p.addParameter('MaxIter', 40, @(x) isscalar(x) && x >= 1);
p.addParameter('TolGrad', 1e-8, @(x) isscalar(x) && x > 0);
p.addParameter('TolStep', 1e-10, @(x) isscalar(x) && x > 0);
p.addParameter('EtaPad', 10, @(x) isscalar(x) && x > 0);
p.addParameter('EtaCap', 40, @(x) isscalar(x) && x > 0);
p.addParameter('Display', false, @(x) isscalar(x) || islogical(x));
p.parse(varargin{:});
opts = p.Results;

% Broad but finite log-parameter box for robustness
LOG_ABS_MIN = -20;   % exp(-20) ~ 2e-9
LOG_ABS_MAX =  40;   % exp(40)  ~ 2e17

% -------------------- Validate posterior --------------------
if ~(isscalar(chi) && isscalar(nu) && isfinite(chi) && isfinite(nu))
    error('cb_to_beta_peakfit:badInput', 'chi and nu must be finite scalars.');
end
if nu <= 0 || chi <= 0 || chi >= nu
    % Degenerate or boundary posterior: return a safe fallback.
    alpha = chi + 0.5;
    beta  = nu - chi + 0.5;
    info = struct( ...
        'converged', false, ...
        'message', 'Degenerate/boundary posterior; returned Jeffreys-style fallback.', ...
        'iters', 0, ...
        'phi', NaN, ...
        'gradnorm', NaN, ...
        'eta_star', NaN, ...
        'x_star', NaN);
    return;
end

% -------------------- Initialization --------------------
if isempty(opts.alpha0)
    alpha0 = chi + 1;          % cheesy-Beta warm start
else
    alpha0 = opts.alpha0;
end
if isempty(opts.beta0)
    beta0  = (nu - chi) + 1;
else
    beta0  = opts.beta0;
end

a = min(max(log(alpha0), LOG_ABS_MIN), LOG_ABS_MAX);
b = min(max(log(beta0),  LOG_ABS_MIN), LOG_ABS_MAX);

% CB mode for eta search window
eta_cb = cb_mode(chi, nu);

% Inverse-Hessian approximation for BFGS
H = eye(2);

% Initial objective/gradient
[phi, g, eta_star, x_star] = obj_grad(a, b, chi, nu, eta_cb, opts);

if opts.Display
    fprintf('iter %2d  phi=% .6e  |g|=% .3e  a=% .4f  b=% .4f\n', ...
        0, phi, norm(g), a, b);
end

% -------------------- Outer BFGS loop --------------------
converged = false;
msg = 'Maximum iterations reached.';

for iter = 1:opts.MaxIter
    gnorm = norm(g);
    if gnorm <= opts.TolGrad
        converged = true;
        msg = 'Gradient tolerance satisfied.';
        break;
    end

    % BFGS search direction
    pdir = -H * g;
    if ~(all(isfinite(pdir)) && dot(g, pdir) < 0)
        pdir = -g;  % fallback to steepest descent
    end

    % Backtracking Armijo line search
    t = 1.0;
    c1 = 1e-4;
    phi0 = phi;
    gTp = dot(g, pdir);

    accepted = false;
    for ls = 1:25
        a_try = min(max(a + t * pdir(1), LOG_ABS_MIN), LOG_ABS_MAX);
        b_try = min(max(b + t * pdir(2), LOG_ABS_MIN), LOG_ABS_MAX);

        [phi_try, g_try, eta_try, x_try] = obj_grad(a_try, b_try, chi, nu, eta_cb, opts);

        if isfinite(phi_try) && (phi_try <= phi0 + c1 * t * gTp)
            accepted = true;
            break;
        end

        t = 0.5 * t;
        if t < opts.TolStep
            break;
        end
    end

    if ~accepted
        msg = 'Line search failed.';
        break;
    end

    s = [a_try - a; b_try - b];
    y = g_try - g;

    % BFGS update (inverse Hessian form), safeguarded
    ys = dot(y, s);
    if isfinite(ys) && ys > 1e-12
        rho = 1 / ys;
        I2 = eye(2);
        V  = I2 - rho * (s * y.');
        H  = V * H * V.' + rho * (s * s.');
    else
        H = eye(2);  % reset if update is unreliable
    end

    % Accept step
    a = a_try
    b = b_try
    phi = phi_try;
    g = g_try;
    eta_star = eta_try;
    x_star = x_try;

    if opts.Display
        fprintf('iter %2d  phi=% .6e  |g|=% .3e  step=% .3e  a=% .4f  b=% .4f\n', ...
            iter, phi, norm(g), norm(s), a, b);
    end

    if norm(s) <= opts.TolStep * (1 + norm([a; b]))
        converged = true;
        msg = 'Step tolerance satisfied.';
        break;
    end
end

alpha = exp(a);
beta  = exp(b);

info = struct();
info.converged = converged;
info.message   = msg;
info.iters     = iter - (~converged && iter == opts.MaxIter);
info.phi       = phi;
info.peak      = exp(phi);
info.grad      = g;
info.gradnorm  = norm(g);
info.log_alpha = a;
info.log_beta  = b;
info.eta_star  = eta_star;
info.x_star    = x_star;

end


% ========================================================================
function [phi, g_ab, eta_star, x_star] = obj_grad(a, b, chi, nu, eta_cb, opts)
% Objective and gradient in log-parameters:
%   phi(a,b) = log max_x p_CB(x)/p_Beta(x)
%
% The CB normalizing constant is omitted because it is independent of (a,b).

alpha = exp(a);
beta  = exp(b);

% Search window in eta-space
eta_ctr = 0.5 * (eta_cb + safe_logit(alpha / (alpha + beta)));
eta_hw  = max([opts.EtaCap, abs(eta_cb) + opts.EtaPad, abs(eta_ctr) + opts.EtaPad]);
eta_lo  = -eta_hw;
eta_hi  =  eta_hw;

% 1D maximization by Brent/fminbnd on -L(eta)
negL = @(eta) -log_ratio_eta(eta, chi, nu, alpha, beta);
[eta_star, fval] = fminbnd(negL, eta_lo, eta_hi);
phi = -fval;

x_star = sigmoid_stable(eta_star);
logx   = log_sigmoid(eta_star);
log1mx = log1m_sigmoid(eta_star);

% Envelope-theorem gradient in (alpha,beta):
%   dPhi/dalpha = psi(alpha) - psi(alpha+beta) - log x*
%   dPhi/dbeta  = psi(beta)  - psi(alpha+beta) - log(1-x*)
%
% Use stable differences when parameters are highly imbalanced.
dPhi_dalpha = psi_diff_stable(alpha, beta) - logx;
dPhi_dbeta  = psi_diff_stable(beta, alpha) - log1mx;

% Chain rule to log-parameters (a = log alpha, b = log beta)
g_ab = [alpha * dPhi_dalpha;
        beta  * dPhi_dbeta];

% Defensive cleanup
if ~isfinite(phi), phi = Inf; end
g_ab(~isfinite(g_ab)) = 0;

end


% ========================================================================
function val = log_ratio_eta(eta, chi, nu, alpha, beta)
% Unnormalized log p_CB(x(eta)) - log p_Beta(x(eta); alpha,beta)
%
% Since the CB normalizing constant is independent of (alpha,beta), omitting
% it does not change the minimizer of the peak objective.

logx   = log_sigmoid(eta);
log1mx = log1m_sigmoid(eta);

% q_CB(x) in x-space (up to eta-normalization constant):
%   q_CB(x) propto exp(chi*eta - nu*B(eta)) / (x(1-x))
% so log q_CB = chi*eta - nu*B(eta) - log x - log(1-x)
% and log p_Beta = (alpha-1)log x + (beta-1)log(1-x) - betaln(alpha,beta)
%
% Subtracting gives:
%   chi*eta - nu*B(eta) + betaln(alpha,beta) - alpha*log x - beta*log(1-x)

val = chi * eta - nu * bft_b(eta) + betaln(alpha, beta) ...
      - alpha * logx - beta * log1mx;
end


% ========================================================================
function y = sigmoid_stable(eta)
% Numerically stable sigmoid
if eta >= 0
    e = exp(-eta);
    y = 1 / (1 + e);
else
    e = exp(eta);
    y = e / (1 + e);
end
end


% ========================================================================
function y = log_sigmoid(eta)
% log(sigmoid(eta)) stably
if eta >= 0
    y = -log1p(exp(-eta));
else
    y = eta - log1p(exp(eta));
end
end


% ========================================================================
function y = log1m_sigmoid(eta)
% log(1 - sigmoid(eta)) stably
if eta >= 0
    y = -eta - log1p(exp(-eta));
else
    y = -log1p(exp(eta));
end
end


% ========================================================================
function y = safe_logit(x)
% Stable logit for x in (0,1)
x = min(max(x, realmin), 1 - eps);
y = log(x / (1 - x));
end


% ========================================================================
function d = psi_diff_stable(x, a)
% Stable computation of psi(x) - psi(x+a),  with x>0, a>0.
%
% Direct subtraction is fine in the ordinary regime.  When x is much larger
% than a, use an asymptotic difference expansion to avoid catastrophic
% cancellation.
%
% psi(t) = log t - 1/(2t) - 1/(12 t^2) + 1/(120 t^4) - ...
% hence
% psi(x) - psi(x+a)
%   = -log(1+a/x)
%     + 1/(2(x+a)) - 1/(2x)
%     + 1/(12(x+a)^2) - 1/(12x^2)
%     - 1/(120(x+a)^4) + 1/(120x^4) + ...

if x > 1e6 || x > 1e4 * a
    xa = x + a;
    d = -log1p(a / x) ...
        + 0.5 * (1 / xa - 1 / x) ...
        + (1/12) * (1 / xa^2 - 1 / x^2) ...
        - (1/120) * (1 / xa^4 - 1 / x^4);
else
    d = psi(x) - psi(x + a);
end
end
