function [a_opt, b_opt, rho, H2] = cb_to_beta_hellinger(chi, nu, varargin)
% CB_TO_BETA_HELLINGER  Beta(a,b) maximizing Hellinger affinity with CB posterior.
%
%   [a_opt, b_opt, rho, H2] = cb_to_beta_hellinger(chi, nu)
%   [a_opt, b_opt, rho, H2] = cb_to_beta_hellinger(chi, nu, 'N_grid', 2000, ...)
%
%   Finds Beta(a_opt, b_opt) maximizing the Hellinger affinity
%
%       rho = integral_0^1 sqrt(p_CB(theta) * p_Beta(theta)) dtheta  in [0,1]
%
%   and returns H2 = 1 - rho (Hellinger distance squared, in [0,1]).
%
%   FORMULA
%   -------
%       rho = I_total / sqrt(Z_eta * B(alpha, beta))
%   =>  log rho = log(I_total) - (1/2)*log(Z_eta) - (1/2)*betaln(alpha, beta)
%
%   where I_total = integral exp((chi*eta - nu*B(eta))/2)
%                             * sig^(alpha/2) * (1-sig)^(beta/2) deta
%   and Z_eta = integral exp(chi*eta - nu*B(eta)) deta.
%
%   GRADIENT (exact on fixed grid, envelope theorem)
%   -------------------------------------------------
%   With probability weights pi_j = g_j / S0 and weighted sums
%   Sk = sum(g .* (log_sig).^k) * deta,  S02 = sum(g .* log_sig .* log_1msig) * deta:
%
%       d(log rho)/d(log alpha) = alpha/2 * (S1/S0 + psi(alpha+beta) - psi(alpha))
%       d(log rho)/d(log beta)  = beta/2  * (S2/S0 + psi_diff_stable(alpha, beta))
%
%   HESSIAN (of log rho, in (log alpha, log beta) coordinates)
%   ----------------------------------------------------------
%   Let V11 = S11/S0 - (S1/S0)^2  = Var_pi[log sigma]
%       V22 = S22/S0 - (S2/S0)^2  = Var_pi[log(1-sigma)]
%       V12 = S12/S0 - (S1/S0)*(S2/S0) = Cov_pi[log sigma, log(1-sigma)]
%   where S11 = sum(g .* log_sig.^2)*deta, etc.
%
%       H_aa = ga + alpha^2/4 * V11 + alpha^2/2 * (psi1(alpha+beta) - psi1(alpha))
%       H_bb = gb + beta^2/4  * V22 + beta^2/2  * (psi1(alpha+beta) - psi1(beta))
%       H_ab = alpha*beta/4   * V12 + alpha*beta/2 * psi1(alpha+beta)
%
%   where ga, gb are the current gradient components and psi1 = trigamma.
%   (The ga,gb terms vanish at the optimum but are included for correctness
%   away from it.)
%
%   OPTIMIZER: DORMAND-PRINCE RKF45 (DOPRI5)
%   ------------------------------------------
%   Solves the gradient-ascent ODE d(theta)/dt = grad(log rho)(theta)
%   using the Dormand-Prince explicit RK 4(5) method with adaptive step size.
%   (Same method as MATLAB's ode45.)
%
%   Requires only gradient evaluations (no Hessian in the step itself).
%   6 gradient evaluations per accepted step (FSAL: k7 reused as k1).
%
%   Step size adapts automatically via local error estimate:
%       h_new = h * min(5, max(0.1, 0.9*(RKF_TOL/err)^(1/5)))
%   Large steps when landscape is flat (small error); small steps near
%   optima or curved regions.
%
%   STIFFNESS WARNING: this is an explicit method.  If the Hessian has
%   large negative eigenvalues (|lambda| >> 1), the method will take
%   arbitrarily tiny steps without making progress (classic stiffness
%   symptom).  For our problem, Hessian eigenvalues are O(1) or smaller,
%   so this is expected to be safe.  If h collapses below H_MIN in a
%   non-converged state, the problem is stiff; switch to gl4_step fallback.
%
%   ALTERNATIVE (stiff/uncertain cases): the gl4_step function below
%   implements the 2-stage A-stable Gauss-Legendre RK4.  It is commented
%   out in the main loop but retained for reference.
%
%   NUMERICAL STABILITY: THREE CANCELLATION HAZARDS FIXED
%   -------------------------------------------------------
%   1. betaln(a,b) for large max(a,b): gammaln subtraction loses all precision.
%      Fix: betaln_stable uses gammaln(a) - [a*log(b) + a*(a-1)/(2b)].
%
%   2. psi(a+b) - psi(b) for large b >> a: both ~log(b), difference ~a/b.
%      Fix: psi_diff_stable uses series a/b - a^2/(2b^2) + ...
%      THIS CAUSED A SILENT ZERO GRADIENT -- hardest to diagnose.
%
%   3. psi1(a+b) - psi1(b) for large b: both ~1/b, difference ~a/b^2.
%      Fix: psi1_diff_stable uses series a/b^2 - a^2/b^3 + ...
%      Needed in Hessian H_bb and H_ab for extreme-Z cases.
%
%   INITIALIZATION: four starting points, best selected
%     1. Cheesy Beta      (alpha=chi+1, beta=nu-chi+1)
%     2. Boundary-matched (alpha=chi, beta=nu-chi)
%     3. Mode-matched low-Z  (alpha=chi, beta=chi/theta_star)
%     4. Mode-matched high-Z (alpha=(nu-chi)/theta_hi, beta=nu-chi)
%
%   INPUTS
%     chi    sum of observed rewards (>= 0)
%     nu     pull count (> 0)
%
%   OPTIONS
%     'N_grid'   eta-grid points (default 2000)
%     'max_iter' max steps (default 200)
%     'tol'      gradient-norm tolerance (default 1e-9)
%     'verbose'  print trace (default false)
%
%   OUTPUTS
%     a_opt, b_opt   Hellinger-optimal Beta parameters
%     rho            Hellinger affinity in [0,1]
%     H2             1 - rho
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
% By the CB symmetry identity p(theta|chi,nu) = p(1-theta|nu-chi,nu)
% and the Beta reflection Beta(a,b)(1-theta) = Beta(b,a)(theta):
%   H^2(CB(chi,nu), Beta(a,b)) = H^2(CB(nu-chi,nu), Beta(b,a))
% So for Z > 0.5 we solve the reflected problem and swap (a,b) on return.
% This matches the sampler's sign-bit reflection: near theta=1, IEEE 754
% doubles have only eps/2 ~ 1e-16 resolution, while near theta=0 the
% full subnormal range is available.  The grid is always built in the
% low-Z (well-resolved) half.
reflected = (chi / nu > 0.5);
if reflected
    chi = nu - chi;   % now Z_eff = 1 - Z_orig < 0.5
end

%% Stable special functions

ASYM = 1e8;   % threshold for asymptotic expansions

    function bl = betaln_stable(a, b)
    % log B(a,b): avoids gammaln subtraction cancellation for large max(a,b).
    % gammaln(b+a) - gammaln(b) = a*log(b) + a*(a-1)/(2b) + O(1/b^2)
        if b > ASYM && b > a
            bl = gammaln(a) - (a*log(b) + a*(a-1)/(2*b));
        elseif a > ASYM && a > b
            bl = gammaln(b) - (b*log(a) + b*(b-1)/(2*a));
        else
            bl = betaln(a, b);
        end
    end

    function ps = psi_stable(x)
    % digamma for large x: psi(x) ~ log(x) - 1/(2x) - 1/(12x^2)
        if x > ASYM
            ps = log(x) - 1/(2*x) - 1/(12*x^2);
        else
            ps = psi(x);
        end
    end

    function d = psi_diff_stable(a, b)
    % psi(a+b) - psi(b): direct series for b >> a avoids cancellation.
    % psi(b+a) - psi(b) = a/b - a^2/(2b^2) + a^3/(6b^3) - ...
    % The naive subtraction loses ~log(b)/eps ~ 1e12 digits for b~1e14.
        if b > ASYM && b > 10*a
            t = a/b;
            d = t*(1 - t/2 + t^2/6 - t^3/12 + t^4/20 - t^5/30);
        else
            d = psi(a + b) - psi(b);
        end
    end

    function d = psi1_diff_stable(a, b)
    % psi1(a+b) - psi1(b): direct series for b >> a.
    % psi1(b+a) - psi1(b) = -a/b^2 + a^2/b^3 - ... = -(a/b^2)*(1 - a/b + ...)
    % Needed in Hessian H_bb and H_ab for extreme-Z cases.
        if b > ASYM && b > 10*a
            t = a/b;
            d = -(t/b)*(1 - t + t^2 - t^3 + t^4);
        else
            d = psi(1, a + b) - psi(1, b);
        end
    end

%% Build fixed eta-grid
eta_star = cb_mode(chi, nu);
[~, ~, bpp] = bft_all(eta_star);
sig_eta = 1.0 / sqrt(max(nu * bpp, 1e-6));

f_peak = chi * eta_star - nu * bft_b(eta_star);
LOG_THR = 35;
step = max(sig_eta, 0.5);

eL = eta_star;
while chi*eL - nu*bft_b(eL) > f_peak - LOG_THR, eL = eL - step; end
eR = eta_star;
while chi*eR - nu*bft_b(eR) > f_peak - LOG_THR, eR = eR + step; end
eL = eL - step;  eR = eR + step;

eta  = linspace(eL, eR, N)';
deta = eta(2) - eta(1);

log_sig   = -log1p(exp(-eta));
log_1msig = -log1p(exp(eta));

log_f0 = (chi*eta - nu*bft_b(eta)) / 2;
log_f0 = log_f0 - max(log_f0);
f0     = exp(log_f0);
Z_f0   = sum(f0.^2) * deta;
log_Z_f0 = log(Z_f0);

%% Core: rho, gradient, and Hessian at (a,b) = (log alpha, log beta)
    function [rv, gv, Hv] = rho_grad_hess(ab)
        alpha = exp(ab(1));
        beta  = exp(ab(2));

        % Integrand weights
        log_g  = log_f0 + (alpha/2)*log_sig + (beta/2)*log_1msig;
        lmax_g = max(log_g);
        g      = exp(log_g - lmax_g);

        % Weighted sums
        S0  = sum(g)                         * deta;
        S1  = sum(g .* log_sig)              * deta;
        S2  = sum(g .* log_1msig)            * deta;
        S11 = sum(g .* log_sig.^2)           * deta;
        S22 = sum(g .* log_1msig.^2)         * deta;
        S12 = sum(g .* log_sig .* log_1msig) * deta;

        % log rho
        log_rv = lmax_g + log(S0) - 0.5*log_Z_f0 - 0.5*betaln_stable(alpha, beta);
        rv = exp(log_rv);

        if nargout < 2, return; end

        % Gradient in (log alpha, log beta)
        diff_a = psi_stable(alpha+beta) - psi(alpha);    % no cancellation
        diff_b = psi_diff_stable(alpha, beta);            % stable series

        ga = alpha * rv/2 * (S1/S0 + diff_a);
        gb = beta  * rv/2 * (S2/S0 + diff_b);
        gv = [ga; gb];

        if nargout < 3, return; end

        % Hessian of log rho in (log alpha, log beta)
        % Conditional variance/covariance of log_sig, log_1msig under weights pi
        V11 = S11/S0 - (S1/S0)^2;
        V22 = S22/S0 - (S2/S0)^2;
        V12 = S12/S0 - (S1/S0)*(S2/S0);

        % Trigamma terms — need stable differences for large beta
        p1_a   = psi(1, alpha);                      % psi1(alpha)
        p1_b   = psi(1, beta);                       % psi1(beta)
        % psi1(alpha+beta) - psi1(alpha): no cancellation (both small if large)
        % psi1(alpha+beta) - psi1(beta): needs stable version for large beta
        p1_ab  = psi(1, alpha + beta);               % psi1(alpha+beta)
        dp1_b  = psi1_diff_stable(alpha, beta);      % psi1(a+b) - psi1(b), stable

        H_aa = ga + alpha^2/4 * V11 + alpha^2/2 * (p1_ab - p1_a);
        H_bb = gb + beta^2/4  * V22 + beta^2/2  * dp1_b;
        H_ab = alpha*beta/4   * V12 + alpha*beta/2 * p1_ab;
        Hv   = [H_aa, H_ab; H_ab, H_bb];
    end

%% Initialization: five starting points, best selected
% Starting points cover: cheesy, boundary-matched, mode-matched (low-Z and
% high-Z), and moment-matched (near-optimal for narrow posteriors / CF regime).
theta_star = 1 / (1 + exp(-eta_star));
theta_lo   = max(theta_star, 1e-300);
theta_hi   = min(1 - theta_star, 1 - 1e-300);
alpha_bm   = max(chi, 0.5);
beta_bm    = max(nu - chi, 0.5);

% Start 5: moment-matched Beta (cb_to_beta_num).
% For narrow posteriors (CF regime, large nu) this is already near-optimal
% and gives the DOPRI5 integrator a much better starting point than any of
% the geometric heuristics.  This prevents H2_h > H2_mm artifacts.
try
    [a_mm0, b_mm0] = cb_to_beta_num(chi, nu);
    if ~isfinite(a_mm0) || ~isfinite(b_mm0) || a_mm0 <= 0 || b_mm0 <= 0
        a_mm0 = chi + 1;  b_mm0 = nu - chi + 1;
    end
catch
    a_mm0 = chi + 1;  b_mm0 = nu - chi + 1;
end

starts = [chi + 1,                    nu - chi + 1;   % cheesy
           alpha_bm,                   beta_bm;         % boundary-matched
           alpha_bm,                   max(alpha_bm/theta_lo, 0.5);  % mode low-Z
           max(beta_bm/theta_hi, 0.5), beta_bm;         % mode high-Z
           a_mm0,                      b_mm0];           % moment-matched

rho_s = zeros(size(starts,1), 1);
for s = 1:size(starts,1)
    rho_s(s) = rho_grad_hess(log(starts(s,:)'));
end
[rho_best_start, ibest] = max(rho_s);
ab = log(starts(ibest,:)');
[rho_val, grad] = rho_grad_hess(ab);

if verb
    fprintf('cb_to_beta_hellinger: start %d (a=%.4g b=%.4g) rho=%.6f\n', ...
        ibest, exp(ab(1)), exp(ab(2)), rho_val);
end

%% DOPRI5 optimizer (Dormand-Prince RKF45)
% Explicit adaptive-step gradient flow integrator.
% No Hessian needed per step; FSAL: k7 of accepted step = k1 of next step.
%
% STIFFNESS SYMPTOM: if h collapses to H_MIN without convergence, the
% Hessian has large eigenvalues.  Uncomment the gl4_step fallback below.

% Dormand-Prince tableau
dp_a21 = 1/5;
dp_a31 = 3/40;       dp_a32 = 9/40;
dp_a41 = 44/45;      dp_a42 = -56/15;     dp_a43 = 32/9;
dp_a51 = 19372/6561; dp_a52 = -25360/2187; dp_a53 = 64448/6561; dp_a54 = -212/729;
dp_a61 = 9017/3168;  dp_a62 = -355/33;    dp_a63 = 46732/5247;
dp_a64 = 49/176;     dp_a65 = -5103/18656;
% 5th-order weights (propagated solution)
dp_b1 = 35/384;   dp_b3 = 500/1113; dp_b4 = 125/192;
dp_b5 = -2187/6784; dp_b6 = 11/84;
% Error coefficients (difference of 4th and 5th order weights)
dp_e1 =  71/57600;    dp_e3 = -71/16695;  dp_e4 =  71/1920;
dp_e5 = -17253/339200; dp_e6 = 22/525;    dp_e7 = -1/40;

RKF_TOL = max(TOL * 100, 1e-7);   % step acceptance tolerance
H_MAX   = 4.0;
H_MIN   = 1e-13;
h       = 0.5;                     % initial step

% Söderlind PI controller parameters (MODES / ACM TOMS 2003).
% Replaces the simple proportional (tol/err)^(1/p) controller with:
%   scale = (tol/err_n)^(k1/p) * (err_{n-1}/tol)^(k2/p)
% where p=5 (DOPRI5 order), k1=0.7, k2=0.4.
% This damps the "bang-bang" oscillation of the pure I-controller.
% Reference: Söderlind (2003), "Digital Filters in Adaptive Time-Stepping",
%            ACM TOMS 29(1):1-26.  Arévalo & Söderlind (2017), MODES package.
PI_K1 = 0.7 / 5;   % 0.14  -- current error exponent
PI_K2 = 0.4 / 5;   % 0.08  -- previous error exponent (damping term)
err_prev = RKF_TOL; % neutral initialisation: no history bias on first step

% FSAL: k1 = grad at current point (already computed)
k1 = grad;

for iter = 1:MAX_ITER
    gnorm = norm(k1);
    if gnorm < TOL
        if verb
            fprintf('cb_to_beta_hellinger: converged iter=%d |g|=%.2e rho=%.10f\n', ...
                iter, gnorm, rho_val);
        end
        break;
    end

    % Compute the 6 stages (gradient evaluations only)
    [~, k2] = rho_grad_hess(ab + h*dp_a21*k1);
    [~, k3] = rho_grad_hess(ab + h*(dp_a31*k1 + dp_a32*k2));
    [~, k4] = rho_grad_hess(ab + h*(dp_a41*k1 + dp_a42*k2 + dp_a43*k3));
    [~, k5] = rho_grad_hess(ab + h*(dp_a51*k1 + dp_a52*k2 + dp_a53*k3 + dp_a54*k4));
    [~, k6] = rho_grad_hess(ab + h*(dp_a61*k1 + dp_a62*k2 + dp_a63*k3 + dp_a64*k4 + dp_a65*k5));

    % 5th-order step
    ab_try = ab + h*(dp_b1*k1 + dp_b3*k3 + dp_b4*k4 + dp_b5*k5 + dp_b6*k6);
    [rho_try, k7] = rho_grad_hess(ab_try);   % k7 = grad at new point (FSAL)

    % Local error estimate (4th vs 5th order)
    err_vec = h*(dp_e1*k1 + dp_e3*k3 + dp_e4*k4 + dp_e5*k5 + dp_e6*k6 + dp_e7*k7);
    err = norm(err_vec) / max(norm(ab_try - ab), 1e-10);

    % Step acceptance — Söderlind PI controller
    % scale = safety * (tol/err_n)^(k1/p) * (err_{n-1}/tol)^(k2/p)
    err_safe = max(err, 1e-300);
    scale = 0.9 * (RKF_TOL / err_safe)^PI_K1 * (err_prev / RKF_TOL)^PI_K2;
    scale = max(0.1, min(5.0, scale));

    if err <= RKF_TOL && rho_try >= rho_val   % accept
        ab       = ab_try;
        rho_val  = rho_try;
        k1       = k7;           % FSAL: reuse k7 as k1 for next step
        err_prev = err_safe;     % update history for PI controller
        h = min(h * scale, H_MAX);

        if verb && mod(iter,25)==0
            fprintf('  iter %3d: rho=%.10f  |g|=%.2e  a=%.5g b=%.4g  h=%.2e\n', ...
                iter, rho_val, gnorm, exp(ab(1)), exp(ab(2)), h);
        end
    else                                       % reject
        h = max(h * scale, H_MIN);
        if h <= H_MIN
            if gnorm < TOL * 1000
                if verb
                    fprintf('cb_to_beta_hellinger: h->H_MIN, near-converged |g|=%.2e\n', gnorm);
                end
            else
                if verb
                    fprintf('cb_to_beta_hellinger: h->H_MIN, possible stiffness |g|=%.2e\n', gnorm);
                    fprintf('  Consider gl4_step fallback for this case.\n');
                end
            end
            break;
        end
    end
end

% Post-optimization guarantee: the optimizer must not return something
% worse than the best starting point.  If it does (e.g. early stall),
% fall back to the best starting point found during initialization.
% This makes the function monotone: rho_output >= rho at every starting point.
if rho_val < rho_best_start - 1e-10
    if verb
        fprintf('cb_to_beta_hellinger: optimizer regressed (rho=%.8f < start=%.8f); using best start\n', ...
            rho_val, rho_best_start);
    end
    ab      = log(starts(ibest,:)');
    rho_val = rho_best_start;
end

a_opt = exp(ab(1));
b_opt = exp(ab(2));
rho   = max(0.0, min(1.0, rho_val));
H2    = max(0.0, 1.0 - rho);

% Un-reflect: for Z > 0.5 we solved Beta(b,a) for the reflected CB,
% so the answer for the original CB(chi_orig, nu) is Beta(b,a).
if reflected
    [a_opt, b_opt] = deal(b_opt, a_opt);
end

if verb
    fprintf('cb_to_beta_hellinger: a=%.6g  b=%.6g  rho=%.8f  H2=%.3e\n', ...
        a_opt, b_opt, rho, H2);
end
end  % main function


function [d, ok] = gl4_step(h, grad, H, a11, a12, a21, a22)
% GL4_STEP  One linearized 2-stage Gauss-Legendre RK4 step.
%
%   Solves the 4x4 stage system:
%     [I - h*a11*H,  -h*a12*H ] [K1]   [grad]
%     [-h*a21*H,  I - h*a22*H] [K2] = [grad]
%
%   Returns d = h*(K1+K2)/2  (the parameter update, since b1=b2=1/2).
%   ok = false if system is near-singular (caller falls back to gradient).
%
%   The 4x4 system is assembled from four 2x2 blocks and solved with MATLAB's
%   backslash.  For d=2 (our case) this is a trivial 4x4 real solve.

I2 = eye(2);
M  = [I2 - h*a11*H,  -h*a12*H; ...
         -h*a21*H,   I2 - h*a22*H];

% Condition check: if M is degenerate, fall back
cond_est = rcond(M);
if cond_est < 1e-14
    d  = h * grad;
    ok = false;
    return;
end

rhs = [grad; grad];
KK  = M \ rhs;
K1  = KK(1:2);
K2  = KK(3:4);
d   = h * (K1 + K2) / 2;   % b1=b2=1/2
ok  = true;
end  % gl4_step
