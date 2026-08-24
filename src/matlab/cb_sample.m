function theta = cb_sample(chi, nu, n)
% CB_SAMPLE  Draw n iid samples from the CB conjugate posterior.
%
%   theta = cb_sample(chi, nu, n)
%
%   Algorithm triage:
%
%   0. POINT:  chi=0 or chi=nu with nu>0 returns delta_0 or delta_1
%
%   1. GAMMA:  Z_bar < 1/(2*log(nu)) or (1-Z_bar) < 1/(2*log(nu))
%      tau ~ Gamma(nu+1, chi), theta = exp(-tau)
%
%   2. CF:     sigma = 1/sqrt(nu*B''(eta*)) < 0.20
%      Cornish-Fisher corrected normal.
%      Threshold 0.20 (tighter than initial 0.25): validated at N=50000.
%
%   3. ARS:    otherwise
%      Adaptive rejection sampling with Gilks-Wild chord squeeze.
%
%   See also: cb_sample_gamma, cb_sample_laplace_cf, cb_mode, cb_ars

if nargin < 3, n = 1; end

if ~isfinite(chi) || ~isfinite(nu) || nu < 0 || chi < 0 || chi > nu
    error('cb_sample:badStats', ...
        'Require finite sufficient statistics 0 <= chi <= nu.');
end

if nu <= 0
    theta = rand(n, 1);
    return
end

% The boundary density on finite eta is improper, but the closed posterior
% family has the weak limits delta_0 and delta_1.  Return those point masses
% directly rather than presenting a boundary target to ARS.
if chi == 0
    theta = zeros(n, 1);
    return
elseif chi == nu
    theta = ones(n, 1);
    return
end

% --- Regime 1: Gamma ---
% Only use Gamma for nu >= 50: below that, the approximation error is
% detectable and the throughput advantage over ARS is negligible (~1.17x).
% At small nu, arms are selected infrequently and the ARS hull rebuild
% cost is fully amortized.  See STATUS.md, session 2026-03-21.
NU_MIN_GAMMA = 50.0;
NU_MIN_CF    = 50.0;

zbar      = chi / nu;
log_nu    = log(max(nu, 2));
gam_thresh = 1.0 / (2.0 * log_nu);

if nu >= NU_MIN_GAMMA && (zbar < gam_thresh || (1.0 - zbar) < gam_thresh)
    zmin = min(zbar, 1-zbar);
    if zmin * sqrt(nu) < 0.18       % width guard
        % Precision guard for high-Z: 1-exp(-tau) clips to 1.0 when
        % mean_tau = (nu+1)/(nu-chi) > 3.67 (P(clip) > 0.01%).
        if zbar > 0.5 && (nu+1)/(nu-chi) > 3.67
            % fall through to ARS
        else
            theta = cb_sample_gamma(chi, nu, n);
            return
        end
    end
end

% --- Regime 2: Cornish-Fisher ---
% Only use CF for nu >= 50, same rationale as Gamma.
SIGMA_CF  = 0.20;

eta_star  = cb_mode(chi, nu);
[~, ~, bpp_star] = bft_all(eta_star);
sigma     = 1.0 / sqrt(nu * bpp_star);

if nu >= NU_MIN_CF && sigma < SIGMA_CF
    theta = cb_sample_laplace_cf(chi, nu, n);
    return
end

% --- Regime 3: ARS ---
% Always sample the low-Z parameterisation and reflect if needed.
% Exact identity: p(theta|chi,nu) = p(1-theta|nu-chi,nu).
% Sign-bit convention: negative phi encodes high-Z draw; theta = 1 + phi.
reflect  = (chi / nu > 0.5);
chi_eff  = reflect * (nu - chi) + (~reflect) * chi;  % nu-chi or chi, no branch
eta_eff  = cb_mode(chi_eff, nu);
eta_samp = cb_ars(chi_eff, nu, eta_eff, n);
phi      = 1.0 ./ (1.0 + exp(-eta_samp));   % always in (0,1), near 0
if reflect
    theta = 1.0 - phi;   % reflect: near 1, best precision available
else
    theta = phi;
end
end
