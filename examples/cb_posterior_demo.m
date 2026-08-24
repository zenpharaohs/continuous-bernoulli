function [H2_bb, H2_dl, H2_hmu] = cb_posterior_demo(chi, nu, varargin)
% CB_POSTERIOR_DEMO  CB posterior vs three Beta strategies (theta and mu space).
%
%   cb_posterior_demo(chi, nu)
%   [H2_bb, H2_dl, H2_hmu] = cb_posterior_demo(chi, nu, ...)
%
%   Three Beta strategies, in order of increasing accuracy:
%
%     (1) Bernoulli-Beta  Beta(chi+1, nu-chi+1)
%         Correct conjugate update for Bernoulli(theta) data from a flat prior:
%         a <- a + Z(t), b <- b + (1-Z(t)).  Works perfectly for Bernoulli
%         rewards; the issue is the likelihood -- CB rewards are not Bernoulli.
%         Reference: Loaiza-Ganem & Cunningham (2018, NeurIPS).
%
%     (2) Data-level MM   cb_to_beta_data_mm
%         Matches E[X] = B'(eta*) and Var[X] = B''(eta*) under the CB data
%         distribution.  O(1), no quadrature.
%
%     (3) mu-Hellinger-optimal  cb_to_beta_hellinger_mu
%         Minimises H^2 between the CB oracle's Thompson sampling loss proxy
%         distribution and Beta(a,b) used directly as a loss proxy.  This is
%         the best Beta for Thompson sampling use: it concentrates near the
%         true expected loss mu with the same spread as the CB oracle.
%
%   H^2 values shown in the legend are with respect to the CB posterior in
%   theta-space ('theta' plot) or with respect to the CB push-forward loss
%   proxy distribution ('mu' plot).
%
%   TWO VIEWS (xaxis option)
%   -----------------------------------------------------------------------
%   'theta' (default): posterior densities over the CB arm parameter theta.
%     H^2 here measures approximation to the CB posterior.
%     The mu-Hellinger optimal Beta concentrates near theta* but has small
%     H^2_mu -- the two objectives are different.
%
%   'mu': Thompson sampling LOSS PROXY distributions.
%     X-axis = the selection statistic each strategy actually samples.
%     All Beta curves: p ~ Beta(a,b) used directly as loss proxy.
%     CB-exact oracle: push-forward of CB posterior through theta->mu=B'(logit(theta)).
%     H^2_mu values shown: these measure quality as loss proxy distributions.
%     This is the view that matters for Thompson sampling regret.
%
%   INPUTS
%     chi    sufficient statistic: sum of observed losses  (0 < chi < nu)
%     nu     sufficient statistic: number of observations  (nu > 0)
%
%   OPTIONS (name-value)
%     'N_theta'    plot grid size (default 800)
%     'fig'        figure handle (default: new figure)
%     'verbose'    show optimizer trace (default false)
%     'max_iter'   max DOPRI5 iterations (default 200)
%     'xaxis'      'theta' (default) or 'mu'
%
%   OUTPUTS (optional)
%     H2_bb       H^2(CB push-forward, Bernoulli-Beta)    [in mu-space]
%     H2_dl       H^2(CB push-forward, data-level MM Beta) [in mu-space]
%     H2_hmu      H^2(CB push-forward, mu-Hellinger-opt)   [in mu-space]
%
%   EXAMPLES
%     cb_posterior_demo(6, 20)                    % mu=0.30, nu=20
%     cb_posterior_demo(1, 10)                    % mu=0.10, nu=10
%     cb_posterior_demo(9, 10)                    % mu=0.90, nu=10
%     cb_posterior_demo(10, 20)                   % mu=0.50, nu=20
%     cb_posterior_demo(15, 50)                   % mu=0.30, nu=50
%     cb_posterior_demo(15, 50, 'xaxis', 'mu')    % Thompson proxy distributions
%
% MIT License.  Andrew Mullhaupt, Stony Brook University AMS/QF, 2026.

%% Options
p = inputParser;
addRequired(p,  'chi');
addRequired(p,  'nu');
addParameter(p, 'N_theta',  800,    @(x) isscalar(x) && x >= 50);
addParameter(p, 'fig',      [],     @(x) isempty(x) || isgraphics(x));
addParameter(p, 'verbose',  false,  @(x) islogical(x) || isscalar(x));
addParameter(p, 'max_iter', 200,    @(x) isscalar(x) && x >= 1);
addParameter(p, 'xaxis',   'theta', @(x) ischar(x) && any(strcmp(x,{'theta','mu'})));
parse(p, chi, nu, varargin{:});
N_th  = p.Results.N_theta;
hfig  = p.Results.fig;
verb  = logical(p.Results.verbose);
MXIT  = p.Results.max_iter;
xaxis = p.Results.xaxis;

%% Guard
if nu <= 0 || chi <= 0 || chi >= nu
    warning('cb_posterior_demo: degenerate (chi=%.4g, nu=%.4g)', chi, nu);
    [H2_bb, H2_dl, H2_hmu] = deal(NaN, NaN, NaN);
    return;
end

%% -----------------------------------------------------------------------
%  PARAMETERIZATION QUANTITIES
% -----------------------------------------------------------------------
mu         = chi / nu;
eta_star   = cb_mode(chi, nu);
[~, ~, bpp] = bft_all(eta_star);
theta_star = 1 / (1 + exp(-eta_star));

%% -----------------------------------------------------------------------
%  ETA GRID FOR CB POSTERIOR
% -----------------------------------------------------------------------
sig_eta = 1 / sqrt(max(nu * bpp, 1e-6));
f_peak  = chi * eta_star - nu * bft_b(eta_star);
step    = max(sig_eta, 0.5);

eL = eta_star;
while chi*eL - nu*bft_b(eL) > f_peak - 40,  eL = eL - step;  end
eR = eta_star;
while chi*eR - nu*bft_b(eR) > f_peak - 40,  eR = eR + step;  end
eL = eL - step;
eR = eR + step;

N_eta  = max(N_th * 4, 4000);
eg     = linspace(eL, eR, N_eta)';
de     = eg(2) - eg(1);
lp_eta = chi * eg - nu * bft_b(eg);
lmax   = max(lp_eta);
log_Z_eta = lmax + log(sum(exp(lp_eta - lmax)) * de);

%% -----------------------------------------------------------------------
%  THETA PLOT RANGE
% -----------------------------------------------------------------------
cdf_eta = cumsum(exp(lp_eta - lmax)) / sum(exp(lp_eta - lmax));
idx_lo  = find(cdf_eta >= 0.001, 1);
idx_hi  = find(cdf_eta <= 0.999, 1, 'last');
if isempty(idx_lo),  idx_lo = 1;      end
if isempty(idx_hi),  idx_hi = N_eta;  end

theta_lo = max(1 / (1 + exp(-eg(max(1,      idx_lo - 5)))), 1e-8);
theta_hi = min(1 / (1 + exp(-eg(min(N_eta,  idx_hi + 5)))), 1-1e-8);

use_log_lo = theta_star < 0.04;
use_log_hi = theta_star > 0.96;

if use_log_lo
    theta_plot = logspace(log10(max(theta_lo, 1e-8)), log10(theta_hi), N_th)';
elseif use_log_hi
    theta_plot = sort(1 - logspace(log10(max(1-theta_hi, 1e-8)), ...
                                   log10(1-theta_lo), N_th)');
else
    rng_th     = theta_hi - theta_lo;
    theta_plot = linspace(max(1e-5, theta_lo - 0.05*rng_th), ...
                          min(1-1e-5, theta_hi + 0.05*rng_th), N_th)';
end

%% -----------------------------------------------------------------------
%  CB DENSITY ON THETA_PLOT
% -----------------------------------------------------------------------
eta_th   = log(theta_plot ./ (1 - theta_plot));
lp_th    = chi * eta_th - nu * bft_b(eta_th);
log_J_th = -log1p(exp(-eta_th)) - log1p(exp(eta_th));
p_cb     = exp(lp_th - log_J_th - log_Z_eta);

%% -----------------------------------------------------------------------
%  THREE BETA APPROXIMATIONS
% -----------------------------------------------------------------------
a_ch = chi + 1;   b_ch = nu - chi + 1;             % Bernoulli-Beta
[a_dl, b_dl] = cb_to_beta_data_mm(chi, nu);         % Data-level MM

if verb,  fprintf('cb_posterior_demo: computing mu-Hellinger-optimal Beta...\n');  end
[a_hmu, b_hmu, ~, H2_hmu] = cb_to_beta_hellinger_mu(chi, nu, ...
    'max_iter', MXIT, 'verbose', verb);

%% -----------------------------------------------------------------------
%  B'(eta) AND B''(eta) ON FULL ETA GRID (for mu-space push-forward)
% -----------------------------------------------------------------------
bp_eg  = zeros(N_eta, 1);
bpp_eg = zeros(N_eta, 1);
for k = 1:N_eta
    [~, bp_eg(k), bpp_eg(k)] = bft_all(eg(k));
end

%% -----------------------------------------------------------------------
%  H^2 IN MU-SPACE FOR ALL THREE
%  rho = integral sqrt(p_CB_mu(m) * betapdf(m,a,b)) dm
%  On eta grid: integral sqrt(exp(lp_eta - log_Z_eta) * B'' * betapdf(B'(eta),a,b)) deta
% -----------------------------------------------------------------------
mu_k   = max(bp_eg,   1e-300);
omu_k  = max(1-bp_eg, 1e-300);
bpp_k  = max(bpp_eg,  1e-300);
lp_eff = lp_eta + log(bpp_k);
lmx_e  = max(lp_eff);
Zf_mu  = sum(exp(lp_eta - lmx_e)) * de;   % Z_CB * exp(-lmax_eff)
lZf_mu = log(Zf_mu);

    function H2 = h2_mu(a, b)
        if isnan(a)||isnan(b)||a<=0||b<=0, H2=NaN; return; end
        lg  = (lp_eff - lmx_e)/2 + ((a-1)/2)*log(mu_k) + ((b-1)/2)*log(omu_k);
        lmg = max(lg);
        S0  = sum(exp(lg - lmg)) * de;
        if max(a,b) > 1e8
            if b>a, bn=gammaln(a)-(a*log(b)+a*(a-1)/(2*b));
            else,   bn=gammaln(b)-(b*log(a)+b*(b-1)/(2*a)); end
        else
            bn = betaln(a,b);
        end
        rho = exp(lmg + log(S0) - 0.5*lZf_mu - 0.5*bn);
        H2  = max(0, 1 - min(1, rho));
    end

H2_bb  = h2_mu(a_ch,  b_ch);
H2_dl  = h2_mu(a_dl,  b_dl);
% H2_hmu returned from cb_to_beta_hellinger_mu

%% -----------------------------------------------------------------------
%  COLOUR SCHEME
% -----------------------------------------------------------------------
col_cb  = [0.10 0.45 0.85];   % blue:   exact CB
col_ch  = [0.85 0.20 0.10];   % red:    Bernoulli-Beta
col_dl  = [0.90 0.55 0.00];   % orange: data-level MM
col_hmu = [0.10 0.65 0.25];   % green:  mu-Hellinger-optimal
lw = 2.0;

%% =======================================================================
%  MU-SPACE PLOT  (xaxis = 'mu')
% =======================================================================
if strcmp(xaxis, 'mu')

    % CB-exact proxy distribution via eta-grid push-forward
    [mu_grid_sorted, sort_idx] = sort(bp_eg);
    lp_sorted  = lp_eta(sort_idx);
    bpp_sorted = bpp_eg(sort_idx);
    log_p_mu   = lp_sorted - log(bpp_sorted) - log_Z_eta;
    p_mu_cb    = exp(log_p_mu);
    good       = isfinite(p_mu_cb) & p_mu_cb >= 0;
    mu_grid_sorted = mu_grid_sorted(good);
    p_mu_cb        = p_mu_cb(good);

    % Beta proxy distributions: Beta(a,b) density on the loss axis.
    % Each strategy draws p~Beta(a,b) and uses p directly as the loss proxy.
    mu_means    = [a_ch/(a_ch+b_ch), a_dl/(a_dl+b_dl), a_hmu/(a_hmu+b_hmu)];
    mu_lo       = max(0.001, min([min(mu_grid_sorted)*0.85, min(mu_means)*0.7]));
    mu_hi       = min(0.999, max([max(mu_grid_sorted)*1.10, max(mu_means)*1.2]));
    mu_plot_vec = linspace(mu_lo, mu_hi, N_th);

    p_ch_mu  = betapdf(mu_plot_vec, a_ch,  b_ch);
    p_dl_mu  = betapdf(mu_plot_vec, a_dl,  b_dl);
    p_hmu_mu = betapdf(mu_plot_vec, a_hmu, b_hmu);

    if isempty(hfig)
        figure('Position', [80 80 960 580], ...
            'Name', sprintf('CB demo (proxy): mu=%.4f nu=%g', mu, nu));
    else
        figure(hfig);  clf;
    end

    hold on;
    h_ch2  = plot(mu_plot_vec, p_ch_mu,  '--', 'Color', col_ch,  'LineWidth', lw);
    h_dl2  = plot(mu_plot_vec, p_dl_mu,  '-.', 'Color', col_dl,  'LineWidth', lw);
    h_hmu2 = plot(mu_plot_vec, p_hmu_mu, '-',  'Color', col_hmu, 'LineWidth', lw);
    h_cb2  = plot(mu_grid_sorted, p_mu_cb, '-', 'Color', col_cb, 'LineWidth', lw+0.5);

    safe_max = @(v) max(v(isfinite(v) & v >= 0));
    ylim([0, max([safe_max(p_mu_cb), safe_max(p_ch_mu), ...
                  safe_max(p_dl_mu), safe_max(p_hmu_mu)]) * 1.08]);

    yl = ylim;
    plot([mu mu], [0, 0.92*yl(2)], ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.2);
    text(mu, 0.94*yl(2), sprintf('  \\mu = %.3f', mu), ...
        'FontSize', 8, 'Color', [0.3 0.3 0.3], 'VerticalAlignment', 'bottom', ...
        'Interpreter', 'tex');
    hold off;

    grid on;  box on;
    xlabel('Sampled loss proxy  (Thompson sampling selection statistic)', 'FontSize', 11);
    ylabel('Density of loss proxy', 'FontSize', 11);
    title({'Thompson sampling loss proxy distributions', ...
           sprintf('mu = %.4f  (true expected loss)     nu = %g  (observations)', mu, nu), ...
           'All Beta curves: p~Beta(a,b) used directly.  CB-exact: B''(logit(theta)), theta~posterior.'}, ...
          'FontSize', 9, 'Interpreter', 'none');

    mu_ch_proxy  = a_ch/(a_ch+b_ch);
    mu_dl_proxy  = a_dl/(a_dl+b_dl);
    mu_hmu_proxy = a_hmu/(a_hmu+b_hmu);
    legend([h_cb2, h_ch2, h_dl2, h_hmu2], ...
        {sprintf('CB exact oracle:    proxy mean = %.3f  [oracle]', mu), ...
         sprintf('Bernoulli-Beta:     proxy mean = %.3f    H^2_mu = %.4f', ...
                 mu_ch_proxy,  H2_bb), ...
         sprintf('Data-level MM:      proxy mean = %.3f    H^2_mu = %.4f', ...
                 mu_dl_proxy,  H2_dl), ...
         sprintf('mu-Hellinger-opt:   proxy mean = %.3f    H^2_mu = %.4f', ...
                 mu_hmu_proxy, H2_hmu)}, ...
        'Location', 'best', 'FontSize', 8, 'Interpreter', 'none');

    drawnow;
    if nargout == 0,  clear H2_bb H2_dl H2_hmu;  end
    return;

end  % mu-space branch

%% =======================================================================
%  THETA-SPACE PLOT  (xaxis = 'theta', default)
% =======================================================================
if isempty(hfig)
    figure('Position', [80 80 960 580], ...
        'Name', sprintf('CB demo: mu=%.4f nu=%g theta*=%.4g', mu, nu, theta_star));
else
    figure(hfig);  clf;
end

p_ch_plt  = betapdf(theta_plot, a_ch,  b_ch);
p_dl_plt  = betapdf(theta_plot, a_dl,  b_dl);
p_hmu_plt = betapdf(theta_plot, a_hmu, b_hmu);

hold on;
h_ch  = plot(theta_plot, p_ch_plt,  '--', 'Color', col_ch,  'LineWidth', lw);
h_dl  = plot(theta_plot, p_dl_plt,  '-.', 'Color', col_dl,  'LineWidth', lw);
h_hmu = plot(theta_plot, p_hmu_plt, '-',  'Color', col_hmu, 'LineWidth', lw);
h_cb  = plot(theta_plot, p_cb,      '-',  'Color', col_cb,  'LineWidth', lw+0.5);

safe_max = @(v) max(v(isfinite(v) & v >= 0));
ylim([0, max([safe_max(p_cb), safe_max(p_ch_plt), ...
              safe_max(p_dl_plt), safe_max(p_hmu_plt)]) * 1.08]);

yl = ylim;
plot([theta_star theta_star], [0, 0.92*yl(2)], ':', ...
    'Color', [0.5 0.5 0.5], 'LineWidth', 1.2);
text(theta_star, 0.94*yl(2), sprintf('  \\theta^* = %.4g', theta_star), ...
    'FontSize', 8, 'Color', [0.3 0.3 0.3], 'VerticalAlignment', 'bottom', ...
    'Interpreter', 'tex');
hold off;

if use_log_lo || use_log_hi
    set(gca, 'XScale', 'log');
end
grid on;  box on;

xlabel('\theta  (CB canonical / arm parameter)', 'FontSize', 11);
ylabel('Posterior density  p(\theta | \chi, \nu)', 'FontSize', 11);
title({'CB posterior in theta-space', ...
       sprintf('mu = chi/nu = %.4f  (observed mean loss)     nu = %g  (observations)', mu, nu), ...
       sprintf('theta* = %.4g  (arm parameter = where CB posterior concentrates)', theta_star)}, ...
      'FontSize', 9, 'Interpreter', 'none');

% H^2 in theta-space for the theta-space plot
lf0    = (lp_eta - lmax) / 2;
Z_f0   = sum(exp(2 * lf0)) * de;
ls_eg  = -log1p(exp(-eg));
ls1_eg = -log1p(exp( eg));
H2_ch_th  = h2_th(a_ch,  b_ch,  lf0, ls_eg, ls1_eg, de, Z_f0);
H2_dl_th  = h2_th(a_dl,  b_dl,  lf0, ls_eg, ls1_eg, de, Z_f0);
H2_hmu_th = h2_th(a_hmu, b_hmu, lf0, ls_eg, ls1_eg, de, Z_f0);

legend([h_cb, h_ch, h_dl, h_hmu], ...
    {'Exact CB posterior', ...
     sprintf('Bernoulli-Beta(%s,%s)    H^2_theta = %.4f    H^2_mu = %.4f', ...
             fmt(a_ch),  fmt(b_ch),  H2_ch_th,  H2_bb), ...
     sprintf('Data-level MM(%s,%s)     H^2_theta = %.4f    H^2_mu = %.4f', ...
             fmt(a_dl),  fmt(b_dl),  H2_dl_th,  H2_dl), ...
     sprintf('mu-Hellinger(%s,%s)      H^2_theta = %.4f    H^2_mu = %.4f', ...
             fmt(a_hmu), fmt(b_hmu), H2_hmu_th, H2_hmu)}, ...
    'Location', 'best', 'FontSize', 8, 'Interpreter', 'none');

drawnow;
if nargout == 0,  clear H2_bb H2_dl H2_hmu;  end

end  % cb_posterior_demo


%% -----------------------------------------------------------------------
function H2 = h2_th(a, b, lf0, ls, ls1, de, Z_f0)
% H^2 in theta-space (CB posterior vs Beta(a,b) over arm parameter theta).
if isnan(a)||isnan(b)||a<=0||b<=0,  H2=NaN; return;  end
lg  = lf0 + (a/2)*ls + (b/2)*ls1;
lmg = max(lg);
S0  = sum(exp(lg - lmg)) * de;
if max(a,b) > 1e8
    if b>a, bn=gammaln(a)-(a*log(b)+a*(a-1)/(2*b));
    else,   bn=gammaln(b)-(b*log(a)+b*(b-1)/(2*a)); end
else
    bn = betaln(a,b);
end
H2 = max(0, 1 - min(1, exp(lmg + log(S0) - 0.5*log(Z_f0) - 0.5*bn)));
end


%% -----------------------------------------------------------------------
function s = fmt(x)
if     x >= 1000 || x < 0.001,  s = sprintf('%.3g', x);
elseif x >= 100,                 s = sprintf('%.1f', x);
elseif x >= 10,                  s = sprintf('%.2f', x);
else,                            s = sprintf('%.3f', x);
end
end
