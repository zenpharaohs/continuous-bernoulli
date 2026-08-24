%% cb_validate_beta.m
%
% Validation of the Beta moment-matched pseudo-prior against the true
% Continuous Bernoulli conjugate posterior.
%
% WHAT THIS DOES
%   For each (chi, nu) test case from cb_validate, we:
%     (1) Compute Beta(a,b) moment-matched to the CB posterior (chi, nu).
%         Moments are computed by CERTIFIED adaptive quadrature (MATLAB
%         integral, G7K15 adaptive Gauss-Kronrod) to AbsTol=1e-14.
%         The integration domain is (-inf, +inf) in eta-space with a
%         log-normalisation shift at the mode for numerical stability.
%     (2) Draw N samples from Beta(a,b) using betarnd.
%     (3) Apply the EXACT CB CDF (certified grid in eta-space) as the
%         PIT map: u_i = F_CB(theta_i).
%     (4) Run KS / AD / CvM / adaptive Legendre chi^2 tests on u_i ~ U(0,1).
%
% KS CRITICAL VALUE
%   Uses ks_critical(N, alpha) -- exact Marsaglia-Tsang-Wang (2003) for
%   N <= 10000, asymptotic Kolmogorov series for N > 10000.
%   The asymptotic approximation sqrt(-log(alpha/2)/(2N)) underestimates
%   the critical value by ~5% at N=20 and ~1% at N=100, causing over-
%   rejection at nominal alpha (anticonservative).  Exact values are used
%   throughout regardless of N.
%
% LEGENDRE SNR VALIDITY
%   The block SNR ~ N(0,1) null distribution requires N >> D_MAX (CLT
%   on the Legendre projections).  A warning is printed when N < D_MAX.
%   For strong-signal cases (SNR >> 5) this caveat is irrelevant; for
%   borderline cases at small N the SNR threshold should be treated with
%   extra caution.
%
% H^2 REPORTING
%   H2_leg = chi2_leg / 8 is the second-order approximation to the
%   Hellinger distance, valid when distributions are close (chi2 << 8).
%   When chi2_leg > 8 (distributions are far apart), H2_leg is clamped
%   to 1.0 and flagged: the true H^2 is close to 1 but the formula
%   saturates.  The SNR is still valid as a detection statistic.
%
% USAGE
%   addpath('src/matlab')
%   cb_validate_beta                          % N=1e6 (standard)
%   cb_validate_beta_mode = 'quick';    cb_validate_beta   % N=1e5
%   cb_validate_beta_mode = 'thorough'; cb_validate_beta   % N=1e7
%   N_QUAL = 20; cb_validate_beta             % arbitrary N from workspace
%
% MIT License.  Andrew Mullhaupt, Stony Brook University AMS/QF, 2026.

clear pi;

%% Path setup
cbv_root = fileparts(mfilename('fullpath'));
if isempty(cbv_root), cbv_root = pwd; end
if exist(fullfile(cbv_root,'src','matlab','cb_stream.m'),'file')
    addpath(cbv_root);
    addpath(fullfile(cbv_root,'src','matlab'));
end
clear cbv_root;

%% N_QUAL: workspace variable overrides mode switch
if exist('N_QUAL','var') && isnumeric(N_QUAL) && N_QUAL >= 1
    N_QUAL    = round(N_QUAL);
    mode_desc = sprintf('CUSTOM N=%d', N_QUAL);
else
    if exist('cb_validate_beta_mode','var'), mode_str = cb_validate_beta_mode;
    else,                                    mode_str = 'standard';
    end
    switch lower(mode_str)
        case 'quick',    N_QUAL = 100000;   mode_desc = 'QUICK';
        case 'thorough', N_QUAL = 10000000; mode_desc = 'THOROUGH';
        otherwise,       N_QUAL = 1000000;  mode_desc = 'STANDARD';
    end
end

%% Quadrature and grid parameters
QUAD_ATOL = 1e-14;
QUAD_RTOL = 1e-12;
CDF_NP    = 20000;

%% Test cases
cases = {
   0.25,    5,   'Z=0.05 nu=5   ';
   0.50,   10,   'Z=0.05 nu=10  ';
   0.50,    5,   'Z=0.10 nu=5   ';
   2.00,   20,   'Z=0.10 nu=20  ';
   4.00,   40,   'Z=0.10 nu=40  ';
   4.00,   10,   'Z=0.40 nu=10  ';
   5.00,   10,   'Z=0.50 nu=10  ';
   3.00,   10,   'Z=0.30 nu=10  ';
   10.0,   20,   'Z=0.50 nu=20  ';
   6.00,   20,   'Z=0.30 nu=20  ';
   18.0,   40,   'Z=0.45 nu=40  ';
   25.0,   50,   'Z=0.50 nu=50  ';
   40.0,  100,   'Z=0.40 nu=100 ';
   50.0,  100,   'Z=0.50 nu=100 ';
   90.0,  200,   'Z=0.45 nu=200 ';
  100.0,  200,   'Z=0.50 nu=200 ';
  150.0,  500,   'Z=0.30 nu=500 ';
  200.0,  500,   'Z=0.40 nu=500 ';
  250.0,  500,   'Z=0.50 nu=500 ';
  400.0, 1000,   'Z=0.40 nu=1k  ';
  500.0, 1000,   'Z=0.50 nu=1k  ';
  100.0,  500,   'Z=0.20 nu=500 ';
 2500.0, 5000,   'Z=0.50 nu=5k  ';
};
n_cases = size(cases, 1);

%% Adaptive Legendre parameters
D_MAX        = 128;
SNR_STOP     = 2.0;
BLOCK_LO     = [1,  9, 17, 33,  65];
BLOCK_HI     = [8, 16, 32, 64, 128];
N_BLOCKS_LEG = numel(BLOCK_LO);
LEG_NORMS    = sqrt(3:2:(2*D_MAX+1));

%% Critical values
KS_ALPHA = 0.01;
ks_crit  = ks_critical(N_QUAL, KS_ALPHA);   % exact MTW (not asymptotic)
ad_crit  = 3.857;
cvm_crit = 0.743;

%% Header
fprintf('\n');
fprintf('=================================================================\n');
fprintf('  CB Validate Beta: Beta-MM vs CB posterior  [%s]\n', mode_desc);
fprintf('  %s\n', datestr(now));
fprintf('=================================================================\n\n');
fprintf('N per case:  %s\n', fmt_N(N_QUAL));
fprintf('KS crit:     %.6f  (exact MTW; asymptotic would be %.6f)\n', ...
    ks_crit, sqrt(-log(KS_ALPHA/2)/(2*N_QUAL)));
fprintf('Moments:     certified adaptive quadrature (G7K15, AbsTol=%.0e)\n', QUAD_ATOL);
fprintf('CDF grid:    %d-point pchip in eta-space\n', CDF_NP);
fprintf('Verdict:     Leg_SNR > 5 => Beta MM detectable at this N.\n');
if N_QUAL < D_MAX
    fprintf('\nWARNING: N=%d < D_MAX=%d. Legendre block SNR null distribution\n', N_QUAL, D_MAX);
    fprintf('  requires N >> D_MAX (CLT on Legendre projections).\n');
    fprintf('  SNR values are reliable for strong signal (>>5) but\n');
    fprintf('  borderline cases (SNR 3-10) should be verified at larger N.\n');
end
fprintf('\n');

fprintf('%-18s %5s %5s  %6s %6s %6s  %+7s  %4s  %8s %8s\n', ...
    'Case','Z','nu','KS','AD','CvM','LegSNR','deff','a','b');
fprintf('%s\n', repmat('-',92,1));

rng(137);

%% Main loop
Results = struct('chi',{},'nu',{},'desc',{},...
                 'a',{},'b',{},'m_cb',{},'v_cb',{},...
                 'ks',{},'ad',{},'cvm',{},...
                 'ks_pass',{},'ad_pass',{},'cvm_pass',{},...
                 'chi2_leg',{},'H2_leg',{},'H2_saturated',{},...
                 'leg_snr',{},'d_eff',{});

for c = 1:n_cases
    chi_p = cases{c,1};
    nu    = cases{c,2};
    desc  = strtrim(cases{c,3});
    Z     = chi_p / nu;

    % ----------------------------------------------------------------
    % STEP 1: Certified moments via adaptive quadrature in eta-space.
    % ----------------------------------------------------------------
    eta_s    = cb_mode(chi_p, nu);
    log_peak = chi_p*eta_s - nu*bft_b(eta_s);

    p_s  = @(eta) exp( chi_p*eta - nu*cb_bft_b_vec(eta) - log_peak );
    sig  = @(eta) 1 ./ (1 + exp(-eta));
    f0   = @(eta) p_s(eta);
    f1   = @(eta) sig(eta) .* p_s(eta);
    f2   = @(eta) sig(eta).^2 .* p_s(eta);

    opts = {'AbsTol', QUAD_ATOL, 'RelTol', QUAD_RTOL, 'ArrayValued', false};
    Z_q  = integral(f0, -Inf, Inf, opts{:});
    m1   = integral(f1, -Inf, Inf, opts{:}) / Z_q;
    m2   = integral(f2, -Inf, Inf, opts{:}) / Z_q;
    v_cb = max(m2 - m1^2, 0);

    % ----------------------------------------------------------------
    % STEP 2: Beta moment-match.
    % ----------------------------------------------------------------
    [a_b, b_b] = moments_to_beta(m1, v_cb);

    % ----------------------------------------------------------------
    % STEP 3: Certified CB CDF for PIT.
    % ----------------------------------------------------------------
    [~, ~, bpp_s] = bft_all(eta_s);
    sig_eta   = 1 / sqrt(max(nu * bpp_s, 1e-30));
    tail_tol  = 1e-10;
    half_span = 5 * sig_eta;
    while true
        tail_lo = integral(f0, -Inf, eta_s - half_span, opts{:});
        tail_hi = integral(f0, eta_s + half_span, Inf,  opts{:});
        if (tail_lo + tail_hi) / Z_q < tail_tol,  break;  end
        half_span = half_span * 1.5;
    end
    eL = eta_s - half_span;
    eR = eta_s + half_span;

    eg  = linspace(eL, eR, CDF_NP);
    lp  = chi_p*eg - nu*arrayfun(@bft_b, eg);
    lp  = lp - max(lp);
    pe  = exp(lp);
    de  = diff(eg);
    pm  = 0.5*(pe(1:end-1)+pe(2:end));
    wts = pm .* de;
    cg  = [0, cumsum(wts / sum(wts))];  cg(end) = 1;

    % ----------------------------------------------------------------
    % STEP 4: Draw from Beta(a,b), PIT, accumulate Legendre projections.
    % ----------------------------------------------------------------
    blk_sz = max(1, min(500000, round(N_QUAL/20)));
    n_blks = ceil(N_QUAL / blk_sz);
    S_leg  = zeros(1, D_MAX);
    u_all  = zeros(N_QUAL, 1);
    ptr    = 0;

    for blk = 1:n_blks
        bs      = min(blk_sz, N_QUAL - (blk-1)*blk_sz);
        theta_b = betarnd(a_b, b_b, bs, 1);
        tc_b    = max(realmin, min(1-eps, theta_b));
        eta_b   = log(tc_b ./ (1 - tc_b));
        u_b     = interp1(eg, cg, max(eL, min(eR, eta_b)), 'pchip', 'extrap');
        u_b     = max(0, min(1, u_b));
        u_all(ptr+1:ptr+bs) = u_b;
        ptr     = ptr + bs;

        x   = 2*u_b - 1;
        Pk2 = ones(bs,1);  Pk1 = x;
        S_leg(1) = S_leg(1) + LEG_NORMS(1)*sum(x);
        for k = 2:D_MAX
            Pk = ((2*k-1)*x.*Pk1 - (k-1)*Pk2) / k;
            S_leg(k) = S_leg(k) + LEG_NORMS(k)*sum(Pk);
            Pk2 = Pk1;  Pk1 = Pk;
        end
    end

    % ----------------------------------------------------------------
    % STEP 5: KS / AD / CvM  (exact KS critical value via ks_critical)
    % ----------------------------------------------------------------
    u_sorted = sort(u_all);
    i_vec    = (1:N_QUAL)';
    ks       = max(abs(i_vec/N_QUAL - u_sorted));
    ks_pass  = ks < ks_crit;
    u_ad     = max(1e-300, min(1-1e-300, u_sorted));
    ad       = -N_QUAL - mean((2*i_vec-1).*(log(u_ad)+log(1-u_ad(end:-1:1))));
    ad_pass  = ad < ad_crit;
    cvm      = 1/(12*N_QUAL) + sum((u_sorted - (2*i_vec-1)/(2*N_QUAL)).^2);
    cvm_pass = cvm < cvm_crit;

    % ----------------------------------------------------------------
    % STEP 6: Adaptive Legendre chi^2
    % ----------------------------------------------------------------
    c_hat     = S_leg / N_QUAL;
    blk_snr   = zeros(1, N_BLOCKS_LEG);
    blk_chi2d = zeros(1, N_BLOCKS_LEG);
    for j = 1:N_BLOCKS_LEG
        lo_j = BLOCK_LO(j);  hi_j = BLOCK_HI(j);  bs_j = hi_j-lo_j+1;
        B_j  = sum(c_hat(lo_j:hi_j).^2);
        blk_snr(j)   = (N_QUAL*B_j - bs_j) / sqrt(2*bs_j);
        blk_chi2d(j) = B_j - bs_j/N_QUAL;
    end
    sig_blocks = find(abs(blk_snr) > SNR_STOP);
    if isempty(sig_blocks)
        d_eff = 0;  chi2_leg = 0;
    else
        d_eff    = BLOCK_HI(sig_blocks(end));
        chi2_leg = max(sum(blk_chi2d(1:sig_blocks(end))), 0);
    end
    sig_H2  = sqrt(max(d_eff,1)) / (2*sqrt(2)*N_QUAL);
    leg_snr = chi2_leg / (sig_H2 * 8);

    % H^2: second-order approximation H2 ~ chi2/8, valid when chi2 << 8.
    % Clamp to [0,1]: when chi2 > 8 the distributions are far apart and
    % H^2 is close to 1 -- the approximation saturates.
    H2_leg       = min(chi2_leg / 8, 1.0);
    H2_saturated = (chi2_leg > 8);   % flag: approximation not in linear regime

    Results(c).chi=chi_p; Results(c).nu=nu; Results(c).desc=desc;
    Results(c).a=a_b; Results(c).b=b_b; Results(c).m_cb=m1; Results(c).v_cb=v_cb;
    Results(c).ks=ks; Results(c).ad=ad; Results(c).cvm=cvm;
    Results(c).ks_pass=ks_pass; Results(c).ad_pass=ad_pass; Results(c).cvm_pass=cvm_pass;
    Results(c).chi2_leg=chi2_leg; Results(c).H2_leg=H2_leg;
    Results(c).H2_saturated=H2_saturated;
    Results(c).leg_snr=leg_snr; Results(c).d_eff=d_eff;

    pass_str = @(p) sel(p,'  ok','FAIL');
    fprintf('%-18s %5.3f %5d  %6s %6s %6s  %+7.1f  %4d  %8.4f %8.4f\n', ...
        desc, Z, nu, pass_str(ks_pass), pass_str(ad_pass), pass_str(cvm_pass), ...
        leg_snr, d_eff, a_b, b_b);
end

%% Summary
fprintf('\n%s\n\n', repmat('=',92,1));
n_kf = sum(~[Results.ks_pass]);
n_af = sum(~[Results.ad_pass]);
n_cf = sum(~[Results.cvm_pass]);
fprintf('ORDER-STATISTIC FAILURES  (N=%s, alpha=%.2f, exact KS crit=%.6f)\n', ...
    fmt_N(N_QUAL), KS_ALPHA, ks_crit);
fprintf('  KS: %d   AD: %d   CvM: %d\n\n', n_kf, n_af, n_cf);

leg_snr_vals = [Results.leg_snr];
n_det = sum(abs(leg_snr_vals) > 5);
n_bdr = sum(abs(leg_snr_vals) > 3 & abs(leg_snr_vals) <= 5);
n_ok  = sum(abs(leg_snr_vals) <= 3);
fprintf('ADAPTIVE LEGENDRE chi^2 VERDICT  (|SNR|>5 => detectable)\n');
fprintf('  Detectable    (|SNR|>5):       %d/%d\n', n_det, n_cases);
fprintf('  Borderline    (3<|SNR|<=5):    %d/%d\n', n_bdr, n_cases);
fprintf('  Indistinguishable (|SNR|<=3):  %d/%d\n', n_ok, n_cases);
if N_QUAL < D_MAX
    fprintf('  [N=%d < D_MAX=%d: SNR null distribution is approximate;\n', N_QUAL, D_MAX);
    fprintf('   borderline cases should be confirmed at larger N]\n');
end
fprintf('\n');

fprintf('Numerical moments and Beta parameters (certified G7K15 quadrature):\n');
fprintf('  %-18s %5s %5s  %10s  %10s  %8s  %8s\n','Case','Z','nu','m_CB','v_CB','a','b');
fprintf('  %s\n', repmat('-',80,1));
for c = 1:n_cases
    r = Results(c);
    fprintf('  %-18s %5.3f %5d  %10.8f  %10.3e  %8.4f  %8.4f\n', ...
        r.desc, r.chi/r.nu, r.nu, r.m_cb, r.v_cb, r.a, r.b);
end
fprintf('\n');

if n_det > 0
    fprintf('Detectable cases:\n');
    fprintf('  %-18s %5s %5s  %+9s  %4s  %9s\n', ...
        'Case','Z','nu','LegSNR','deff','H^2_leg');
    for c = 1:n_cases
        r = Results(c);
        if abs(r.leg_snr) > 5
            h2_str = sprintf('%9.2e', r.H2_leg);
            if r.H2_saturated
                h2_str = '  ~1 (sat)';   % approximation saturated; true H^2 ~ 1
            end
            fprintf('  %-18s %5.3f %5d  %+9.1f  %4d  %s\n', ...
                r.desc, r.chi/r.nu, r.nu, r.leg_snr, r.d_eff, h2_str);
        end
    end
    fprintf('\n  (sat) = chi2_leg > 8: H^2 ~ chi2/8 approximation saturated;\n');
    fprintf('         true H^2 is close to 1 (distributions nearly singular).\n\n');
end

fprintf('-------------------------------------------------------------\n');
if n_det > 0
    fprintf('RESULT: Beta MM is STATISTICALLY DISTINGUISHABLE from CB\n');
    fprintf('  in %d/%d cases at N=%s.\n', n_det, n_cases, fmt_N(N_QUAL));
elseif n_bdr > 0
    fprintf('RESULT: BORDERLINE -- rerun at larger N to confirm\n');
else
    fprintf('RESULT: Beta MM is INDISTINGUISHABLE from CB at N=%s.\n', fmt_N(N_QUAL));
    fprintf('  (Not H^2=0; test lacks power at this N.)\n');
end
fprintf('-------------------------------------------------------------\n');
fprintf('\nDone: %s\n\n', datestr(now));

%% =========================================================================
%% Local helpers
%% =========================================================================

function out = cb_bft_b_vec(eta_vec)
    out = arrayfun(@bft_b, eta_vec);
end

function [a, b] = moments_to_beta(m, v)
    v = min(v, m*(1-m) * 0.9999);
    v = max(v, 1e-9);
    c = m*(1-m)/v - 1;
    c = max(c, 1e-4);
    a = m * c;
    b = (1-m) * c;
end

function s = sel(c, a, b)
    if c, s = a; else, s = b; end
end

function s = fmt_N(n)
    if n >= 1e9,     s = sprintf('%.1fG', n/1e9);
    elseif n >= 1e6, s = sprintf('%.1fM', n/1e6);
    elseif n >= 1e3, s = sprintf('%.1fk', n/1e3);
    else,            s = sprintf('%d', round(n));
    end
end
