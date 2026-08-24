%% cb_validate.m
%
% Installation validation for the Continuous Bernoulli conjugate posterior
% sampler (cb_stream C backend + MATLAB wrapper).
%
% USAGE
%   addpath('src/matlab')
%   cb_validate                        % standard  (N=1e7 per case, ~90 sec)
%   cb_validate_mode='quick';    cb_validate   % N=1e5, ~5 sec
%   cb_validate_mode='thorough'; cb_validate   % N=1e8, ~15 min
%   cb_validate_hq_cv=true; cb_validate_mode='thorough'; cb_validate
%       Also run fitted-Beta control-variate Hellinger certification.
%   cb_validate_case_idx=23; cb_validate_hq_cv=true; cb_validate_mode='thorough'; cb_validate
%       Run only selected case rows, useful for high-N Hellinger checks.
%
%   % Extended streaming at arbitrary N (Legendre + bark rates, no sort):
%   cb_validate_N_stream = 1e9;  cb_validate
%
% TESTS (after PIT: U_i = F_P(theta_i) ~ Uniform(0,1) under H0)
%
% TIER 1: order-statistic tests  (O(N) memory, cannot be batched)
%   KS   Kolmogorov-Smirnov:    sup|F_N - U|
%   AD   Anderson-Darling:      -N - (1/N)*sum(2i-1)*(log u_(i)+log(1-u_(N+1-i)))
%   CvM  Cramer-von Mises:      1/(12N) + sum(u_(i) - (2i-1)/(2N))^2
%   Critical values (alpha=0.01): KS: 1.36/sqrt(N), AD: 3.857, CvM: 0.743
%
% TIER 2: adaptive streaming Legendre chi^2  (O(d_max) memory, any N)
%   Computes chi^2(P_U || U) = sum_{k=1}^{d_eff} c_k^2 via Parseval.
%   H^2_est = chi^2_debiased / 8   [exact to 2nd order when p_U near 1]
%
%   ADAPTIVE ORDER via dyadic block energies:
%     Block j covers modes [lo_j, hi_j] (sizes 8,8,16,32,64,64,...).
%     Block SNR_j = (N*B_j - size_j) / sqrt(2*size_j)  ~ N(0,1) under H0.
%     d_eff = last mode in last block with |SNR| > SNR_STOP (default 2).
%     Modes beyond d_eff are at noise floor and contribute nothing but bias.
%     chi^2 = sum of debiased block energies up to d_eff.
%
%   VARIANCE:
%     sigma(H^2_est) = sqrt(d_eff) / (2*sqrt(2)*N)   -- O(1/N), not O(1/sqrt(N)).
%     At N=1e7, d_eff=40: ~1000x lower sigma than spacing D_B estimator.
%
%   VERDICT LOGIC (primary arbiter):
%     Leg SNR_total > 5:    real signal -- approximation error present
%     Leg SNR_total in 3-5: borderline -- increase N_stream
%     Leg SNR_total < 3:    no signal -- sampler correct at this N
%     KS/AD/CvM failures are cross-classified: if Leg SNR<3 they are noise.
%
% THE CERTIFICATION NUMBER (from spacing D_B estimate)
%   N_cert_lo = log(4/3)/(2*D_B)  [necessary: below this, no test can detect]
%   N_cert_hi = log(3)  /(2*D_B)  [sufficient: above this, NP test can detect]
%   N_cert_cert = log(3)/(2*(D_B+2.33*sigma_DB))  [99% conservative]
%
% MIT License.  Andrew Mullhaupt, Stony Brook University AMS/QF, 2026.

clear pi;

%% Path setup (survives session crashes -- re-adds src/matlab if needed)
cbv_root = fileparts(mfilename('fullpath'));
if isempty(cbv_root), cbv_root = pwd; end
if exist(fullfile(cbv_root,'src','matlab','cb_stream.m'),'file')
    addpath(cbv_root);
    addpath(fullfile(cbv_root,'src','matlab'));
end
clear cbv_root;

%% Configuration
if exist('cb_validate_mode','var'), mode_str = cb_validate_mode;
else,                               mode_str = 'standard';
end

switch lower(mode_str)
    case 'quick'
        N_QUAL = 100000;     mode_desc = 'QUICK';     eta_np = 6000;
    case 'thorough'
        N_QUAL = 100000000;  mode_desc = 'THOROUGH';  eta_np = 20000;
    otherwise
        N_QUAL = 10000000;   mode_desc = 'STANDARD';  eta_np = 12000;
end

% Adaptive Legendre parameters
D_MAX     = 128;       % maximum modes computed
SNR_STOP  = 2.0;       % block SNR threshold
% Dyadic block layout: [1..8], [9..16], [17..32], [33..64], [65..128]
BLOCK_LO  = [1,  9, 17, 33,  65];
BLOCK_HI  = [8, 16, 32, 64, 128];
N_BLOCKS_LEG = numel(BLOCK_LO);
% Precompute Legendre normalisation constants sqrt(2k+1) for k=1..D_MAX
% (avoids computing sqrt inside the hot loop -- ~2x speedup)
LEG_NORMS = sqrt(3:2:(2*D_MAX+1));  % LEG_NORMS(k) = sqrt(2k+1)

% Extended streaming
if exist('cb_validate_N_stream','var') && cb_validate_N_stream > N_QUAL
    N_STREAM = round(cb_validate_N_stream);
else
    N_STREAM = 0;
end
DO_HQ_CV = exist('cb_validate_hq_cv','var') && logical(cb_validate_hq_cv);

KS_ALPHA  = 0.01;
ks_crit   = ks_critical(N_QUAL, KS_ALPHA);   % exact MTW, not asymptotic
ad_crit   = 3.857;
cvm_crit  = 0.743;
sigma_DB  = sqrt(pi-1) / (2*sqrt(N_QUAL));
z99       = 2.326;
NC_floor  = log(3) / (2 * z99 * sigma_DB);

%% Header
fprintf('\n');
fprintf('=================================================================\n');
fprintf('  CB Sampler: Installation Validation  [%s]\n', mode_desc);
fprintf('  %s\n', datestr(now));
fprintf('=================================================================\n\n');
fprintf('N_qual:      %s per case\n', fmt_N(N_QUAL));
fprintf('Tests:       KS/AD/CvM  (order-statistic, O(N) memory)\n');
fprintf('             Legendre chi^2 adaptive (d_max=%d, SNR_stop=%.1f)\n', D_MAX, SNR_STOP);
fprintf('             Dyadic blocks: [1..8],[9..16],[17..32],[33..64],[65..128]\n');
if DO_HQ_CV
    fprintf('             + fitted-Beta CV smoothed Hellinger certification\n');
end
if N_STREAM > 0
    fprintf('             + streaming-only pass at N=%s\n', fmt_N(N_STREAM));
end
fprintf('sigma_DB:    %.2e   (5-sigma = %.2e)\n', sigma_DB, 5*sigma_DB);
fprintf('\n');

%% Check MEX
try
    s_t = cb_stream(0.5, 10, 'seed', uint64(1)); s_t.draw(10); s_t.delete();
    fprintf('MEX backend: OK\n\n');
catch e
    fprintf('ERROR: MEX not available: %s\n', e.message);
    fprintf('Run:  mex(''-O'',''-Isrc\\c'',''-outdir'',''.'',''src\\c\\cb_stream_mex.c'')\n\n');
    return;
end

%% Test cases
% Only Z <= 0.5 cases are tested directly.
% High-Z (Z > 0.5) correctness follows from the exact reflection identity
%   p(theta | chi, nu) = p(1-theta | nu-chi, nu)  [from B(eta)-B(-eta)=eta]
% which is enforced in all sampling paths.  Testing both halves would be
% redundant and would introduce PIT conditioning issues near theta=1 due to
% the integrable (1-theta)^{-1/2} singularity of the CB density there.
% Users are informed that high-Z output precision is IEEE 754-limited near
% theta=1; this is a representation constraint, not a sampler error.
cases = {
   0.25,    5,   'ARS   Z=0.05 nu=5    sig~8.9      ';   % wide, low-Z extreme
   0.50,   10,   'ARS   Z=0.05 nu=10   sig~6.3      ';   % wide, low-Z extreme
   0.50,    5,   'ARS   Z=0.10 nu=5    sig~4.5      ';   % mirror of old Z=0.90 nu=5
   2.00,   20,   'ARS   Z=0.10 nu=20   sig~2.2      ';   % mirror of old Z=0.90 nu=20
   4.00,   40,   'ARS   Z=0.10 nu=40   sig~1.6      ';   % mirror of old Z=0.90 nu=40
   4.00,   10,   'ARS   Z=0.40 nu=10   sig~1.1      ';
   5.00,   10,   'ARS   Z=0.50 nu=10   sig~1.1      ';
   3.00,   10,   'ARS   Z=0.30 nu=10   sig~1.2      ';
   10.0,   20,   'ARS   Z=0.50 nu=20   sig~0.77     ';
   6.00,   20,   'ARS   Z=0.30 nu=20   sig~0.87     ';
   18.0,   40,   'ARS   Z=0.45 nu=40   sig~0.55     ';
   25.0,   50,   'ARS   Z=0.50 nu=50   sig~0.49     ';
   40.0,  100,   'ARS   Z=0.40 nu=100  sig~0.36     ';
   50.0,  100,   'ARS   Z=0.50 nu=100  sig~0.35     ';
   90.0,  200,   'ARS   Z=0.45 nu=200  sig~0.25     ';
  100.0,  200,   'ARS/CF Z=0.50 nu=200 sig~0.22     ';
  150.0,  500,   'ARS/CF Z=0.30 nu=500 sig~0.18     ';
  200.0,  500,   'ARS/CF Z=0.40 nu=500 sig~0.15     ';
  250.0,  500,   'CF    Z=0.50 nu=500  sig~0.14     ';
  400.0, 1000,   'CF    Z=0.40 nu=1k   (routes ARS) ';
  500.0, 1000,   'CF    Z=0.50 nu=1k   sig~0.11     ';
  100.0,  500,   'CF    Z=0.20 nu=500  (routes ARS) ';
 2500.0, 5000,   'CF    Z=0.50 nu=5k   sig~0.049    ';
};
if exist('cb_validate_case_idx','var') && ~isempty(cb_validate_case_idx)
    cases = cases(cb_validate_case_idx, :);
end
n_cases = size(cases,1);
rnames  = {'prior','Gamma','CF','ARS'};

%% Per-case header
fprintf('%-34s %-5s  %5s  %6s %6s %6s  %7s  %4s\n', ...
    'Case','Reg','sigma','KS','AD','CvM','LegSNR','deff');
fprintf('%s\n', repmat('-',80,1));

%% Main loop
all_pass = true;
R = struct('chi',{},'nu',{},'desc',{},'regime',{},'sigma',{},...
           'ks',{},'ad',{},'cvm',{},...
           'ks_pass',{},'ad_pass',{},'cvm_pass',{},...
           'DB_raw',{},'DB',{},'H2',{},'TVD_hi',{},'above_floor',{},...
           'H2_smooth',{},'H2_cv',{},'NC_no_smooth',{},'NC_some_smooth',{},...
           'NC_no_cv',{},'NC_some_cv',{},'cv_alpha',{},'cv_beta',{},'cv_beta_h2_est',{},'cv_beta_h2_exact',{},...
           'chi2_leg',{},'H2_leg',{},'leg_snr',{},'d_eff',{},'block_snr',{},...
           'rho_proj',{},'H2_proj',{},'DB_proj',{},...
           'NC_lo',{},'NC_hi',{},'NC_cert',{},'rate',{},'rate_draw',{});

for c = 1:n_cases
    chi_p = cases{c,1};
    nu    = cases{c,2};
    desc  = strtrim(cases{c,3});

    % ---- Exact CDF (adaptive grid) ----------------------------------
    eta_s = cb_mode(chi_p, nu);
    f_s   = chi_p*eta_s - nu*bft_b(eta_s);
    step=0.5; thr=log(1e-12);
    eL=eta_s; while chi_p*eL-nu*bft_b(eL)-f_s>thr, eL=eL-step; end
    eR=eta_s; while chi_p*eR-nu*bft_b(eR)-f_s>thr, eR=eR+step; end
    eL=eL-2; eR=eR+2;
    np = max(eta_np, round(3*(eR-eL)*400));
    eg = linspace(eL, eR, np);
    lp = chi_p*eg - nu*arrayfun(@bft_b, eg); lp=lp-max(lp);
    pe=exp(lp); de=diff(eg); pm=0.5*(pe(1:end-1)+pe(2:end));
    cg=[0,cumsum(pm.*de/sum(pm.*de))]; cg(end)=1;

    % ---- Draw + streaming Legendre (d_max modes) + KS/AD/CvM -------
    blk_sz = min(500000, max(10000, round(N_QUAL/20)));
    n_blks = ceil(N_QUAL / blk_sz);
    S_leg  = zeros(1, D_MAX);
    u_all  = zeros(N_QUAL, 1);
    ptr    = 0;

    s = cb_stream(chi_p, nu, 'seed', uint64(c*31+7), 'buf_size', 4096);
    [~,~,reg,sig] = s.peek();
    tic;
    for b = 1:n_blks
        bs = min(blk_sz, N_QUAL - (b-1)*blk_sz);
        theta_b = s.draw(bs);
        tc_b    = max(realmin, min(1-eps, theta_b(:)));
        u_b     = interp1(eg, cg, max(eL,min(eR,log(tc_b./(1-tc_b)))), 'pchip', 'extrap');
        u_b     = max(0, min(1, u_b));
        u_all(ptr+1:ptr+bs) = u_b;
        ptr = ptr + bs;
        % Legendre recurrence up to D_MAX (precomputed norms, ~2x vs naive)
        x    = 2*u_b - 1;
        Pk2  = ones(bs,1); Pk1 = x;
        S_leg(1) = S_leg(1) + LEG_NORMS(1)*sum(x);
        for k = 2:D_MAX
            Pk = ((2*k-1)*x.*Pk1 - (k-1)*Pk2) / k;
            S_leg(k) = S_leg(k) + LEG_NORMS(k)*sum(Pk);
            Pk2 = Pk1; Pk1 = Pk;
        end
    end
    t_c = toc;
    s.delete();

    % ---- KS / AD / CvM (sorted PIT) --------------------------------
    u_sorted = sort(u_all);
    i_vec    = (1:N_QUAL)';
    ks       = max(abs(i_vec/N_QUAL - u_sorted));
    ks_pass  = ks < ks_crit;
    u_ad     = max(1e-300, min(1-1e-300, u_sorted));
    ad       = -N_QUAL - mean((2*i_vec-1).*(log(u_ad)+log(1-u_ad(end:-1:1))));
    ad_pass  = ad < ad_crit;
    cvm      = 1/(12*N_QUAL) + sum((u_sorted - (2*i_vec-1)/(2*N_QUAL)).^2);
    cvm_pass = cvm < cvm_crit;
    all_pass = all_pass && ks_pass && ad_pass && cvm_pass;

    % ---- Adaptive Legendre: block energies, d_eff -------------------
    c_hat = S_leg / N_QUAL;
    blk_snr = zeros(1, N_BLOCKS_LEG);
    blk_chi2d = zeros(1, N_BLOCKS_LEG);
    for j = 1:N_BLOCKS_LEG
        lo_j = BLOCK_LO(j); hi_j = BLOCK_HI(j); bs_j = hi_j - lo_j + 1;
        B_j  = sum(c_hat(lo_j:hi_j).^2);
        blk_snr(j)  = (N_QUAL*B_j - bs_j) / sqrt(2*bs_j);
        blk_chi2d(j) = B_j - bs_j/N_QUAL;
    end
    sig_blocks = find(abs(blk_snr) > SNR_STOP);
    if isempty(sig_blocks)
        d_eff = 0; chi2_leg = 0;
    else
        d_eff    = BLOCK_HI(sig_blocks(end));
        chi2_leg = max(sum(blk_chi2d(1:sig_blocks(end))), 0);
    end
    H2_leg  = chi2_leg / 8;
    sig_H2  = sqrt(max(d_eff,1)) / (2*sqrt(2)*N_QUAL);
    leg_snr = chi2_leg / (sig_H2 * 8);

    % ---- Projected Hellinger (QP to nearest valid density) ----------
    % Use d_eff modes only; if d_eff=0 use block 1 (8 modes) as minimum.
    % This avoids projecting noise-dominated high-order coefficients.
    d_proj = max(d_eff, 8);
    if ~isempty(which('quadprog'))
        try
            [~, rho_proj, H2_proj, DB_proj, proj_info] = ...
                leg_proj_hellinger(c_hat(1:d_proj), N_QUAL);
        catch
            rho_proj = NaN; H2_proj = NaN; DB_proj = NaN;
            proj_info.min_density = NaN; proj_info.n_iter = 0;
        end
    else
        rho_proj = NaN; H2_proj = NaN; DB_proj = NaN;
        proj_info.min_density = NaN; proj_info.n_iter = 0;
    end

    % ---- Spacing D_B (Ding-Mullhaupt 2023) -------------------------
    d_pit  = max(diff([0; u_sorted]), 1e-300);
    A_hat  = sum(sqrt(d_pit)) / sqrt(N_QUAL);
    BC_raw = (2/sqrt(pi)) * A_hat;
    DB_raw = -log(max(BC_raw, 1e-300));
    DB     = max(DB_raw, 0);
    H2     = max(1-exp(-DB), 0);
    TVD_hi = sqrt(2*H2);
    above_floor = (DB_raw > 5*sigma_DB);

    % ---- N_cert -----------------------------------------------------
    DB_pt   = max(DB, 1e-300);
    NC_lo   = log(4/3) / (2*DB_pt);
    NC_hi   = log(3)   / (2*DB_pt);
    NC_cert = log(3)   / (2*(max(DB,0) + z99*sigma_DB));

    % ---- Smoothed Hellinger + fitted-Beta CV (optional, O(N log N)) ----
    if DO_HQ_CV
        cv_hq = hq_beta_cv_fixed_sorted(u_sorted);
        H2_smooth = cv_hq.H2_obs;
        H2_cv = cv_hq.H2_cv;
        NC_no_smooth = cv_hq.obs_no;
        NC_some_smooth = cv_hq.obs_some;
        NC_no_cv = cv_hq.cv_no;
        NC_some_cv = cv_hq.cv_some;
        cv_alpha = cv_hq.alpha;
        cv_beta = cv_hq.beta;
        cv_beta_h2_est = cv_hq.H2_beta_est;
        cv_beta_h2_exact = cv_hq.H2_beta_exact;
    else
        H2_smooth=NaN; H2_cv=NaN;
        NC_no_smooth=NaN; NC_some_smooth=NaN; NC_no_cv=NaN; NC_some_cv=NaN;
        cv_alpha=NaN; cv_beta=NaN; cv_beta_h2_est=NaN; cv_beta_h2_exact=NaN;
    end

    % ---- Store -------------------------------------------------------
    R(c).chi=chi_p; R(c).nu=nu; R(c).desc=desc; R(c).regime=reg; R(c).sigma=sig;
    R(c).ks=ks; R(c).ad=ad; R(c).cvm=cvm;
    R(c).ks_pass=ks_pass; R(c).ad_pass=ad_pass; R(c).cvm_pass=cvm_pass;
    R(c).DB_raw=DB_raw; R(c).DB=DB; R(c).H2=H2; R(c).TVD_hi=TVD_hi;
    R(c).above_floor=above_floor;
    R(c).H2_smooth=H2_smooth; R(c).H2_cv=H2_cv;
    R(c).NC_no_smooth=NC_no_smooth; R(c).NC_some_smooth=NC_some_smooth;
    R(c).NC_no_cv=NC_no_cv; R(c).NC_some_cv=NC_some_cv;
    R(c).cv_alpha=cv_alpha; R(c).cv_beta=cv_beta;
    R(c).cv_beta_h2_est=cv_beta_h2_est; R(c).cv_beta_h2_exact=cv_beta_h2_exact;
    R(c).chi2_leg=chi2_leg; R(c).H2_leg=H2_leg; R(c).leg_snr=leg_snr;
    R(c).d_eff=d_eff; R(c).block_snr=blk_snr;
    R(c).rho_proj=rho_proj; R(c).H2_proj=H2_proj; R(c).DB_proj=DB_proj;
    R(c).NC_lo=NC_lo; R(c).NC_hi=NC_hi; R(c).NC_cert=NC_cert;
    R(c).rate=N_QUAL/t_c;  % full pipeline (draw + PIT + Legendre)

    % ---- Sampler-only throughput (draw with no PIT/Legendre) --------
    N_BENCH = min(N_QUAL, 2000000);
    s_bench = cb_stream(chi_p, nu, 'seed', uint64(c*31+999), 'buf_size', 4096);
    tic; s_bench.draw(N_BENCH); t_bench = toc;
    s_bench.delete();
    R(c).rate_draw = N_BENCH / t_bench;

    % ---- Print per-case line ----------------------------------------
    pass_str = @(p) sel(p,'  ok','FAIL');
    fprintf('%-34s %-5s  %5.3f  %6s %6s %6s  %+7.1f  %4d\n', ...
        desc, rnames{reg+1}, sig, ...
        pass_str(ks_pass), pass_str(ad_pass), pass_str(cvm_pass), ...
        leg_snr, d_eff);
end

%% Extended streaming pass (Legendre + bark rates, no sort)
if N_STREAM > 0
    sig_H2_stream = sqrt(D_MAX) / (2*sqrt(2)*N_STREAM);
    blk_sz_s   = 1000000;
    n_blks_s   = ceil(N_STREAM / blk_sz_s);
    B_stream   = n_blks_s;
    bark_sigma = sqrt(KS_ALPHA*(1-KS_ALPHA)/B_stream);
    ks_crit_b  = ks_critical(blk_sz_s, KS_ALPHA);   % exact MTW per block

    fprintf('\n--- Extended streaming (N=%s, d_max=%d, sigma_H2=%.2e) ---\n', ...
        fmt_N(N_STREAM), D_MAX, sig_H2_stream);
    fprintf('  B=%d blocks,  bark rate H0: %.3f +/- %.3f  (+5z: %.3f)\n\n', ...
        B_stream, KS_ALPHA, bark_sigma, KS_ALPHA+5*bark_sigma);
    fprintf('%-34s  %8s  %6s  %4s  |  Bark rates (KS / AD / CvM)\n', ...
        'Case','chi2_deb','Leg_SNR','deff');
    fprintf('%s\n', repmat('-',95,1));

    for c = 1:n_cases
        chi_p=cases{c,1}; nu=cases{c,2}; desc=strtrim(cases{c,3});
        % Rebuild CDF
        eta_s2=cb_mode(chi_p,nu); f_s2=chi_p*eta_s2-nu*bft_b(eta_s2);
        step=0.5; thr=log(1e-12);
        eL2=eta_s2; while chi_p*eL2-nu*bft_b(eL2)-f_s2>thr, eL2=eL2-step; end
        eR2=eta_s2; while chi_p*eR2-nu*bft_b(eR2)-f_s2>thr, eR2=eR2+step; end
        eL2=eL2-2; eR2=eR2+2;
        np2=max(eta_np,round(3*(eR2-eL2)*400));
        eg2=linspace(eL2,eR2,np2); lp2=chi_p*eg2-nu*arrayfun(@bft_b,eg2); lp2=lp2-max(lp2);
        pe2=exp(lp2); de2=diff(eg2); pm2=0.5*(pe2(1:end-1)+pe2(2:end));
        cg2=[0,cumsum(pm2.*de2/sum(pm2.*de2))]; cg2(end)=1;

        S2=zeros(1,D_MAX); N2=0; n_ks_b=0; n_ad_b=0; n_cvm_b=0;
        s2=cb_stream(chi_p,nu,'seed',uint64(c*31+199),'buf_size',4096);
        for b=1:n_blks_s
            bs2=min(blk_sz_s,N_STREAM-(b-1)*blk_sz_s);
            th2=s2.draw(bs2);
            tc2=max(realmin,min(1-eps,th2(:)));
            u2=interp1(eg2,cg2,max(eL2,min(eR2,log(tc2./(1-tc2)))),'pchip','extrap');
            u2=max(0,min(1,u2));
            % Per-block KS/AD/CvM (sort then discard)
            u2s=sort(u2); iv=(1:bs2)';
            ks_b=max(abs(iv/bs2-u2s));
            u2a=max(1e-300,min(1-1e-300,u2s));
            ad_b=-bs2-mean((2*iv-1).*(log(u2a)+log(1-u2a(end:-1:1))));
            cvm_b=1/(12*bs2)+sum((u2s-(2*iv-1)/(2*bs2)).^2);
            if ks_b>ks_crit_b,  n_ks_b=n_ks_b+1;  end
            if ad_b>ad_crit,    n_ad_b=n_ad_b+1;  end
            if cvm_b>cvm_crit,  n_cvm_b=n_cvm_b+1; end
            % Legendre streaming (precomputed norms)
            x2=2*u2-1; Pk2_=ones(bs2,1); Pk1_=x2;
            S2(1)=S2(1)+LEG_NORMS(1)*sum(x2);
            for k=2:D_MAX
                Pk_=((2*k-1)*x2.*Pk1_-(k-1)*Pk2_)/k;
                S2(k)=S2(k)+LEG_NORMS(k)*sum(Pk_);
                Pk2_=Pk1_; Pk1_=Pk_;
            end
            N2=N2+bs2;
        end
        s2.delete();

        % Adaptive order
        c2=S2/N2;
        blk2_snr=zeros(1,N_BLOCKS_LEG); blk2_chi2d=zeros(1,N_BLOCKS_LEG);
        for j=1:N_BLOCKS_LEG
            lo_j=BLOCK_LO(j); hi_j=BLOCK_HI(j); bs_j=hi_j-lo_j+1;
            B_j=sum(c2(lo_j:hi_j).^2);
            blk2_snr(j)=(N2*B_j-bs_j)/sqrt(2*bs_j);
            blk2_chi2d(j)=B_j-bs_j/N2;
        end
        sig2=find(abs(blk2_snr)>SNR_STOP);
        if isempty(sig2), d2=0; chi2_2=0;
        else, d2=BLOCK_HI(sig2(end)); chi2_2=max(sum(blk2_chi2d(1:sig2(end))),0); end
        H2_2=chi2_2/8;
        sig_H2_2=sqrt(max(d2,1))/(2*sqrt(2)*N2);
        snr2=chi2_2/(sig_H2_2*8);

        rate_ks=n_ks_b/B_stream; rate_ad=n_ad_b/B_stream; rate_cvm=n_cvm_b/B_stream;
        z_ks=(rate_ks-KS_ALPHA)/bark_sigma;
        z_ad=(rate_ad-KS_ALPHA)/bark_sigma;
        z_cvm=(rate_cvm-KS_ALPHA)/bark_sigma;

        fprintf('%-34s  %+8.2e  %+6.1f  %4d  |  KS:%.3f(%+4.1fz) AD:%.3f(%+4.1fz) CvM:%.3f(%+4.1fz)\n', ...
            desc, chi2_2, snr2, d2, ...
            rate_ks,z_ks, rate_ad,z_ad, rate_cvm,z_cvm);
    end
    fprintf('\n  sigma_H2 at N=%s: %.2e\n', fmt_N(N_STREAM), sig_H2_stream);
end

%% Detailed metric table
fprintf('\n%-34s  %8s  %8s  %7s  %8s  %7s  %7s  %4s\n', ...
    'Case','KS','AD','CvM','D_B(raw)','H^2_leg','H2_proj','deff');
fprintf('%s\n', repmat('-',98,1));
for c=1:n_cases
    r=R(c);
    flag=''; if r.above_floor, flag='  [D_B real]'; end
    if ~r.ks_pass||~r.ad_pass||~r.cvm_pass, flag=[flag ' *ord_fail']; end
    bspec = sprintf('[%+.0f', r.block_snr(1));
    for j=2:N_BLOCKS_LEG, bspec=[bspec sprintf(',%+.0f',r.block_snr(j))]; end
    bspec = [bspec ']'];
    h2p_str = '  n/a  ';
    if ~isnan(r.H2_proj), h2p_str = sprintf('%7.2e', r.H2_proj); end
    fprintf('%-34s  %8.5f  %8.3f  %7.4f  %+8.2e  %7.2e  %s  %4d%s\n', ...
        strtrim(r.desc), r.ks, r.ad, r.cvm, r.DB_raw, r.H2_leg, h2p_str, r.d_eff, flag);
    fprintf('  block_snr: %s\n', bspec);
end

%% Summary
fprintf('\n%s\n\n', repmat('=',90,1));

n_ks_fail=sum(~[R.ks_pass]); n_ad_fail=sum(~[R.ad_pass]); n_cvm_fail=sum(~[R.cvm_pass]);
fprintf('ORDER-STATISTIC TESTS  (N=%s, alpha=0.01)\n', fmt_N(N_QUAL));
fprintf('  KS: %d  AD: %d  CvM: %d  failures\n', n_ks_fail, n_ad_fail, n_cvm_fail);
fprintf('\n');

leg_snr_vals=[R.leg_snr]; d_eff_vals=[R.d_eff];
sig_H2_qual=sqrt(max(d_eff_vals,1))./(2*sqrt(2)*N_QUAL);
fprintf('ADAPTIVE LEGENDRE chi^2  (d_max=%d, SNR_stop=%.1f)\n', D_MAX, SNR_STOP);
fprintf('  d_eff: max=%d  median=%d\n', max(d_eff_vals), round(median(d_eff_vals)));
n_sig=sum(abs(leg_snr_vals)>5); n_bdr=sum(abs(leg_snr_vals)>3 & abs(leg_snr_vals)<=5);
fprintf('  |Leg_SNR|>5 (real):     %d case(s)\n', n_sig);
fprintf('  |Leg_SNR| in (3,5) (borderline): %d case(s)\n', n_bdr);
for c=1:n_cases
    if abs(R(c).leg_snr)>3
        fprintf('    %-36s  SNR=%+.1f  d_eff=%d  H^2=%.2e\n', ...
            strtrim(R(c).desc), R(c).leg_snr, R(c).d_eff, R(c).H2_leg);
    end
end
fprintf('\n');

DB_vals=[R.DB]; above=[R.above_floor];
fprintf('HELLINGER CHAIN  D_B(raw) max=%+.2e  (5-sigma=%.2e)\n', max(DB_vals), 5*sigma_DB);
if any(above)
    for c=1:n_cases
        if R(c).above_floor
            fprintf('  Real D_B: %-36s  D_B=%.2e(%.0fsig)  H^2=%.2e  TVD<=%.4f\n',...
                strtrim(R(c).desc),R(c).DB,R(c).DB/sigma_DB,R(c).H2,R(c).TVD_hi);
        end
    end
end
fprintf('\n');

ks_n=[R.ks]*sqrt(N_QUAL);
fprintf('KS*sqrt(N): range [%.3f, %.3f]  median %.3f\n\n', min(ks_n),max(ks_n),median(ks_n));

if DO_HQ_CV
    fprintf('FITTED-BETA CV SMOOTHED HELLINGER CERTIFICATION\n');
    fprintf('%-34s  %10s  %10s  %10s  %10s  %10s\n', ...
        'Case','H2_smooth','H2_CV','no_CV','some_CV','BetaFit');
    fprintf('%s\n', repmat('-',95,1));
    for c=1:n_cases
        r=R(c);
        fprintf('%-34s  %10.3e  %10.3e  %10s  %10s  a=%.3g b=%.3g\n', ...
            strtrim(r.desc), r.H2_smooth, r.H2_cv, fmt_N(r.NC_no_cv), fmt_N(r.NC_some_cv), ...
            r.cv_alpha, r.cv_beta);
    end
    fprintf('\n');
end

fprintf('Throughput (sampler only  /  full pipeline incl. PIT+Legendre):\n');
for reg=1:3
    mask=arrayfun(@(r)r.regime==reg,R); if ~any(mask),continue;end
    rd=[R(mask).rate_draw]/1e6;
    rf=[R(mask).rate]/1e6;
    fprintf('  %-6s  %.1f M/s  [%.1f, %.1f]  /  %.1f M/s  [%.1f, %.1f]\n',...
        rnames{reg+1},median(rd),min(rd),max(rd),median(rf),min(rf),max(rf));
end
fprintf('\n');

%% Verdict (Legendre-primary)
leg_real  = abs(leg_snr_vals) > 5;
leg_bdr   = abs(leg_snr_vals) > 3 & ~leg_real;
leg_clean = ~leg_real & ~leg_bdr;
ord_fail  = ~[R.ks_pass]|~[R.ad_pass]|~[R.cvm_pass];
n_noise_fail = sum(ord_fail & leg_clean);

fprintf('-------------------------------------------------------------\n');
if any(leg_real)
    fprintf('RESULT: REAL APPROXIMATION ERROR(S) DETECTED\n');
    fprintf('  Legendre |SNR| > 5 in %d case(s):\n', sum(leg_real));
    for c=1:n_cases
        if leg_real(c)
            fprintf('    %-36s  SNR=%+.1f  d_eff=%d  H^2=%.2e\n', ...
                strtrim(R(c).desc), R(c).leg_snr, R(c).d_eff, R(c).H2_leg);
        end
    end
    if n_noise_fail>0
        fprintf('  (%d order-statistic failure(s) are sampling noise -- Leg<3)\n', n_noise_fail);
    end
elseif any(leg_bdr)
    fprintf('RESULT: BORDERLINE -- run cb_validate_N_stream=1e9 to confirm\n');
    for c=1:n_cases
        if leg_bdr(c)
            fprintf('  %-36s  SNR=%+.1f\n', strtrim(R(c).desc), R(c).leg_snr);
        end
    end
    if n_noise_fail>0
        fprintf('  (%d order-statistic failure(s) are sampling noise -- Leg<3)\n', n_noise_fail);
    end
elseif n_noise_fail>0
    fprintf('RESULT: VALIDATION PASSED  (order-stat failures are sampling noise)\n');
    fprintf('  All |Leg_SNR| < 3.  %d KS/AD/CvM failure(s) are noise at N=%s.\n', ...
        n_noise_fail, fmt_N(N_QUAL));
    fprintf('  Sampler certified correct at N=%s by Legendre chi^2.\n', fmt_N(N_QUAL));
else
    fprintf('RESULT: VALIDATION PASSED\n');
    fprintf('  All KS/AD/CvM pass.  All |Leg_SNR| < 3.\n');
    fprintf('  Sampler certified correct at N=%s.\n', fmt_N(N_QUAL));
end
fprintf('-------------------------------------------------------------\n');
fprintf('\nDone: %s\n\n', datestr(now));

%% Helpers
function s=sel(c,a,b); if c,s=a;else,s=b;end; end
function out=hq_beta_cv_fixed_sorted(u_sorted)
    n = numel(u_sorted);
    points_per_cell = max(8, round(sqrt(n+1)));
    H2_obs = hq_smoothed_sorted(u_sorted, points_per_cell);
    [a,b] = hq_fit_beta_moments(u_sorted);
    beta_ref = sort(betarnd(a, b, n, 1));
    H2_beta_est = hq_smoothed_sorted(beta_ref, points_per_cell);
    H2_beta_exact = hq_beta_uniform_h2(a,b);
    H2_cv = min(max(H2_obs - (H2_beta_est - H2_beta_exact), 0), 1);
    [obs_no, obs_some] = hq_classifier_scales(H2_obs);
    [cv_no, cv_some] = hq_classifier_scales(H2_cv);
    out = struct('H2_obs',H2_obs,'H2_cv',H2_cv, ...
                 'alpha',a,'beta',b, ...
                 'H2_beta_est',H2_beta_est,'H2_beta_exact',H2_beta_exact, ...
                 'obs_no',obs_no,'obs_some',obs_some, ...
                 'cv_no',cv_no,'cv_some',cv_some);
end
function H2=hq_smoothed_sorted(u_sorted, points_per_cell)
    n = numel(u_sorted);
    M = n + 1;
    intervals = max(diff([0; u_sorted(:); 1]), 1e-300);
    h2_raw = 0;
    h2_null = 0;
    for start = 1:points_per_cell:M
        stop = min(start + points_per_cell - 1, M);
        k = stop - start + 1;
        w = k / M;
        slope = max(mean(intervals(start:stop)) * M, 1e-300);
        root = sqrt(slope);
        h2_raw = h2_raw + 0.5 * w * (root - 1)^2;
        h2_null = h2_null + hq_uniform_group_null_h2(M, k);
    end
    H2 = min(max(h2_raw - h2_null, 0), 1);
end
function h=hq_uniform_group_null_h2(M,k)
    if k == M
        eroot = 1;
    else
        log_eroot = 0.5*log(M/k) + gammaln(k+0.5) + gammaln(M) ...
            - gammaln(M+0.5) - gammaln(k);
        eroot = exp(log_eroot);
    end
    h = (k/M) * (1 - eroot);
end
function [a,b]=hq_fit_beta_moments(u)
    mu = min(max(mean(u), 1e-12), 1-1e-12);
    v = var(u, 1);
    vmax = mu*(1-mu);
    if v <= 0
        conc = 1e6;
    else
        conc = max(vmax / min(v, vmax*(1-1e-12)) - 1, 1e-6);
    end
    a = max(mu*conc, 1e-6);
    b = max((1-mu)*conc, 1e-6);
end
function H2=hq_beta_uniform_h2(a,b)
    log_beta_ab = betaln(a,b);
    log_beta_half = betaln((a+1)/2, (b+1)/2);
    rho = min(max(exp(log_beta_half - 0.5*log_beta_ab), 0), 1);
    H2 = 1 - rho;
end
function [n_no,n_some]=hq_classifier_scales(H2)
    H2 = min(max(H2, 0), 1);
    if H2 <= 0
        n_no = 1e300; n_some = 1e300; return;
    end
    rho = 1 - H2;
    if rho <= 0
        n_no = 0; n_some = 1; return;
    end
    lr = log(rho);
    n_no = max(floor(log(sqrt(0.75)) / lr), 0);
    n_some = max(ceil(log(0.5) / lr), 1);
end
function s=fmt_N(n)
    if isnan(n),s='NaN';
    elseif isinf(n)||n>=1e100,s='Inf';
    elseif n>=1e12,s=sprintf('%.1eT',n);
    elseif n>=1e9,s=sprintf('%.1fG',n/1e9);
    elseif n>=1e6,s=sprintf('%.1fM',n/1e6);
    elseif n>=1e3,s=sprintf('%.1fk',n/1e3);
    else,s=sprintf('%d',round(n));end
end
