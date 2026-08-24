% cb_hellinger_invariant_test.m
% Systematic test: verify H2_h <= H2_mm <= H2_cheesy for all (Z, nu).
% Run from cb_sampler root.

repo_root = fileparts(fileparts(mfilename('fullpath')));
addpath(repo_root);
addpath(fullfile(repo_root, 'src', 'matlab'));

Z_vals  = [0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.60 0.70 0.80 0.90 0.95];
nu_vals = [5 10 20 50 100 200];

n_violations = 0;
n_cases      = 0;
worst_h_mm   = 0;   % max(H2_h - H2_mm)

fprintf('Invariant check: H2_hellinger <= H2_moment_matched <= H2_cheesy\n');
fprintf('%-20s  %8s  %8s  %8s  %s\n', 'case', 'H2_chsy', 'H2_mm', 'H2_h', 'status');

for iz = 1:numel(Z_vals)
    for in_ = 1:numel(nu_vals)
        Z = Z_vals(iz); nu_ = nu_vals(in_);
        chi_ = Z * nu_;
        if chi_ <= 0.1 || chi_ >= nu_ - 0.1, continue; end
        n_cases = n_cases + 1;

        % Hellinger-optimal
        [~, ~, ~, H2_h] = cb_to_beta_hellinger(chi_, nu_, 'max_iter', 400);

        % Moment-matched and cheesy via same grid
        [H2_ch, H2_mm] = compute_h2_ch_mm(chi_, nu_);

        ok_h_mm  = H2_h  <= H2_mm  + 1e-4;   % allow tiny numerical slack
        ok_mm_ch = H2_mm <= H2_ch  + 1e-4;
        ok = ok_h_mm && ok_mm_ch;

        margin_h_mm = H2_h - H2_mm;
        if margin_h_mm > worst_h_mm, worst_h_mm = margin_h_mm; end

        if ~ok || margin_h_mm > 5e-4
            flag = 'VIOLATION';
            if ok && margin_h_mm > 5e-4, flag = 'warn(close)'; end
            fprintf('Z=%4.2f nu=%3d: %8.4f  %8.4f  %8.4f  %s\n', ...
                Z, nu_, H2_ch, H2_mm, H2_h, flag);
            if ~ok, n_violations = n_violations + 1; end
        end
    end
end

if n_violations == 0
    fprintf('\nPASS: all %d cases satisfy H2_h <= H2_mm <= H2_cheesy (tol 1e-4)\n', n_cases);
else
    fprintf('\nFAIL: %d/%d violations\n', n_violations, n_cases);
end
fprintf('Worst H2_h - H2_mm across grid: %.4e\n', worst_h_mm);

%% Helper: compute H2 for cheesy and moment-matched Betas via grid formula
function [H2_ch, H2_mm] = compute_h2_ch_mm(chi_, nu_)
    eta_s = cb_mode(chi_, nu_);
    [~,~,bpp] = bft_all(eta_s);
    fp = chi_*eta_s - nu_*bft_b(eta_s);
    st = max(1/sqrt(max(nu_*bpp,1e-6)), 0.5);
    eL=eta_s; while chi_*eL-nu_*bft_b(eL)>fp-35, eL=eL-st; end
    eR=eta_s; while chi_*eR-nu_*bft_b(eR)>fp-35, eR=eR+st; end
    eL=eL-st; eR=eR+st;
    eg=linspace(eL,eR,2000)'; de=eg(2)-eg(1);
    ls=-log1p(exp(-eg)); ls1=-log1p(exp(eg));
    lf=(chi_*eg-nu_*bft_b(eg))/2; lf=lf-max(lf); f0=exp(lf);
    Zf=sum(f0.^2)*de;

    a_ch = chi_+1; b_ch = nu_-chi_+1;
    [a_mm, b_mm] = cb_to_beta_num(chi_, nu_);

    H2_ch = max(0, 1 - grid_rho(a_ch,b_ch,lf,ls,ls1,de,Zf));
    if isnan(a_mm)||isnan(b_mm)||a_mm<=0||b_mm<=0
        H2_mm = H2_ch;
    else
        H2_mm = max(0, 1 - grid_rho(a_mm,b_mm,lf,ls,ls1,de,Zf));
    end
end

function rv = grid_rho(a,b,lf,ls,ls1,de,Zf)
    lg=lf+(a/2)*ls+(b/2)*ls1; lmg=max(lg); g=exp(lg-lmg); S0=sum(g)*de;
    if max(a,b)>1e8
        if b>a, bn=gammaln(a)-(a*log(b)+a*(a-1)/(2*b));
        else,   bn=gammaln(b)-(b*log(a)+b*(b-1)/(2*a)); end
    else
        bn=betaln(a,b);
    end
    rv = max(0, min(1, exp(lmg+log(S0)-0.5*log(Zf)-0.5*bn)));
end
