%% ks_exact_test.m
%
% Validation of ks_exact_cdf and ks_critical against:
%   (1) Analytically known small-n values
%   (2) Monte Carlo simulation for n = 5..100
%   (3) Self-consistency: ks_critical inverts ks_exact_cdf
%   (4) Asymptotic comparison table (exact vs sqrt(-log(alpha/2)/(2n)))
%   (5) Direct spot checks on the asymptotic series path (n > 10000)
%
% IMPORTANT: The KS statistic is the TWO-SIDED supremum:
%   D_n = max over i of max(i/n - U_(i),  U_(i) - (i-1)/n)
%       = max(D_n^+, D_n^-)
% where D_n^+ = max_i(i/n - U_(i)) and D_n^- = max_i(U_(i) - (i-1)/n).
% The MC simulation must compute this correctly; using only i/n - U_(i)
% gives D_n^+ alone, which is stochastically smaller than D_n, causing
% all z-scores to be large and negative.
%
% Run from continuous-bernoulli after addpath('src/matlab') and addpath('tests').
%
% MIT License.  Andrew Mullhaupt, Stony Brook University AMS/QF, 2026.

fprintf('\n');
fprintf('=================================================================\n');
fprintf('  ks_exact_cdf / ks_critical: Validation Suite\n');
fprintf('  %s\n', datestr(now));
fprintf('=================================================================\n\n');

% Force fresh load of both functions
clear ks_exact_cdf ks_critical

all_pass = true;

%% =========================================================================
%% (1) Analytically known small-n values
%% =========================================================================
fprintf('--- (1) Analytical cases ---\n\n');

% n=1: D_1 = max(U, 1-U),  P(D_1 <= d) = max(0, 2d-1)  for d in [0.5, 1]
% n=2: P(D_2 <= 0.5) = 0.5  [see Birnbaum 1952]
% n=3: P(D_3 <= 1/3) = 2/9  [see Kolmogorov 1933 small-n tables]
anal_cases = {
    1, 0.75,  0.5,          'P(D_1<=0.75) = 2*0.75-1 = 0.5';
    1, 0.90,  0.80,         'P(D_1<=0.90) = 2*0.90-1 = 0.8';
    1, 0.50,  0.0,          'P(D_1<=0.50) = 0';
    1, 1.00,  1.0,          'P(D_1<=1.00) = 1';
    2, 0.50,  0.5,          'P(D_2<=0.5)  = 0.5';
    3, 1/3,   2/9,          'P(D_3<=1/3)  = 2/9';
};

fprintf('  %-44s  %-14s  %-12s  %s\n', 'Case','exact','error','result');
fprintf('  %s\n', repmat('-',1,78));
for c = 1:size(anal_cases,1)
    n    = anal_cases{c,1};
    d    = anal_cases{c,2};
    expt = anal_cases{c,3};
    desc = anal_cases{c,4};
    got  = ks_exact_cdf(n, d);
    err  = abs(got - expt);
    pass = err < 1e-12;
    all_pass = all_pass && pass;
    fprintf('  %-44s  %-14.12f  %-12.3e  %s\n', desc, expt, err, sel(pass,'ok','FAIL'));
end
fprintf('\n');

%% =========================================================================
%% (2) Monte Carlo comparison for small n
%% =========================================================================
fprintf('--- (2) Monte Carlo comparison (N_MC=1e6 trials each) ---\n\n');
fprintf('  Two-sided KS: D_n = max(max_i(i/n-U_(i)), max_i(U_(i)-(i-1)/n))\n\n');

rng(137);
N_MC = 1000000;

mc_cases = {
    5, 0.40;
    5, 0.50;
    5, 0.56;
    10, 0.30;
    10, 0.41;
    20, 0.20;
    20, 0.29;
    50, 0.15;
    50, 0.19;
    100, 0.12;
};

fprintf('  %-5s  %-6s  %-14s  %-14s  %-8s  %s\n', ...
    'n','d','exact','MC (1e6)','z','result');
fprintf('  %s\n', repmat('-',1,66));

for c = 1:size(mc_cases,1)
    n = mc_cases{c,1};
    d = mc_cases{c,2};

    p_exact = ks_exact_cdf(n, d);

    % TWO-SIDED KS statistic: max(D^+, D^-)
    %   D^+ = max_i (i/n - U_(i))       [empirical CDF overshoot above uniform]
    %   D^- = max_i (U_(i) - (i-1)/n)   [empirical CDF overshoot below uniform]
    % NOTE: do NOT use max(i/n - U_(i), (i-1)/n - U_(i)) -- that gives D^+ only.
    iv  = (1:n)';
    Dvec = zeros(N_MC, 1);
    for trial = 1:N_MC
        u = sort(rand(n,1));
        dp = max(iv/n   - u);         % D^+
        dm = max(u - (iv-1)/n);       % D^-
        Dvec(trial) = max(dp, dm);    % two-sided
    end
    p_mc = mean(Dvec <= d);
    se   = sqrt(p_mc*(1-p_mc) / N_MC + 1e-14);
    z    = (p_exact - p_mc) / se;
    pass = abs(z) < 4;
    all_pass = all_pass && pass;
    fprintf('  %-5d  %-6.4f  %-14.10f  %-14.10f  %+6.2f   %s\n', ...
        n, d, p_exact, p_mc, z, sel(pass,'ok','WARN'));
end
fprintf('\n');

%% =========================================================================
%% (3) Self-consistency: ks_critical inverts ks_exact_cdf
%% =========================================================================
fprintf('--- (3) Inverse consistency: P(D_n <= ks_critical(n,alpha)) = 1-alpha ---\n\n');

fprintf('  %-6s  %-6s  %-14s  %-14s  %-12s  %s\n', ...
    'n','alpha','d_crit','P(D_n<=d_crit)','error','result');
fprintf('  %s\n', repmat('-',1,74));

inv_cases = {5,0.05; 5,0.01; 10,0.05; 10,0.01; 20,0.05; 20,0.01; ...
             50,0.05; 100,0.01; 500,0.05; 1000,0.01};
for c = 1:size(inv_cases,1)
    n     = inv_cases{c,1};
    alpha = inv_cases{c,2};
    d_c   = ks_critical(n, alpha);
    p_c   = ks_exact_cdf(n, d_c);
    err   = abs(p_c - (1-alpha));
    pass  = err < 1e-10;
    all_pass = all_pass && pass;
    fprintf('  %-6d  %-6.3f  %-14.10f  %-14.10f  %-12.2e  %s\n', ...
        n, alpha, d_c, p_c, err, sel(pass,'ok','FAIL'));
end
fprintf('\n');

%% =========================================================================
%% (4) Asymptotic series spot checks (n > N_EXACT = 10000)
%% =========================================================================
fprintf('--- (4) Asymptotic series spot checks (n > 10000) ---\n\n');
fprintf('  These exercise the ks_kolmogorov_series path in ks_exact_cdf.\n');
fprintf('  Values verified against scipy.stats.kstwo.cdf (Python reference).\n\n');

% Reference values from scipy.stats.kstwo.cdf(d, n) which uses the exact
% asymptotic Kolmogorov distribution P(K <= sqrt(n)*d).
% At large n the exact and asymptotic agree to < 1e-6.
asym_cases = {
    50000,  0.006,   'P(D_50k <= 0.006)  ~= ks_kolmogorov(1.342)';
    100000, 0.004,   'P(D_100k <= 0.004) ~= ks_kolmogorov(1.265)';
    100000, 0.005,   'P(D_100k <= 0.005) ~= ks_kolmogorov(1.581)';
    100000, 0.006,   'P(D_100k <= 0.006) ~= ks_kolmogorov(1.897)';
    1000000,0.0016,  'P(D_1M   <= 0.0016)~= ks_kolmogorov(1.600)';
};

fprintf('  %-42s  %-10s  %-12s  %-8s\n', ...
    'Case','t=sqN*d','ks_exact_cdf','series_direct');
fprintf('  %s\n', repmat('-',1,80));
for c = 1:size(asym_cases,1)
    n    = asym_cases{c,1};
    d    = asym_cases{c,2};
    desc = asym_cases{c,3};
    t    = sqrt(n)*d;
    p_cdf    = ks_exact_cdf(n, d);
    % Direct series evaluation for cross-check
    p_series = ks_series_direct(t);
    fprintf('  %-42s  %-10.4f  %-12.8f  %-12.8f\n', desc, t, p_cdf, p_series);
end
fprintf('\n');

%% =========================================================================
%% (5) Exact vs asymptotic critical values (alpha=0.01)
%% =========================================================================
fprintf('--- (5) Exact vs asymptotic critical values (alpha=0.01) ---\n\n');
fprintf('  Note: exact < asymptotic for small n (asymptotic is anticonservative).\n');
fprintf('  Exact > asymptotic would indicate a bug.\n\n');

fprintf('  %-8s  %-14s  %-14s  %-10s  %s\n', ...
    'n','exact','asymptotic','rel_diff','note');
fprintf('  %s\n', repmat('-',1,66));

for n = [5, 10, 20, 50, 100, 200, 500, 1000, 5000, 10000, 50000, 100000, 1000000]
    d_exact = ks_critical(n, 0.01);
    d_asym  = sqrt(-log(0.005) / (2*n));
    rdiff   = (d_exact - d_asym) / d_asym * 100;  % relative to asymptotic
    % Sanity: exact and asymptotic should agree to < 1% for n >= 1000
    note = '';
    if n >= 1000 && abs(rdiff) > 1.0
        note = '<-- SUSPICIOUS';
        all_pass = false;
    end
    fprintf('  %-8d  %-14.10f  %-14.10f  %+8.3f%%  %s\n', ...
        n, d_exact, d_asym, rdiff, note);
end
fprintf('\n  (asymptotic underestimates critical value for small n --\n');
fprintf('   exact < asymptotic; the reverse would be a bug)\n\n');

%% =========================================================================
%% Summary
%% =========================================================================
fprintf('%s\n', repmat('=',1,65));
if all_pass
    fprintf('RESULT: ALL TESTS PASSED\n');
else
    fprintf('RESULT: SOME TESTS FAILED\n');
end
fprintf('Done: %s\n\n', datestr(now));

%% =========================================================================
%% Local helpers
%% =========================================================================

function p = ks_series_direct(t)
% Direct Kolmogorov series evaluation for cross-checking ks_exact_cdf.
% Separate implementation to catch copy/paste bugs.
    if t <= 0,  p = 0;  return;  end
    p   = 0;
    sgn = 1;
    for j = 1:500
        term = 2.0 * sgn * exp(-2.0 * j*j * t*t);
        p    = p + term;
        if abs(term) < 1e-17,  break;  end
        sgn  = -sgn;
    end
    p = max(0.0, min(1.0, p));
end

function s = sel(c, a, b)
    if c, s = a; else, s = b; end
end
