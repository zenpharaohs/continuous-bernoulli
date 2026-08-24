%% beta_vs_cb_bandit.m
%
% Example: CB-exact vs Beta-approximate Thompson sampling on a CB bandit.
%
% BACKGROUND
%   The Continuous Bernoulli (CB) distribution is the correct likelihood
%   for arms whose rewards are continuous on [0,1].  The conjugate posterior
%   is a CB conjugate family -- NOT a Beta distribution.  A common shortcut
%   is to approximate the CB posterior with a Beta distribution.
%
%   PARAMETERIZATION NOTE (critical):
%     Arms are specified by their EXPECTED LOSS mu_k = E[X_k].
%     The CB canonical parameter theta_k = sigmoid(eta_k) satisfies
%       mu_k = B'(eta_k)  (mean value parameterization)
%     where B is the CB log-partition function.
%     theta_k != mu_k (except at mu=0.5).  See cb_theta_to_mean.m.
%
%     Example: arm with mu=0.10 (expected loss 0.10) has
%       theta = cb_mean_to_theta(0.10) ~ 0.0085 (the CB parameter)
%     As the posterior collects data, it concentrates near theta ~ 0.0085,
%     NOT near 0.10.
%
%   REWARDS:
%     Rewards are drawn from the CB distribution: x ~ CB(theta_k).
%     This uses the exact inverse-CDF sampler cb_data_sample.
%     (Previous version used Bernoulli draws, which is misspecified for CB.)
%
%   THREE COMPETITORS:
%     (A) CB-exact Thompson:         cb_stream posterior, exact draws
%     (B) Cheesy Beta Thompson:      Beta(chi+1, nu-chi+1), fast but wrong
%     (C) Hellinger-optimal Beta:    best possible Beta under Hellinger metric
%                                    (most principled Beta, slowest)
%
%   PARTS:
%     Part 1: Hellinger distance map H^2(CB posterior, best Beta)
%             Confirms structural mismatch -- even best Beta is poor.
%     Part 2: Thompson sampling regret comparison.
%
% MIT License.  Andrew Mullhaupt, Stony Brook University AMS/QF, 2026.

%% Path setup
cb_root = fileparts(fileparts(mfilename('fullpath')));
addpath(cb_root);
addpath(fullfile(cb_root, 'src', 'matlab'));

mex_file = fullfile(cb_root, ['cb_stream_mex.' mexext]);
if ~exist(mex_file, 'file')
    fprintf('Building cb_stream_mex...\n');
    old = cd(cb_root);  run(fullfile(cb_root, 'build_cb_stream.m'));  cd(old);
end

rng(137);

%% =========================================================================
%% PART 1: Hellinger distance map
%% =========================================================================
fprintf('=== Part 1: Hellinger distance map H^2(CB, best Beta) ===\n\n');
fprintf('NOTE: Z = chi/nu = observed mean = MEAN VALUE PARAMETER mu.\n');
fprintf('      The CB arm parameter theta = cb_mean_to_theta(Z) is different.\n\n');

Z_vals  = linspace(0.03, 0.50, 50);
nu_vals = [5, 10, 20, 50, 100, 200, 500, 1000];
H2_map  = zeros(numel(Z_vals), numel(nu_vals));

for iz = 1:numel(Z_vals)
    for in_ = 1:numel(nu_vals)
        H2_map(iz, in_) = hellinger_cb_beta_mm(Z_vals(iz) * nu_vals(in_), nu_vals(in_));
    end
    if mod(iz, 10) == 0
        fprintf('  Z grid: %d/%d\n', iz, numel(Z_vals));
    end
end

figure('Name','Hellinger Map: CB vs Best-Fit Beta','Position',[100 100 820 520]);
imagesc(1:numel(nu_vals), Z_vals, log10(max(H2_map, 1e-10)));
cb_h = colorbar;  cb_h.Label.String = 'log_{10} H^2';
colormap(flipud(hot));  clim([-8, -1]);
set(gca, 'XTick', 1:numel(nu_vals), ...
    'XTickLabel', arrayfun(@num2str, nu_vals, 'UniformOutput', false));
xlabel('\nu  (number of observations)');
ylabel('\mu = Z = \chi/\nu  (mean value parameter = observed mean)');
title({'H^2(CB posterior, Hellinger-optimal Beta)', ...
       'Red = large discrepancy (Beta family structurally wrong here)'});
hold on;
contour(1:numel(nu_vals), Z_vals, log10(max(H2_map, 1e-10)), ...
    [-2, -3], 'w', 'LineWidth', 1.5, 'ShowText', 'on');
hold off;

[H2_max, idx_max] = max(H2_map(:));
[iz_max, in_max]  = ind2sub(size(H2_map), idx_max);
fprintf('\nH^2 summary (Hellinger-optimal Beta):\n');
fprintf('  Max H^2 = %.2e  at Z=%.2f, nu=%d\n', ...
    H2_max, Z_vals(iz_max), nu_vals(in_max));
fprintf('  H^2 > 1e-2 in %.1f%% of (Z,nu) grid cells\n', ...
    100*mean(H2_map(:) > 1e-2));
fprintf('  H^2 > 1e-3 in %.1f%% of (Z,nu) grid cells\n\n', ...
    100*mean(H2_map(:) > 1e-3));

%% =========================================================================
%% PART 2: Thompson sampling bandit -- CB rewards (correctly specified)
%% =========================================================================
fprintf('=== Part 2: Thompson sampling bandit (CB rewards) ===\n\n');

K        = 8;
T        = 1000;
N_trials = 10;

% Arms specified by EXPECTED LOSS (mean value parameter mu).
% The CB canonical parameter theta_k = cb_mean_to_theta(mu_k).
mu_true   = [0.08, 0.12, 0.18, 0.25, 0.33, 0.42, 0.50, 0.62];
mu_opt    = mu_true(1);
assert(numel(mu_true) == K);

% Corresponding CB canonical parameters
theta_cb  = arrayfun(@cb_mean_to_theta, mu_true);

fprintf('Arms (mean value parameterization):\n');
for k = 1:K
    fprintf('  Arm %d: mu=%.2f  theta=%.4f  E[X|theta]=%.4f\n', ...
        k, mu_true(k), theta_cb(k), cb_theta_to_mean(theta_cb(k)));
end
fprintf('Optimal arm: 1 (mu=%.2f, expected loss %.2f)\n\n', mu_opt, mu_opt);

regret_cb    = zeros(N_trials, T);
regret_ch    = zeros(N_trials, T);   % cheesy Beta
regret_hopt  = zeros(N_trials, T);   % Hellinger-optimal Beta

% Pre-compute Hellinger-optimal Beta parameters at prior (0 pulls):
% Beta(1,1) -- will update on each arm selection.
a_h0 = ones(1,K);
b_h0 = ones(1,K);

for trial = 1:N_trials

    % Pre-generate CB rewards from the correct CB likelihood
    % x_{k,t} ~ CB(theta_cb(k)), using exact inverse-CDF sampler
    rewards = zeros(K, T);
    for k = 1:K
        rewards(k,:) = cb_data_sample(theta_cb(k), T);
    end

    % ----- (A) CB-exact Thompson -----
    BLOCK  = 256;
    chi_c  = zeros(1,K);
    nu_c   = zeros(1,K);
    buf_c  = rand(K, BLOCK);
    bpos_c = zeros(1,K);

    for t = 1:T
        draws = zeros(1,K);
        for k = 1:K
            if bpos_c(k) == 0
                chi_k = chi_c(k);  nu_k = nu_c(k);
                if chi_k <= 0 || chi_k >= nu_k
                    a_fb = chi_k + 0.5;  b_fb = nu_k - chi_k + 0.5;
                    buf_c(k,:) = betarnd(a_fb, b_fb, 1, BLOCK);
                else
                    s_tmp = cb_stream(chi_k, nu_k, ...
                        'seed', uint64(trial*100 + k + t*13), ...
                        'buf_size', BLOCK);
                    buf_c(k,:) = s_tmp.draw(BLOCK);
                    s_tmp.delete();
                end
                bpos_c(k) = BLOCK;
            end
            draws(k) = buf_c(k, bpos_c(k));
            bpos_c(k) = bpos_c(k) - 1;
        end
        [~, arm] = min(draws);
        r = rewards(arm, t);
        chi_c(arm) = chi_c(arm) + r;
        nu_c(arm)  = nu_c(arm)  + 1;
        bpos_c(arm) = 0;
        regret_cb(trial, t) = mu_true(arm) - mu_opt;
    end

    % ----- (B) Cheesy Beta Thompson -----
    % Beta(chi+1, nu-chi+1) -- fast but structurally wrong for CB data
    chi_ch = zeros(1,K);
    nu_ch  = zeros(1,K);

    for t = 1:T
        % Parameters: Beta(chi+1, nu-chi+1)
        a_ch = chi_ch + 1;
        b_ch = nu_ch - chi_ch + 1;
        draws = betarnd(a_ch, b_ch);
        [~, arm] = min(draws);
        r = rewards(arm, t);
        chi_ch(arm) = chi_ch(arm) + r;
        nu_ch(arm)  = nu_ch(arm)  + 1;
        regret_ch(trial, t) = mu_true(arm) - mu_opt;
    end

    % ----- (C) Hellinger-optimal Beta Thompson -----
    % Best possible Beta under Hellinger metric -- principled, slower.
    chi_h = zeros(1,K);
    nu_h  = zeros(1,K);
    a_k   = a_h0;
    b_k   = b_h0;

    for t = 1:T
        draws = betarnd(a_k, b_k);
        [~, arm] = min(draws);
        r = rewards(arm, t);
        chi_h(arm) = chi_h(arm) + r;
        nu_h(arm)  = nu_h(arm)  + 1;
        if chi_h(arm) > 0 && chi_h(arm) < nu_h(arm)
            [a_k(arm), b_k(arm)] = cb_to_beta_hellinger(chi_h(arm), nu_h(arm));
        end
        regret_hopt(trial, t) = mu_true(arm) - mu_opt;
    end

    if mod(trial, 2) == 0
        fprintf('  Trial %d/%d done\n', trial, N_trials);
    end
end

%% Plot cumulative regret
cr_cb   = cumsum(mean(regret_cb,   1));
cr_ch   = cumsum(mean(regret_ch,   1));
cr_hopt = cumsum(mean(regret_hopt, 1));
se_cb   = cumsum(std(regret_cb,  [], 1)) / sqrt(N_trials);
se_ch   = cumsum(std(regret_ch,  [], 1)) / sqrt(N_trials);
se_hopt = cumsum(std(regret_hopt,[], 1)) / sqrt(N_trials);

figure('Name','Cumulative Regret -- CB rewards','Position',[940 100 860 520]);
t_vec     = 1:T;
col_cb    = [0.10 0.40 0.80];
col_ch    = [0.85 0.25 0.10];
col_hopt  = [0.10 0.65 0.25];

shf = @(cr,se) [cr+se, fliplr(cr-se)];
fill([t_vec,fliplr(t_vec)], shf(cr_cb,se_cb),    col_cb,   'FaceAlpha',0.18,'EdgeColor','none'); hold on;
fill([t_vec,fliplr(t_vec)], shf(cr_ch,se_ch),    col_ch,   'FaceAlpha',0.18,'EdgeColor','none');
fill([t_vec,fliplr(t_vec)], shf(cr_hopt,se_hopt),col_hopt, 'FaceAlpha',0.18,'EdgeColor','none');
plot(t_vec, cr_cb,   'Color',col_cb,   'LineWidth',2.0, 'DisplayName','CB exact');
plot(t_vec, cr_ch,   'Color',col_ch,   'LineWidth',2.0, 'DisplayName','Cheesy Beta');
plot(t_vec, cr_hopt, 'Color',col_hopt, 'LineWidth',2.0, 'DisplayName','Hellinger-optimal Beta');
hold off;
xlabel('Round'); ylabel('Cumulative regret');
title(sprintf('Thompson sampling on CB bandit  (K=%d arms, %d trials)', K, N_trials));
legend('Location','northwest'); grid on;

fprintf('\nFinal cumulative regret at T=%d:\n', T);
fprintf('  CB exact             : %6.1f\n', cr_cb(end));
fprintf('  Cheesy Beta          : %6.1f  (+%.1f%% vs CB)\n', ...
    cr_ch(end),   100*(cr_ch(end)/cr_cb(end)-1));
fprintf('  Hellinger-opt Beta   : %6.1f  (+%.1f%% vs CB)\n', ...
    cr_hopt(end), 100*(cr_hopt(end)/cr_cb(end)-1));
fprintf('\nEven the best possible Beta costs %.1f%% more regret than exact CB.\n', ...
    100*(cr_hopt(end)/cr_cb(end)-1));

%% =========================================================================
%% Helper: Hellinger-optimal H^2 (for Part 1 map -- uses cb_to_beta_hellinger)
%% =========================================================================
function H2 = hellinger_cb_beta_mm(chi_p, nu)
    if nu <= 0 || chi_p <= 0 || chi_p >= nu
        H2 = NaN; return;
    end
    [~,~,~,H2] = cb_to_beta_hellinger(chi_p, nu, 'max_iter', 200);
end
