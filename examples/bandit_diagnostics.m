%% bandit_diagnostics.m
%
% Post-hoc diagnostics for beta_vs_cb_bandit.m results.
% Uses sel_cb_all, sel_beta_all saved during the actual bandit run —
% NOT a re-run, which would use different random state.
%
% MIT License.  Andrew Mullhaupt, Stony Brook University AMS/QF, 2026.

if ~exist('regret_cb','var') || ~exist('sel_cb_all','var')
    error('Run beta_vs_cb_bandit first (with selection recording).');
end

[N_trials, T] = size(regret_cb);
K             = numel(theta_true);
t_vec         = 1:T;

cr_cb   = cumsum(mean(regret_cb,   1));
cr_beta = cumsum(mean(regret_beta, 1));

col_cb   = [0.15 0.45 0.85];
col_beta = [0.85 0.30 0.15];
cmap     = lines(K);

%% =========================================================================
%% (1) Log-log cumulative regret
%% =========================================================================
t_ref   = logspace(1, log10(T), 300);
c_log   = cr_cb(end) / log(T);
c_lin   = cr_beta(end) / T;

figure('Name','Log-log regret','Position',[100 100 820 520]);
loglog(t_vec, cr_cb,   'Color',col_cb,   'LineWidth',2.5); hold on;
loglog(t_vec, cr_beta, 'Color',col_beta, 'LineWidth',2.5);
loglog(t_ref, c_log * log(t_ref),  'k--', 'LineWidth',1.2);
loglog(t_ref, c_lin * t_ref,        'k-.', 'LineWidth',1.2);
xlabel('Round (log scale)');  ylabel('Cumulative regret (log scale)');
title(sprintf('Log-log regret (K=%d, %d trials)', K, N_trials));
legend({'CB exact','Beta MM','O(log T) ref','O(T) ref'}, 'Location','northwest');
grid on;

%% =========================================================================
%% (2) Instantaneous regret rate — smoothed mean across all trials
%% =========================================================================
W = 500;
kern      = ones(1,W)/W;
inst_cb   = conv(mean(regret_cb,   1), kern, 'same');
inst_beta = conv(mean(regret_beta, 1), kern, 'same');

figure('Name','Instantaneous regret','Position',[100 640 820 380]);
semilogx(t_vec, inst_cb,   'Color',col_cb,   'LineWidth',2); hold on;
semilogx(t_vec, inst_beta, 'Color',col_beta, 'LineWidth',2);
xlabel('Round (log scale)');
ylabel(sprintf('Mean inst. regret (W=%d smoother)',W));
title('Instantaneous regret rate — decaying => sublinear; flat => linear');
legend({'CB exact','Beta MM'}, 'Location','northeast');
yline(0,'k:'); grid on;

%% =========================================================================
%% (3) Arm 1 (optimal) selection rate over time — from actual runs
%% =========================================================================
W2   = 1000;
kern2 = ones(1,W2)/W2;

% Fraction of trials selecting arm 1 at each round
opt_cb_t   = mean(sel_cb_all   == 1, 1);   % [1 x T]
opt_beta_t = mean(sel_beta_all == 1, 1);

opt_cb_sm   = conv(opt_cb_t,   kern2, 'same');
opt_beta_sm = conv(opt_beta_t, kern2, 'same');

figure('Name','Arm 1 selection rate','Position',[940 100 820 420]);
semilogx(t_vec, opt_cb_sm,   'Color',col_cb,   'LineWidth',2); hold on;
semilogx(t_vec, opt_beta_sm, 'Color',col_beta, 'LineWidth',2);
yline(1/K, 'k--', 'LineWidth',1);
xlabel('Round (log scale)');
ylabel(sprintf('Arm 1 selection rate (W=%d smoother)', W2));
title('Optimal arm (arm 1) selection rate — actual runs');
legend({'CB exact','Beta MM','Uniform 1/K'}, 'Location','southeast');
ylim([0 1.05]); grid on;

%% =========================================================================
%% (4) All arm selection frequencies: early vs late
%% =========================================================================
early = 1:min(500,T);
late  = max(1,T-4999):T;

freq_cb_early   = mean(sel_cb_all(:,early)   == permute(1:K,[1 3 2]), [1 2]);
freq_beta_early = mean(sel_beta_all(:,early) == permute(1:K,[1 3 2]), [1 2]);
freq_cb_late    = mean(sel_cb_all(:,late)    == permute(1:K,[1 3 2]), [1 2]);
freq_beta_late  = mean(sel_beta_all(:,late)  == permute(1:K,[1 3 2]), [1 2]);

fprintf('\n--- Arm selection frequencies (averaged across %d trials) ---\n', N_trials);
fprintf('%-14s  %8s  %8s  |  %8s  %8s\n', '', 'CB early', 'Beta early', 'CB late', 'Beta late');
fprintf('%-14s  %8s  %8s  |  %8s  %8s\n', '', sprintf('(t<=%-d)',max(early)), ...
    sprintf('(t<=%-d)',max(early)), sprintf('(t>%d)',min(late)), sprintf('(t>%d)',min(late)));
fprintf('%s\n', repmat('-',70,1));
for k = 1:K
    fprintf('arm %d (θ=%.2f)   %7.3f%%  %7.3f%%  |  %7.3f%%  %7.3f%%  %s\n', ...
        k, theta_true(k), ...
        100*freq_cb_early(k), 100*freq_beta_early(k), ...
        100*freq_cb_late(k),  100*freq_beta_late(k), ...
        sel(k==1,'<-- optimal',''));
end

fprintf('\n--- Regret breakdown ---\n');
fprintf('Total cumulative regret at T=%d:\n', T);
fprintf('  CB   : %.1f  (%.3f per round)\n', cr_cb(end),   cr_cb(end)/T);
fprintf('  Beta : %.1f  (%.3f per round)\n', cr_beta(end), cr_beta(end)/T);
fprintf('\nRegret in first %d rounds:\n', max(early));
fprintf('  CB   : %.1f\n', sum(mean(regret_cb(:,early),   1)));
fprintf('  Beta : %.1f\n', sum(mean(regret_beta(:,early), 1)));
fprintf('\nRegret in last %d rounds:\n', numel(late));
fprintf('  CB   : %.1f\n', sum(mean(regret_cb(:,late),   1)));
fprintf('  Beta : %.1f\n', sum(mean(regret_beta(:,late), 1)));

function s = sel(c, a, b)
    if c, s = a; else, s = b; end
end
