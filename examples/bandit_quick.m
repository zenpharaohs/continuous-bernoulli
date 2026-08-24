%% bandit_quick.m
%
% Quick comparison: CB-exact vs Beta-MM Thompson sampling.
% N_trials=4, T=1000.  No Hellinger map.  Should run in ~30 seconds.
%
% MIT License.  Andrew Mullhaupt, Stony Brook University AMS/QF, 2026.

rng(137);

K        = 8;
T        = 10000;
N_trials = 20;

theta_true = [0.05, 0.10, 0.18, 0.25, 0.33, 0.42, 0.50, 0.62];
theta_opt  = theta_true(1);

fprintf('Arms: ');  fprintf('theta=%.2f  ', theta_true);  fprintf('\n');
fprintf('Optimal arm: 1 (theta=%.2f),  T=%d,  trials=%d\n\n', theta_opt, T, N_trials);

regret_cb    = zeros(N_trials, T);
regret_beta  = zeros(N_trials, T);
sel_cb_all   = zeros(N_trials, T);
sel_beta_all = zeros(N_trials, T);

BLOCK = 256;

for trial = 1:N_trials
    rewards = rand(K, T) < repmat(theta_true(:), 1, T);

    %% --- CB-exact Thompson ---
    chi_c  = zeros(1,K);
    nu_c   = zeros(1,K);
    buf_c  = rand(K, BLOCK);
    bpos_c = zeros(1,K);

    for t = 1:T
        draws = zeros(1,K);
        for k = 1:K
            if bpos_c(k) == 0
                chi_k = chi_c(k);  nu_k = nu_c(k);
                if chi_k == 0 || chi_k >= nu_k
                    a_fb = chi_k + 0.5;  b_fb = nu_k - chi_k + 0.5;
                    buf_c(k,:) = betarnd(a_fb, b_fb, 1, BLOCK);
                else
                    s_tmp = cb_stream(chi_k, nu_k, ...
                        'seed', uint64(trial*1000 + k + t*13), ...
                        'buf_size', BLOCK);
                    buf_c(k,:) = s_tmp.draw(BLOCK);
                    s_tmp.delete();
                end
                bpos_c(k) = BLOCK;
            end
            draws(k)  = buf_c(k, bpos_c(k));
            bpos_c(k) = bpos_c(k) - 1;
        end
        [~, arm] = min(draws);
        r = rewards(arm, t);
        chi_c(arm) = chi_c(arm) + r;
        nu_c(arm)  = nu_c(arm)  + 1;
        bpos_c(arm) = 0;
        regret_cb(trial, t)  = theta_true(arm) - theta_opt;
        sel_cb_all(trial, t) = arm;
    end

    %% --- Beta-MM Thompson ---
    chi_b = zeros(1,K);  nu_b = zeros(1,K);
    a_k   = ones(1,K);   b_k  = ones(1,K);

    for t = 1:T
        draws    = betarnd(a_k, b_k);
        [~, arm] = min(draws);
        r = rewards(arm, t);
        chi_b(arm) = chi_b(arm) + r;
        nu_b(arm)  = nu_b(arm)  + 1;
        if chi_b(arm) > 0 && chi_b(arm) < nu_b(arm)
            [a_k(arm), b_k(arm)] = cb_to_beta_num(chi_b(arm), nu_b(arm));
        else
            a_k(arm) = chi_b(arm) + 0.5;
            b_k(arm) = nu_b(arm) - chi_b(arm) + 0.5;
        end
        regret_beta(trial, t)  = theta_true(arm) - theta_opt;
        sel_beta_all(trial, t) = arm;
    end

    fprintf('  Trial %d/%d done\n', trial, N_trials);
end

%% Results
cr_cb   = cumsum(mean(regret_cb,   1));
cr_beta = cumsum(mean(regret_beta, 1));

fprintf('\nFinal cumulative regret at T=%d:\n', T);
fprintf('  CB exact : %.1f\n', cr_cb(end));
fprintf('  Beta MM  : %.1f\n', cr_beta(end));
fprintf('  Ratio    : %.2f  (Beta / CB)\n', cr_beta(end) / cr_cb(end));

%% Arm selection: early (t<=100) vs late (t>900)
early = 1:200;  late = (T-499):T;
fprintf('\n--- Arm selection (averaged over %d trials) ---\n', N_trials);
fprintf('%-14s  %8s  %8s  |  %8s  %8s\n', '', 'CB t<=200', 'Beta t<=200', sprintf('CB t>%d',T-500), sprintf('Beta t>%d',T-500));
fprintf('%s\n', repmat('-',66,1));
for k = 1:K
    ce = mean(mean(sel_cb_all(:,early)   == k));
    be = mean(mean(sel_beta_all(:,early) == k));
    cl = mean(mean(sel_cb_all(:,late)    == k));
    bl = mean(mean(sel_beta_all(:,late)  == k));
    fprintf('arm %d (θ=%.2f)   %7.1f%%  %7.1f%%  |  %7.1f%%  %7.1f%%  %s\n', ...
        k, theta_true(k), 100*ce, 100*be, 100*cl, 100*bl, ...
        sel(k==1,'<-- optimal',''));
end

%% Plot
col_cb   = [0.15 0.45 0.85];
col_beta = [0.85 0.30 0.15];
t_vec    = 1:T;

figure('Name','Quick bandit','Position',[100 100 1200 440]);

subplot(1,2,1);
plot(t_vec, cr_cb,   'Color',col_cb,   'LineWidth',2); hold on;
plot(t_vec, cr_beta, 'Color',col_beta, 'LineWidth',2);
% Individual trials
for tr = 1:N_trials
    plot(t_vec, cumsum(regret_cb(tr,:)),   '--','Color',col_cb,   'LineWidth',0.5,'HandleVisibility','off');
    plot(t_vec, cumsum(regret_beta(tr,:)), '--','Color',col_beta, 'LineWidth',0.5,'HandleVisibility','off');
end
xlabel('Round');  ylabel('Cumulative regret');
title(sprintf('Cumulative regret  (K=%d, T=%d, %d trials)', K, T, N_trials));
legend({'CB exact','Beta MM'},'Location','northwest'); grid on;

subplot(1,2,2);
loglog(t_vec, max(cr_cb,   1e-3), 'Color',col_cb,   'LineWidth',2); hold on;
loglog(t_vec, max(cr_beta, 1e-3), 'Color',col_beta, 'LineWidth',2);
% O(log T) reference calibrated to CB
c_log = cr_cb(end) / log(T);
loglog(t_vec, c_log*log(max(t_vec,1)), 'k--', 'LineWidth',1.2);
% O(T) reference calibrated to Beta
loglog(t_vec, cr_beta(end)/T * t_vec, 'k-.', 'LineWidth',1.2);
xlabel('Round (log)');  ylabel('Cumulative regret (log)');
title('Log-log: slope~0 => O(log T), slope~1 => O(T)');
legend({'CB exact','Beta MM','O(log T)','O(T)'},'Location','northwest'); grid on;

function s = sel(c, a, b)
    if c, s = a; else, s = b; end
end
