%% bandit_likelihood.m
%
% Bandit comparison: CB-exact vs Beta-MM Thompson sampling
% under THREE reward distributions.
%
%   Scenario A: CB rewards   -- CB posterior is exact, Beta is misspecified
%   Scenario B: Beta rewards -- Beta posterior is exact, CB is misspecified
%
% REWARD GENERATION
%   CB rewards:   x ~ CB(lambda_k) where lambda_k satisfies E[CB] = theta_true(k)
%                 Generated via cb_stream(chi_nu, nu, ...) with nu=1 observation
%                 and chi_nu = theta_true(k) * nu as natural sufficient statistic.
%                 Concretely: draw T iid samples from CB(lambda_k).
%
%   Beta rewards: x ~ Beta(a_k, b_k) where a_k/(a_k+b_k) = theta_true(k),
%                 concentration a_k+b_k = CONC (default 5).
%
% POSTERIOR UPDATES
%   CB Thompson:   chi += x, nu += 1 each round.  CB(chi,nu) is the exact
%                  conjugate posterior under CB likelihood.
%   Beta Thompson: moment-matched Beta(a,b) recomputed via cb_to_beta_num.
%
% MIT License.  Andrew Mullhaupt, Stony Brook University AMS/QF, 2026.

rng(137);

K          = 8;
T          = 5000;
N_trials   = 20;
CONC       = 5;     % Beta reward concentration: a+b = CONC
BLOCK      = 256;

theta_true = [0.05, 0.10, 0.18, 0.25, 0.33, 0.42, 0.50, 0.62];
theta_opt  = theta_true(1);

fprintf('Arms: ');  fprintf('theta=%.2f  ', theta_true);  fprintf('\n');
fprintf('Optimal arm: 1 (theta=%.2f)\n', theta_opt);
fprintf('T=%d,  N_trials=%d,  Beta concentration=%d\n\n', T, N_trials, CONC);

%% Pre-compute Beta reward parameters
%   Beta(a_k, b_k): mean = theta_true(k), a_k + b_k = CONC
a_true = theta_true * CONC;
b_true = (1 - theta_true) * CONC;

%% Pre-generate CB reward streams (one stream per arm, reseeded per trial)
%   CB(lambda_k) where lambda_k is the natural parameter giving mean theta_true(k).
%   We use chi = theta_true(k)*nu_seed, nu = nu_seed with nu_seed=10 so that
%   cb_mode(chi, nu) = logit(theta_true(k)) approximately (exact for symmetric
%   posteriors; for asymmetric the true natural parameter differs slightly from
%   logit, but for reward generation we want the CB distribution whose mean IS
%   theta_true(k), so we use the cb_stream prior path at nu=0 then update to
%   the correct natural parameter).
%
%   Simpler and exact: use nu=0 (prior = Uniform) which gives the CB distribution
%   with parameter lambda directly. But cb_stream at nu=0 draws Uniform(0,1).
%
%   Correct approach: CB(lambda) has natural parameter eta = logit(lambda) only
%   when lambda IS the CB mean function evaluated at eta. For a given theta_true(k)
%   (which we interpret as the CB mean), the natural parameter eta_k satisfies
%   B'(eta_k) = theta_true(k), i.e. eta_k = cb_mode(theta_true(k)*nu0, nu0)
%   for any nu0 > 0. We use nu0 = 100 for precision.
%
%   Then to sample from CB(eta_k): use cb_stream(chi=theta*nu0, nu=nu0) which
%   has posterior concentrated at eta_k, and draw individual samples.
%   This is the marginal CB(eta_k) likelihood distribution, not the posterior --
%   but since the posterior at large nu0 concentrates at eta_k, each draw is
%   effectively from CB(eta_k). We use a large enough nu0 that this approximation
%   is tight.

NU0 = 500;   % large enough that cb_stream(theta*NU0, NU0) ~ CB(eta_k)

%% Storage
results = struct();
scenarios = {'CB rewards', 'Beta rewards'};

for scen = 1:2
    scen_name = scenarios{scen};
    fprintf('=== Scenario %d: %s ===\n', scen, scen_name);

    regret_cb   = zeros(N_trials, T);
    regret_beta = zeros(N_trials, T);

    for trial = 1:N_trials

        %% Generate rewards for this trial
        if scen == 1
            % CB rewards: draw T samples per arm from CB(eta_k)
            rewards = zeros(K, T);
            for k = 1:K
                chi_k = theta_true(k) * NU0;
                nu_k  = NU0;
                s_r = cb_stream(chi_k, nu_k, ...
                    'seed', uint64(trial*10000 + k), 'buf_size', T);
                rewards(k,:) = s_r.draw(T);
                s_r.delete();
            end
        else
            % Beta rewards: draw T samples per arm from Beta(a_k, b_k)
            rewards = zeros(K, T);
            for k = 1:K
                rewards(k,:) = betarnd(a_true(k), b_true(k), 1, T);
            end
        end

        %% CB-exact Thompson
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
            regret_cb(trial, t) = theta_true(arm) - theta_opt;
        end

        %% Beta-MM Thompson
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
            regret_beta(trial, t) = theta_true(arm) - theta_opt;
        end

        if mod(trial, 5) == 0
            fprintf('  Trial %d/%d done\n', trial, N_trials);
        end
    end

    %% Store and report
    cr_cb   = cumsum(mean(regret_cb,   1));
    cr_beta = cumsum(mean(regret_beta, 1));
    se_cb   = std(sum(regret_cb,   2)) / sqrt(N_trials);
    se_beta = std(sum(regret_beta, 2)) / sqrt(N_trials);

    results(scen).name      = scen_name;
    results(scen).cr_cb     = cr_cb;
    results(scen).cr_beta   = cr_beta;
    results(scen).se_cb     = se_cb;
    results(scen).se_beta   = se_beta;
    results(scen).regret_cb = regret_cb;
    results(scen).regret_beta = regret_beta;

    fprintf('  Final regret at T=%d:\n', T);
    fprintf('    CB exact : %.1f  (+/- %.1f 1-SE)\n', cr_cb(end),   se_cb);
    fprintf('    Beta MM  : %.1f  (+/- %.1f 1-SE)\n', cr_beta(end), se_beta);
    fprintf('    Ratio    : %.3f  (Beta/CB)\n\n', cr_beta(end)/cr_cb(end));
end

%% Summary table
fprintf('=== SUMMARY ===\n');
fprintf('%-20s  %10s  %10s  %8s\n', 'Scenario', 'CB regret', 'Beta regret', 'Ratio');
fprintf('%s\n', repmat('-',56,1));
for scen = 1:2
    r = results(scen);
    fprintf('%-20s  %10.1f  %10.1f  %8.3f\n', ...
        r.name, r.cr_cb(end), r.cr_beta(end), r.cr_beta(end)/r.cr_cb(end));
end
fprintf('\nInterpretation:\n');
fprintf('  Ratio > 1: Beta worse (CB advantage in this scenario)\n');
fprintf('  Ratio < 1: CB worse  (Beta advantage in this scenario)\n');
fprintf('  Ratio ~ 1: equivalent at this T (may diverge at larger T)\n\n');

%% Plot
col_cb   = [0.15 0.45 0.85];
col_beta = [0.85 0.30 0.15];
t_vec    = 1:T;

figure('Name','CB vs Beta MM by reward type','Position',[100 100 1300 500]);

for scen = 1:2
    subplot(1,2,scen);
    r = results(scen);

    % Individual trials (faint)
    for tr = 1:N_trials
        plot(t_vec, cumsum(results(scen).regret_cb(tr,:)),   '-','Color',[col_cb   0.2],'LineWidth',0.5,'HandleVisibility','off'); hold on;
        plot(t_vec, cumsum(results(scen).regret_beta(tr,:)), '-','Color',[col_beta 0.2],'LineWidth',0.5,'HandleVisibility','off');
    end
    % Mean
    plot(t_vec, r.cr_cb,   'Color',col_cb,   'LineWidth',2.5, 'DisplayName','CB exact');
    plot(t_vec, r.cr_beta, 'Color',col_beta, 'LineWidth',2.5, 'DisplayName','Beta MM');

    xlabel('Round');  ylabel('Cumulative regret');
    title(sprintf('%s\nRatio=%.3f at T=%d', r.name, r.cr_beta(end)/r.cr_cb(end), T));
    legend('Location','northwest'); grid on;
    xlim([1 T]);
end

% Log-log comparison
figure('Name','Log-log by reward type','Position',[100 620 1300 400]);
for scen = 1:2
    subplot(1,2,scen);
    r = results(scen);
    loglog(t_vec, max(r.cr_cb,   1e-3), 'Color',col_cb,   'LineWidth',2); hold on;
    loglog(t_vec, max(r.cr_beta, 1e-3), 'Color',col_beta, 'LineWidth',2);
    c_log = r.cr_cb(end) / log(T);
    loglog(t_vec, c_log*log(max(t_vec,1)), 'k--', 'LineWidth',1.2);
    xlabel('Round (log)');  ylabel('Regret (log)');
    title(sprintf('%s — log-log', r.name));
    legend({'CB exact','Beta MM','O(log T)'},'Location','northwest'); grid on;
end

fprintf('Done: %s\n', datestr(now));
