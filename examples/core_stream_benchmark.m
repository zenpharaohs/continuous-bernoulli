%% core_stream_benchmark.m
% Microbenchmark for the transported-core streaming sampler.
%
% Reports:
%   - fixed-posterior draw throughput for several buffer sizes
%   - Thompson-style update/draw(1) cost for several posterior regimes
%   - transported-core engagement diagnostics
%
% This benchmark is meant for comparing implementations.  MATLAB call
% overhead is included in the update/draw(1) loop, so use deltas between
% versions more than absolute nanoseconds as the decision signal.

clear; clc;

this_file = mfilename('fullpath');
repo_root = fileparts(fileparts(this_file));
addpath(repo_root);
addpath(fullfile(repo_root, 'src', 'matlab'));

if exist('cb_stream_mex', 'file') ~= 3
    build_cb_stream;
end

rng(123);

N_bulk = 1000000;
T_hot  = 20000;
bufs   = [64 128 256 512 1024 4096];

cases = struct( ...
    'name',  {'extreme-hot', 'interior-ars', 'near-cf'}, ...
    'chi',   {15.0,          120.0,          480.0}, ...
    'nu',    {300.0,         300.0,          1000.0}, ...
    'x',     {0.05,          0.40,           0.48});

fprintf('Transported-core stream benchmark\n');
fprintf('N_bulk=%d, T_hot=%d\n\n', N_bulk, T_hot);

%% Fixed posterior bulk draw
fprintf('Fixed posterior bulk draw\n');
fprintf('%-12s %6s %8s %12s %10s %10s %9s\n', ...
    'case', 'buf', 'regime', 'draws/sec', 'alpha', 'core_frac', 'rebuilds');

for c = 1:numel(cases)
    for b = 1:numel(bufs)
        s = cb_stream(cases(c).chi, cases(c).nu, ...
            'seed', uint64(1000 + 17*c + b), ...
            'buf_size', bufs(b));
        [~,~,reg,~] = s.peek();
        chunk = min(bufs(b), 1024);
        n_left = N_bulk;
        t0 = tic;
        while n_left > 0
            n_now = min(chunk, n_left);
            s.draw(n_now);
            n_left = n_left - n_now;
        end
        elapsed = toc(t0);
        st = s.core_stats();
        s.delete();

        total_core = st.core_draws + st.remainder_draws;
        core_frac = st.core_draws / max(1, total_core);
        fprintf('%-12s %6d %8s %12.3g %10.3f %10.3f %9.0f\n', ...
            cases(c).name, bufs(b), regime_name(reg), N_bulk / elapsed, ...
            st.alpha_hat, core_frac, st.rebuilds);
    end
end

%% Thompson-style hot loop
fprintf('\nThompson-style update/draw(1) loop\n');
fprintf('%-12s %6s %8s %10s %12s %10s %10s %9s\n', ...
    'case', 'buf', 'regime', 'us/iter', 'draws/sec', 'alpha', 'core_frac', 'rebuilds');

for c = 1:numel(cases)
    for b = 1:numel(bufs)
        s = cb_stream(cases(c).chi, cases(c).nu, ...
            'seed', uint64(2000 + 17*c + b), ...
            'buf_size', bufs(b));
        [~,~,reg,~] = s.peek();

        x = cases(c).x;
        t0 = tic;
        for k = 1:T_hot
            s.update(x);
            s.draw(1);
        end
        elapsed = toc(t0);

        st = s.core_stats();
        s.delete();
        total_core = st.core_draws + st.remainder_draws;
        core_frac = st.core_draws / max(1, total_core);

        fprintf('%-12s %6d %8s %10.3f %12.3g %10.3f %10.3f %9.0f\n', ...
            cases(c).name, bufs(b), regime_name(reg), ...
            1e6 * elapsed / T_hot, T_hot / elapsed, ...
            st.alpha_hat, core_frac, st.rebuilds);
    end
end

%% Large-request bypass
fprintf('\nLarge-request bypass sanity\n');
fprintf('%-12s %6s %12s\n', 'case', 'N', 'draws/sec');
for c = 1:numel(cases)
    s = cb_stream(cases(c).chi, cases(c).nu, ...
        'seed', uint64(3000 + c), ...
        'buf_size', 256);
    t0 = tic;
    s.draw(N_bulk);
    elapsed = toc(t0);
    s.delete();
    fprintf('%-12s %6d %12.3g\n', cases(c).name, N_bulk, N_bulk / elapsed);
end

function name = regime_name(reg)
switch reg
    case 0, name = 'prior';
    case 1, name = 'gamma';
    case 2, name = 'cf';
    case 3, name = 'ars';
    otherwise, name = 'unknown';
end
end
