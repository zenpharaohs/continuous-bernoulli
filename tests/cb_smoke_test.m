%% cb_smoke_test.m
% Smoke test for the CB sampler package.  Should complete in < 5 seconds.
% Run from the continuous-bernoulli root after build_cb_stream.

repo_root = fileparts(fileparts(mfilename('fullpath')));
addpath(repo_root);
addpath(fullfile(repo_root, 'src', 'matlab'));

fprintf('CB Sampler smoke test...\n\n');
n_fail = 0;

%% Test 1: MEX available
try
    s = cb_stream(5.0, 10.0, 'seed', uint64(1));
    s.draw(100);
    s.delete();
    fprintf('[PASS] MEX backend available\n');
catch e
    fprintf('[FAIL] MEX not available: %s\n', e.message);
    n_fail = n_fail + 1;
end

%% Closed-family boundary regression
for endpoint = [0, 1]
    s = cb_stream(0.0, 0.0, 'seed', uint64(100 + endpoint));
    ok = true;
    for k = 1:32
        s.update(endpoint);
        [~, ~, regime, sigma] = s.peek();
        ok = ok && regime == 4 && sigma == 0 && s.draw(1) == endpoint;
    end
    s.delete();
    if ok
        fprintf('[PASS] endpoint %d uses deterministic point-mass regime\n', endpoint);
    else
        fprintf('[FAIL] endpoint %d did not remain a point mass\n', endpoint);
        n_fail = n_fail + 1;
    end
end

s = cb_stream(0.0, 0.0, 'seed', uint64(103));
s.update(0.0);
z0 = s.draw(1);
s.update(1.0);
[chi_transition, nu_transition, regime_transition, ~] = s.peek();
z1 = s.draw(1);
s.delete();
ok = z0 == 0 && chi_transition == 1 && nu_transition == 2 && ...
     regime_transition ~= 4 && z1 > 0 && z1 < 1;
if ok
    fprintf('[PASS] boundary state returns to continuous family\n');
else
    fprintf('[FAIL] boundary-to-continuous transition\n');
    n_fail = n_fail + 1;
end

%% Test 2: All regimes produce samples in (0,1)
cases = {0.25, 5,   'ARS wide';
         3.5005268846749824e-4, 1, 'ARS near-boundary';
         5.0,  10,  'ARS mid';
         250,  500, 'CF narrow';
         2.0,  100, 'Gamma extreme'};

for c = 1:size(cases,1)
    chi_p=cases{c,1}; nu=cases{c,2}; desc=cases{c,3};
    s = cb_stream(chi_p, nu, 'seed', uint64(c*7));
    th = s.draw(10000);
    s.delete();
    ok = all(th > 0) && all(th < 1) && ~any(isnan(th));
    if ok
        fprintf('[PASS] %s: all samples in (0,1)\n', desc);
    else
        fprintf('[FAIL] %s: %d out-of-range or NaN\n', desc, sum(~(th>0 & th<1)));
        n_fail = n_fail + 1;
    end
end

%% Test 3: Reflection symmetry
% E[theta | chi, nu] should equal 1 - E[1-theta | nu-chi, nu]
chi_p=3.0; nu=10.0; N=100000;
s1 = cb_stream(chi_p, nu, 'seed', uint64(1));
s2 = cb_stream(nu-chi_p, nu, 'seed', uint64(2));
th1 = s1.draw(N); s1.delete();
th2 = s2.draw(N); s2.delete();
m1 = mean(th1); m2 = 1 - mean(th2);
ok = abs(m1 - m2) < 5 * std(th1)/sqrt(N);
if ok
    fprintf('[PASS] Reflection symmetry: E[theta]=%.4f ~ 1-E[phi]=%.4f\n', m1, m2);
else
    fprintf('[FAIL] Reflection symmetry violated: %.4f vs %.4f\n', m1, m2);
    n_fail = n_fail + 1;
end

%% Test 4: Update/draw cycle
s = cb_stream(0.0, 0.0, 'seed', uint64(99));  % prior
for obs = [0,1,0,0,1,0,0,0,0,1]              % 3 successes out of 10
    s.update(obs);
end
[chi_peek, nu_peek, ~, ~] = s.peek();
ok = (chi_peek == 3) && (nu_peek == 10);
s.delete();
if ok
    fprintf('[PASS] Update/peek: chi=%.0f nu=%.0f after 10 observations\n', chi_peek, nu_peek);
else
    fprintf('[FAIL] Update/peek: expected chi=3 nu=10, got chi=%.0f nu=%.0f\n', chi_peek, nu_peek);
    n_fail = n_fail + 1;
end

%% Test 5: Draw after small ARS update uses the updated posterior
% Regression for stale lazy-hull/ziggurat reuse: (0.5,5) -> (0.58,6)
% moves the mode by < 0.1 sigma, so the old cache used to be reused.
chi0=0.5; nu0=5.0; obs=0.08; chi_p=chi0+obs; nu=nu0+1.0; N=50000;
eta_s = cb_mode(chi_p, nu); f_s = chi_p*eta_s - nu*bft_b(eta_s);
step=0.5; thr=log(1e-12);
eL=eta_s; while chi_p*eL-nu*bft_b(eL)-f_s>thr, eL=eL-step; end
eR=eta_s; while chi_p*eR-nu*bft_b(eR)-f_s>thr, eR=eR+step; end
eL=eL-2; eR=eR+2;
np=20000; eg=linspace(eL,eR,np);
lp=chi_p*eg-nu*arrayfun(@bft_b,eg); lp=lp-max(lp);
pe=exp(lp); de=diff(eg); pm=0.5*(pe(1:end-1)+pe(2:end));
cg=[0,cumsum(pm.*de/sum(pm.*de))]; cg(end)=1;

s = cb_stream(chi0, nu0, 'seed', uint64(123), 'buf_size', 4096);
rebuilds_before = s.rebuilds();
s.update(obs);
th = s.draw(N);
rebuilds_after = s.rebuilds();
s.delete();
tc = max(realmin, min(1-eps, th(:)));
u  = sort(max(0,min(1,interp1(eg,cg,max(eL,min(eR,log(tc./(1-tc)))),'pchip','extrap'))));
ks = max(abs((1:N)'/N - u));
crit = sqrt(-log(0.0005)/(2*N));   % alpha=0.001
if ks < crit
    fprintf('[PASS] Update/draw KS: KS=%.5f < crit=%.5f\n', ks, crit);
else
    fprintf('[FAIL] Update/draw KS: KS=%.5f >= crit=%.5f\n', ks, crit);
    n_fail = n_fail + 1;
end
if rebuilds_after >= rebuilds_before
    fprintf('[PASS] Update/draw block path completed: rebuilds %.0f -> %.0f\n', ...
        rebuilds_before, rebuilds_after);
else
    fprintf('[FAIL] Update/draw rebuild counter moved backwards: %.0f -> %.0f\n', ...
        rebuilds_before, rebuilds_after);
    n_fail = n_fail + 1;
end

%% Test 6: Large requests bypass the stream buffer cleanly
s = cb_stream(4.0, 10.0, 'seed', uint64(77), 'buf_size', 64);
th = s.draw(257);
s.delete();
ok = numel(th) == 257 && all(th > 0) && all(th < 1) && ~any(isnan(th));
if ok
    fprintf('[PASS] Large draw bypass: all samples in (0,1)\n');
else
    fprintf('[FAIL] Large draw bypass: invalid sample or count\n');
    n_fail = n_fail + 1;
end

%% Test 7: KS test against exact CDF (N=50k, should always pass at alpha=0.001)
chi_p=4.0; nu=10.0; N=50000;
eta_s = cb_mode(chi_p, nu); f_s = chi_p*eta_s - nu*bft_b(eta_s);
step=0.5; thr=log(1e-12);
eL=eta_s; while chi_p*eL-nu*bft_b(eL)-f_s>thr, eL=eL-step; end
eR=eta_s; while chi_p*eR-nu*bft_b(eR)-f_s>thr, eR=eR+step; end
eL=eL-2; eR=eR+2;
np=20000; eg=linspace(eL,eR,np);
lp=chi_p*eg-nu*arrayfun(@bft_b,eg); lp=lp-max(lp);
pe=exp(lp); de=diff(eg); pm=0.5*(pe(1:end-1)+pe(2:end));
cg=[0,cumsum(pm.*de/sum(pm.*de))]; cg(end)=1;

s = cb_stream(chi_p, nu, 'seed', uint64(42));
th = s.draw(N); s.delete();
tc = max(realmin, min(1-eps, th(:)));
u  = sort(max(0,min(1,interp1(eg,cg,max(eL,min(eR,log(tc./(1-tc)))),'pchip','extrap'))));
ks = max(abs((1:N)'/N - u));
crit = sqrt(-log(0.0005)/(2*N));   % alpha=0.001
if ks < crit
    fprintf('[PASS] KS test Z=0.40 nu=10: KS=%.5f < crit=%.5f\n', ks, crit);
else
    fprintf('[FAIL] KS test Z=0.40 nu=10: KS=%.5f >= crit=%.5f\n', ks, crit);
    n_fail = n_fail + 1;
end

%% Test 8: Transported core engages under Thompson-style update/draw cycling
s = cb_stream(15.0, 300.0, 'seed', uint64(991), 'buf_size', 256);
for k = 1:100
    s.update(0.05);
    s.draw(1);
end
st = s.core_stats();
s.delete();
total_core_path = st.core_draws + st.remainder_draws;
core_frac = st.core_draws / max(1, total_core_path);
ok = st.core_active == 1 && st.rebuilds <= 1 && total_core_path == 100 && core_frac >= 0.90;
if ok
    fprintf('[PASS] Transported core engaged: alpha=%.3f core_frac=%.3f rebuilds=%.0f\n', ...
        st.alpha_hat, core_frac, st.rebuilds);
else
    fprintf('[FAIL] Transported core did not engage: alpha=%.3f core=%.0f rem=%.0f rebuilds=%.0f active=%.0f\n', ...
        st.alpha_hat, st.core_draws, st.remainder_draws, st.rebuilds, st.core_active);
    n_fail = n_fail + 1;
end

%% Summary
fprintf('\n%s\n', repmat('-',40,1));
if n_fail == 0
    fprintf('ALL TESTS PASSED\n');
else
    fprintf('%d TEST(S) FAILED\n', n_fail);
end
fprintf('%s\n', repmat('-',40,1));
