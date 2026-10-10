function R = cb_validate_endpoints(N)
% CB_VALIDATE_ENDPOINTS Self-contained MATLAB/MEX endpoint qualification.
%   R = cb_validate_endpoints(N) tests the loaded production MEX, then builds
%   a diagnostic MEX from the same gateway/core with CB_VALIDATE_ETA enabled.
%   That build disables approximations and emits eta through the stream path
%   before sigmoid rounding. The reference CDF is independent MATLAB math.
%   Requires the configured MEX compiler. No Python or external toolbox.
if nargin < 1, N = 16000; end
validateattributes(N, {'numeric'}, {'scalar','integer','>=',100,'<=',100000});
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
started = tic;
try
    old_normal_budget = cb_stream_mex('validation_budget', 5);
catch e
    error('cb_validate:endpointBuild', ...
        'Rebuild the production MEX with build_cb_stream before endpoint validation: %s', e.message);
end
normal_budget_cleanup = onCleanup(@() cb_stream_mex('validation_budget', old_normal_budget)); %#ok<NASGU>

fprintf('Exact endpoint qualification through MATLAB/MEX (N=%d):\n', N);
normal_interface(started);
fprintf('  Production MEX: replacements, reflection, boundaries, updates PASS\n');

% Always rebuild the diagnostic binary so it cannot certify stale C sources.
clear cb_stream_eta_mex;
mex('-O', '-DCB_VALIDATE_ETA=1', ['-I' fullfile(root,'src','c')], ...
    '-outdir', root, '-output', 'cb_stream_eta_mex', ...
    fullfile(root,'src','c','cb_stream_mex.c'));
old_eta_budget = cb_stream_eta_mex('validation_budget', 5);
eta_budget_cleanup = onCleanup(@() cb_stream_eta_mex('validation_budget', old_eta_budget)); %#ok<NASGU>

states = [594-593.9999812553783, 594; 258-257.99997153659604, 258; ...
          1e-10, 2; .02, 2; .01, 49; 2, 200; ...
          7*3.8758505860895066e-32,7; 7e-20,7; 2e-32,2; ...
          7-(7-eps(7)),7; 2e-150,2;1,1e6];
R = struct('chi',{},'nu',{},'reflected',{},'path',{},'n',{}, ...
           'ks',{},'ks_limit',{},'reference_error',{},'passed',{});
for c = 1:size(states,1)
    gap = states(c,1); nu = states(c,2);
    for high = [false true]
        % Tiny gaps can have no representable interior upper counterpart.
        if high && nu-gap==nu, continue; end
        chi = gap; if high, chi = nu-gap; end
        rate = chi; if high, rate = nu-chi; end
        [grid, cdf, center, scale, ref_error] = reference_cdf(rate, nu);
        paths = [0 1];
        if high && any(nu == [258 594]), paths = [0 1 2]; end
        if gap<1e-18, paths(end+1)=3; end
        paths(end+1)=4;
        for path = paths
            [eta, description] = latent_draw(chi, nu, N, path, started);
            if high, eta = -eta; end % compare in the lower reflected coordinate
            u = sort(interp1(grid, cdf, (eta-center)/scale, 'linear', NaN));
            assert(all(isfinite(u)), 'Latent draw fell outside the reference CDF support.');
            idx = (1:N)';
            ks = max([max(idx/N-u), max(u-(idx-1)/N)]);
            limit = 3/sqrt(N) + ref_error;
            row = struct('chi',chi,'nu',nu,'reflected',high,'path',description, ...
                'n',N,'ks',ks,'ks_limit',limit,'reference_error',ref_error,'passed',ks<=limit);
            R(end+1) = row; %#ok<AGROW>
            fprintf('  gap=%.6g nu=%g %s %-23s KS=%.5f limit=%.5f %s\n', ...
                rate, nu, side_name(high), description, ks, limit, pass_name(row.passed));
            check_time(started);
        end
    end
end
adaptive_reuse(started);
remainder_errors(started);
fprintf('  MEX rejection budget and explicit remainder errors PASS\n');
assert(numel(R)==66 && all([R.passed]), 'cb_validate:endpointDistribution', ...
       'Exact endpoint distribution qualification failed.');
fprintf('  MATLAB/MEX endpoint qualification PASSED in %.2fs\n\n', toc(started));
end

function normal_interface(started)
s = cb_stream(0, 0, 'seed', uint64(1729), 'buf_size', 64);
cleanup = onCleanup(@() s.delete()); %#ok<NASGU>
for k = 1:3000
    nu = double(k); chi = nu-nu*3.2e-8;
    s.set_stats(chi, nu);
    [actual_chi, actual_nu, regime, sigma] = s.peek();
    assert(actual_chi==chi && actual_nu==nu && regime==3);
    assert(abs(sigma*(nu-chi)/sqrt(nu)-1)<1e-8, 'Stale MEX mode/scale after set_stats.');
    theta = s.draw(1);
    assert(all(theta>0 & theta<1 & isfinite(theta)));
    check_time(started);
end
reported = [593.9999812553783,594;257.99997153659604,258];
states = [reported; reported(:,2)-reported(:,1),reported(:,2); ...
          2,5;300,600;0,10;10,10;0,0; ...
          7*3.8758505860895066e-32,7;7e-20,7;2e-32,2];
for repeat = 1:20
    for c = 1:size(states,1)
        chi = states(c,1); nu = states(c,2);
        s.set_stats(chi, nu);
        theta = s.draw(7);
        assert(numel(theta)==7 && all(isfinite(theta)));
        if nu>0 && chi==0
            assert(all(theta==0));
        elseif nu>0 && chi==nu
            assert(all(theta==1));
        else
            assert(all(theta>0 & theta<1));
        end
        check_time(started);
    end
end
% Rejected replacements must leave the stream untouched.
s.set_stats(2,5);
for bad = [-1,3,NaN,Inf]
    caught = false;
    try, s.set_stats(bad,2); catch e, caught = strcmp(e.identifier,'cb_stream:domain'); end
    assert(caught);
    [chi,nu] = s.peek(); assert(chi==2 && nu==5);
end
% Actual update/update_batch gateway calls from both endpoint-adjacent sides.
for high = [false true]
    gap = reported(1,2)-reported(1,1); nu = 594;
    chi = gap; if high, chi=nu-gap; end
    s.set_stats(chi,nu);
    s.draw(1);
    s.update(double(high));
    s.update_batch([double(high),double(high)]);
    [actual_chi,actual_nu] = s.peek();
    assert(actual_chi==chi+3*double(high) && actual_nu==nu+3);
    theta = s.draw(7); assert(all(theta>0 & theta<1 & isfinite(theta)));
end
% Production MEX must retain a covered live core across replacement.
for c = 1:size(reported,1)
    chi = reported(c,1); nu = reported(c,2); gap = nu-chi;
    live = cb_stream((nu-1)-.9*gap,nu-1,'seed',uint64(1729),'buf_size',64);
    live_cleanup = onCleanup(@() live.delete());
    live.draw(1); before = live.core_stats();
    live.set_stats(chi,nu); after = live.core_stats();
    assert(before.core_active && after.core_active && before.rebuilds==after.rebuilds);
    theta = live.draw(7); assert(all(theta>0 & theta<1 & isfinite(theta)));
    clear live_cleanup;
end
end

function adaptive_reuse(started)
ptr=cb_stream_eta_mex('create',2e-32,2,uint64(1729),uint64(0),64);
cleanup=onCleanup(@() cb_stream_eta_mex('destroy',ptr)); %#ok<NASGU>
cb_stream_eta_mex('validation_coarse_hull',ptr);
initial=cb_stream_eta_mex('validation_ars',ptr);
cb_stream_eta_mex('draw',ptr,1);
before=cb_stream_eta_mex('validation_ars',ptr);
for k=1:1024, cb_stream_eta_mex('draw',ptr,1); end
after=cb_stream_eta_mex('validation_ars',ptr);
assert(after(1)==before(1) && after(4)>initial(4),'Adaptive MEX hull was not reused.');
cb_stream_eta_mex('set_stats',ptr,7e-20,7);
cb_stream_eta_mex('draw',ptr,1);
replaced=cb_stream_eta_mex('validation_ars',ptr);
assert(replaced(1)==after(1)+1,'Replaced statistics reused a stale envelope.');
check_time(started);
end

function [eta, description] = latent_draw(chi, nu, N, path, started)
initial_chi = chi; initial_nu = nu;
if path==2
    initial_nu = nu-1;
    initial_chi = initial_nu-.9*(nu-chi);
end
ptr = cb_stream_eta_mex('create',initial_chi,initial_nu,uint64(1729),uint64(0),64);
cleanup = onCleanup(@() cb_stream_eta_mex('destroy',ptr)); %#ok<NASGU>
if path==4
    cb_stream_eta_mex('validation_coarse_hull',ptr);
    before = cb_stream_eta_mex('validation_ars',ptr);
    eta = cb_stream_eta_mex('draw',ptr,N);
    after = cb_stream_eta_mex('validation_ars',ptr);
    assert(before(7)==2 && after(7)>2 && after(4)>before(4));
    description = 'adaptive rescue';
elseif path==3
    eta = zeros(N,1);
    for k=1:N
        eta(k) = cb_stream_eta_mex('draw',ptr,1);
        if mod(k,1024)==0, check_time(started); end
    end
    description = 'stream buffer';
elseif path==0
    cb_stream_eta_mex('validation_core',ptr,0);
    eta = cb_stream_eta_mex('draw',ptr,N);
    description = 'ARS';
else
    if path==2
        ignored = cb_stream_eta_mex('draw',ptr,1); %#ok<NASGU>
        before = cb_stream_eta_mex('validation_state',ptr);
        cb_stream_eta_mex('set_stats',ptr,chi,nu);
        after = cb_stream_eta_mex('validation_state',ptr);
        assert(before(2)==2 && before(3)<before(4) && before(5)==1);
        assert(isequal(before(2:5),after(2:5)), 'Live MEX core buffer was discarded.');
        assert(abs(after(1)*(nu-chi)/nu+1)<1e-8, 'Stale live MEX mode.');
    end
    eta = zeros(N,1);
    for first = 1:32:N
        count = min(32,N-first+1);
        eta(first:first+count-1) = cb_stream_eta_mex('draw',ptr,count);
        check_time(started);
    end
    [~,core_draws,rem_draws,~,active] = cb_stream_eta_mex('core_stats',ptr);
    if path==2
        assert(active && core_draws>0 && rem_draws>0);
        description = 'live core buffer';
    elseif active
        assert(core_draws>0 && rem_draws>0);
        description = 'core/remainder';
    else
        description = 'ARS (core unavailable)';
    end
end
assert(all(isfinite(eta)) && numel(eta)==N);
check_time(started);
end

function [grid,cdf,center,scale,err] = reference_cdf(rate,nu)
% A coordinate choice from the tail density, not the C mode solver.
center = -nu/rate; scale = sqrt(nu)/rate;
upper = min(40,max(20,(nu+45*rate/(nu-rate))/sqrt(nu)));
[coarse_grid,coarse_cdf,coarse_tail] = density_grid(65537,rate,nu,center,scale,upper);
[grid,cdf,fine_tail] = density_grid(131073,rate,nu,center,scale,upper);
err = max(abs(coarse_cdf-interp1(grid,cdf,coarse_grid,'linear')))+max(coarse_tail,fine_tail);
assert(err<=5e-7, 'Independent MATLAB reference CDF did not converge.');
end

function [grid,cdf,tail_bound] = density_grid(count,rate,nu,center,scale,upper)
grid = linspace(-40,upper,count)'; eta = center+scale*grid;
b = zeros(size(eta)); neg = eta<0; pos = eta>0;
b(neg) = log(-expm1(eta(neg)))-log(-eta(neg));
b(pos) = eta(pos)+log(-expm1(-eta(pos)))-log(eta(pos));
lp = rate*eta-nu*b; p = exp(lp-max(lp));
cdf = cumtrapz(grid,p); mass=cdf(end);
edges=eta([1,end]); q=exp(-abs(edges)); bp=-q./(1-q)-1./edges;
bp(edges>0)=1./(1-q(edges>0))-1./edges(edges>0);
slopes=scale*(rate-nu*bp);
assert(slopes(1)>0 && slopes(2)<0);
tail_bound=(p(1)/slopes(1)+p(end)/(-slopes(2)))/mass;
assert(tail_bound<1e-10,'Reference grid omitted too much probability.');
cdf=cdf/mass;
end

function remainder_errors(started)
for count = [1 7]
    ptr = cb_stream_eta_mex('create',593.9999812553783,594,uint64(1729),uint64(0),64);
    cleanup = onCleanup(@() cb_stream_eta_mex('destroy',ptr));
    cb_stream_eta_mex('validation_bad_remainder',ptr);
    caught = false;
    try
        ignored = cb_stream_eta_mex('draw',ptr,count); %#ok<NASGU>
    catch e
        caught = strcmp(e.identifier,'cb_stream:draw');
    end
    assert(caught, 'Invalid remainder must raise a MEX draw error, not return a fallback.');
    clear cleanup;
    check_time(started);
end
% Demonstrate that the MEX guard actually interrupts a bad rejection loop.
ptr = cb_stream_eta_mex('create',593.9999812553783,594,uint64(1729),uint64(0),64);
cleanup = onCleanup(@() cb_stream_eta_mex('destroy',ptr)); %#ok<NASGU>
cb_stream_eta_mex('validation_bad_remainder',ptr);
old = cb_stream_eta_mex('validation_budget',1e-9);
budget_cleanup = onCleanup(@() cb_stream_eta_mex('validation_budget',old)); %#ok<NASGU>
caught = false;
try, ignored=cb_stream_eta_mex('draw',ptr,1); catch e, caught=strcmp(e.identifier,'cb_stream:validationTimeout'); end
assert(caught, 'MEX rejection budget did not interrupt the faulty draw.');
end

function check_time(started)
assert(toc(started)<=60, 'cb_validate:endpointTimeout', 'MATLAB endpoint suite exceeded 60 seconds.');
end
function name = side_name(high)
if high, name='high'; else, name='low'; end
end
function name = pass_name(passed)
if passed, name='PASS'; else, name='FAIL'; end
end
