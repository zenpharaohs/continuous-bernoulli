function R = cb_validate_cf(N)
% Exact CF regression through production and latent MEX; independent CDF.
% cb_validate_endpoints builds cb_stream_eta_mex before this suite runs.
if nargin<1, N=100000; end
states=[150.5,301;250,500;500,1000;2500,5000;500000,1e6; ...
        249.5,500;250.5,500;10000,1e6;990000,1e6];
R=struct('chi',{},'nu',{},'ks',{},'passed',{});
old=cb_stream_eta_mex('validation_budget',5);
budget=onCleanup(@() cb_stream_eta_mex('validation_budget',old)); %#ok<NASGU>
fprintf('Exact CF correction through MATLAB/MEX (N=%d):\n',N);
for j=1:size(states,1)
    chi=states(j,1);nu=states(j,2);
    p=cb_stream_mex('create',chi,nu,uint64(1729),uint64(0),64);
    cleanup=onCleanup(@() cb_stream_mex('destroy',p));
    [~,~,reg,~]=cb_stream_mex('peek',p);assert(reg==2);
    theta=cb_stream_mex('draw',p,257);assert(all(theta>0 & theta<1 & isfinite(theta)));
    clear cleanup;
    p=cb_stream_eta_mex('create',chi,nu,uint64(1729),uint64(0),64);
    cleanup=onCleanup(@() cb_stream_eta_mex('destroy',p));
    cb_stream_eta_mex('validation_cf',p);
    eta=cb_stream_eta_mex('draw',p,N);
    if chi>nu/2, eta=-eta; end
    [~,~,~,sigma]=cb_stream_eta_mex('peek',p);
    rate=min(chi,nu-chi);
    % Mode only selects the numerical grid. Its density is independent math.
    center=0;
    if rate<nu/2
        if rate/nu>.2, a=12*(.5-rate/nu); else, a=nu/rate; end
        for k=1:30
            if a<.1
                b1=.5-a/12+a^3/720-a^5/30240+a^7/1209600;
                b2=1/12-a^2/240+a^4/6048-a^6/172800;
            else
                b1=1/a-1/expm1(a);
                b2=1/a^2-exp(-a)/(-expm1(-a))^2;
            end
            step=(b1-rate/nu)/(-b2);a=max(eps,a-step);
        end
        center=-a;
    end
    grid=center+sigma*linspace(-40,40,131073);
    z=grid.^2;small=abs(grid)<.02;
    c=zeros(size(grid));
    c(small)=z(small).*(1/24+z(small).*(-1/2880+z(small).*(1/181440-z(small)/9676800)));
    c(~small)=log(sinh(grid(~small)/2)./(grid(~small)/2));
    logp=(rate-nu/2)*grid-nu*c;
    density=exp(logp-max(logp));cdf=cumtrapz(grid,density);cdf=cdf/cdf(end);
    pit=sort(interp1(grid,cdf,eta,'linear',0));pit=pit(:);ranks=(1:N)';
    ks=max([max(ranks/N-pit),max(pit-(ranks-1)/N)]);
    assert(ks<=3/sqrt(N)+5e-7,'Exact CF distribution check failed.');
    R(end+1)=struct('chi',chi,'nu',nu,'ks',ks,'passed',true); %#ok<AGROW>
    fprintf('  chi=%g nu=%g KS=%.5f PASS\n',chi,nu,ks);
    clear cleanup;
end
% Statistics replacement while an old CF theta buffer is live.
p=cb_stream_mex('create',250,500,uint64(1729),uint64(0),64);
cleanup=onCleanup(@() cb_stream_mex('destroy',p)); %#ok<NASGU>
for k=1:20
    for state=[250,500,0,.155;10000,1e6,-100,.1;500000,1e6,0,.0035]'
        cb_stream_mex('set_stats',p,state(1),state(2));
        theta=cb_stream_mex('draw',p,1);eta=log(theta)-log1p(-theta);
        assert(abs(eta-state(3))<12*state(4),'Stale CF buffer after set_stats.');
    end
end
fprintf('  CF live-buffer replacement PASS\n');
clear cleanup;
old_fault_budget=cb_stream_eta_mex('validation_budget',0);
fault_budget=onCleanup(@() cb_stream_eta_mex('validation_budget',old_fault_budget)); %#ok<NASGU>
for n=[1,65]
    p=cb_stream_eta_mex('create',250,500,uint64(1729),uint64(0),64);
    cleanup=onCleanup(@() cb_stream_eta_mex('destroy',p));
    cb_stream_eta_mex('validation_bad_cf',p);
    caught=false;
    try, ignored=cb_stream_eta_mex('draw',p,n); catch e, caught=strcmp(e.identifier,'cb_stream:draw'); end
    assert(caught,'CF rejection-budget failure did not reach the MEX caller.');
    clear cleanup;
end
fprintf('  CF buffered and batch bounded-error checks PASS\n\n');
end
