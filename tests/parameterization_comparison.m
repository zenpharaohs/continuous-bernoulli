% parameterization_comparison.m -- run from cb_sampler root
repo_root = fileparts(fileparts(mfilename('fullpath')));
addpath(repo_root);
addpath(fullfile(repo_root, 'src', 'matlab'));

cases = {6,20,'Z=0.30,nu=20'; 3,10,'Z=0.30,nu=10'; 0.3,1,'Z=0.30,nu=1';
         1,10,'Z=0.10,nu=10'; 9,10,'Z=0.90,nu=10'; 25,50,'Z=0.50,nu=50'};

fprintf('%-18s  %8s  %8s  %8s  %8s\n', 'case','H2_chsy','H2_dl_mm','H2_p_mm','H2_hopt');
for k=1:size(cases,1)
    chi_=cases{k,1}; nu_=cases{k,2};
    eta_s=cb_mode(chi_,nu_); [~,bp,bpp]=bft_all(eta_s);

    % Grid for H2 evaluation
    f_pk=chi_*eta_s-nu_*bft_b(eta_s);
    st=max(1/sqrt(max(nu_*bpp,1e-6)),0.5);
    eL=eta_s; while chi_*eL-nu_*bft_b(eL)>f_pk-40, eL=eL-st; end
    eR=eta_s; while chi_*eR-nu_*bft_b(eR)>f_pk-40, eR=eR+st; end
    eL=eL-st; eR=eR+st;
    eg=linspace(eL,eR,4000)'; de=eg(2)-eg(1);
    ls=-log1p(exp(-eg)); ls1=-log1p(exp(eg));
    lf=(chi_*eg-nu_*bft_b(eg))/2; lf=lf-max(lf);
    Zf=sum(exp(2*lf))*de;

    gh2 = @(a,b) max(0, eval_h2(a,b,lf,ls,ls1,de,Zf));

    % 1. Cheesy
    a_ch=chi_+1; b_ch=nu_-chi_+1;

    % 2. Data-level MM: match E[X]=B'(eta*), Var[X]=B''(eta*)
    %    Only uses the sufficient statistic and two B-function evaluations
    s_dl = bp*(1-bp)/bpp - 1;
    a_dl = bp*s_dl; b_dl = (1-bp)*s_dl;

    % 3. Parameter-level MM (quadrature)
    [a_mm,b_mm]=cb_to_beta_num(chi_,nu_);

    % 4. Hellinger-optimal
    [~,~,~,H2_h]=cb_to_beta_hellinger(chi_,nu_,'max_iter',300);

    fprintf('%-18s  %8.4f  %8.4f  %8.4f  %8.4f\n', cases{k,3}, ...
        gh2(a_ch,b_ch), gh2(a_dl,b_dl), gh2(a_mm,b_mm), H2_h);
end

function rv = eval_h2(a,b,lf,ls,ls1,de,Zf)
    if isnan(a)||isnan(b)||a<=0||b<=0, rv=NaN; return; end
    lg=lf+(a/2)*ls+(b/2)*ls1; lmg=max(lg); g=exp(lg-lmg); S0=sum(g)*de;
    rv = max(0,1-exp(lmg+log(S0)-0.5*log(Zf)-0.5*betaln(a,b)));
end
