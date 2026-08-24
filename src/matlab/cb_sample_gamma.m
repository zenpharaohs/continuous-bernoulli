function theta = cb_sample_gamma(chi, nu, n)
% CB_SAMPLE_GAMMA  Fast approximate sampler for CB posterior near theta=0 or theta=1.
%
%   theta = cb_sample_gamma(chi, nu, n)
%
%   Uses the exact reflection p(theta|chi,nu) = p(1-theta|nu-chi,nu)
%   [from B(eta)-B(-eta)=eta] to always run the low-Z parameterisation:
%
%     phi = exp(-tau),   tau ~ Gamma(nu+1, rate)
%     rate = chi         (low-Z,  zbar <= 0.5)
%     rate = nu - chi    (high-Z, zbar >  0.5, via reflection)
%     theta = phi              (low-Z)
%     theta = 1 - phi          (high-Z)
%
%   phi = exp(-tau) is always a representable positive double.
%   For high-Z with mean_tau = (nu+1)/(nu-chi) > 3.67, 1-phi may round to 1.0
%   (point mass at theta=1 for ~0.01% of samples).  In that regime the C stream
%   routes to ARS; this reference sampler is not called for those cases.
%
%   Valid when: zmin * sqrt(nu) < 0.18  AND  mean_tau < 3.67 (if zbar > 0.5)
%   where zmin = min(zbar, 1-zbar),  zbar = chi/nu.
%
%   See also: cb_sample (ARS reference), cb_stream (C backend)

if nargin < 3, n = 1; end

reflect = (chi / nu > 0.5);
rate    = reflect * (nu - chi) + (~reflect) * chi;   % no branch

tau   = gamrnd(nu + 1, 1.0 / rate, n, 1);
phi   = exp(-tau);                     % always representable in (0, 1]
theta = phi;
if reflect
    theta = 1.0 - phi;                 % reflect: p(theta|chi,nu)=p(1-theta|nu-chi,nu)
end

% Clamp to open (0,1) -- safety net only; precision guard in C ensures
% we are not called for extreme cases where 1-phi rounds to 1.0
theta = max(realmin, min(1 - eps, theta));
end
