function [b, bp, bpp] = bft_all(eta)
% BFT_ALL  B(eta), B'(eta), B''(eta) with scalar dispatch.
%   [b, bp, bpp] = bft_all(eta)
%
%   Horner (machine precision derivatives) for |eta| < 1.
%   Direct formulas via expm1 for |eta| >= 1.
%
%   INTERNAL USE ONLY: Newton mode-finder kernel.
%   For B(eta) alone in the sampling hot path, use bft_b.
%
%   Scalar input only.  For vectorized B call bft_b.
%
%   See also: bft_b, bft_horner, cb_mode

if abs(eta) < 1.0
    [b, bp, bpp] = bft_horner(eta);
elseif eta > 0.0
    q    = exp(-eta);
    om   = 1.0 - q;
    b    = eta + log1p(-q) - log(eta);
    bp   = 1.0/om - 1.0/eta;
    bpp  = 1.0/eta^2 - q/om^2;
else
    em1  = expm1(eta);        % exp(eta) - 1, accurate for all eta
    e    = em1 + 1.0;         % exp(eta)
    b    = log(em1 / eta);    % log((exp(eta)-1)/eta)
    bp   = e/em1 - 1.0/eta;   % exp(eta)/(exp(eta)-1) - 1/eta
    bpp  = 1.0/eta^2 - e/em1^2; % 1/eta^2 - exp(eta)/(exp(eta)-1)^2
    % bpp > 0 always (B strictly convex)
end
end
