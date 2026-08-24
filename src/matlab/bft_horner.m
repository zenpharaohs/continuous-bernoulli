function [b, bp, bpp] = bft_horner(eta)
% BFT_HORNER  7-term Horner evaluation of B(eta) = log((exp(eta)-1)/eta)
%   and its first two derivatives, for |eta| < 1.
%
%   [b, bp, bpp] = bft_horner(eta)
%
%   *** INTERNAL USE ONLY ***
%   This function is the reference implementation and Newton solver kernel.
%   It is NOT the sampling hot path -- use bft_pf for B(eta) alone.
%   Derivatives B'(eta) and B''(eta) are machine precision here but are
%   NOT available at machine precision from bft_pf (which approximates G
%   at 3-pole order, amplifying derivative errors by ~1e2 per order).
%   Do NOT expose bft_pf derivatives to users.
%
%   Usage:
%     Newton mode-finder:  [~, bp, bpp] = bft_horner(eta)  [~3x per sample]
%     Regression testing:  b = bft_horner(eta)  [vs bft_pf and bft_direct]
%
%   B(eta)   = log((exp(eta)-1)/eta)          cumulant function of CB dist.
%   B'(eta)  = 1/(1-exp(-eta)) - 1/eta        CB mean at natural param eta
%   B''(eta) = exp(eta)/(exp(eta)-1)^2        CB variance
%              - 1/eta^2
%
%   Representation:
%     B(eta)   = eta/2 + eta^2 * P(u),  u = eta^2
%     B'(eta)  = 1/2   + eta  * Q(u)
%     B''(eta) = R(u)
%   where P, Q, R are degree-6 polynomials in u, evaluated by Horner.
%
%   Coefficients c_j = B_{2j} / (2j * (2j)!) from exact Bernoulli numbers:
%     j:   1      2       3        4          5           6              7
%     B:  1/6  -1/30   1/42    -1/30       5/66      -691/2730        7/6
%     c:  1/24 -1/2880 1/181440 -1/9676800 1/479001600 -691/(...) 7/(...)
%
%   Error: O(eta^16) from 8th Bernoulli term. Full double precision for
%   |eta| < 1 with margin. Validated in bft_test.m.
%
%   See also: bft_pf (sampling hot path), bft_direct (oracle), bft_test

% Exact Bernoulli coefficients c_j = B_{2j} / (2j * (2j)!)
% Computed as constant expressions -- compiler/interpreter folds to doubles
c1 =  1.0/24.0;
c2 = -1.0/2880.0;
c3 =  1.0/181440.0;
c4 = -1.0/9676800.0;
c5 =  1.0/479001600.0;
c6 = -691.0 / (12.0 * 479001600.0 * 2730.0);   % -691 / (2730 * 12!)
c7 =  7.0   / (14.0 * 87178291200.0 * 6.0);    %    7 / (6   * 14!)

u = eta .* eta;   % u = eta^2

%--- B(eta) = eta/2 + eta^2 * P(u) ---
% P(u) = c1 + u*(c2 + u*(... + u*c7))  [Horner, smallest terms first]
P = c7;
P = P.*u + c6;
P = P.*u + c5;
P = P.*u + c4;
P = P.*u + c3;
P = P.*u + c2;
P = P.*u + c1;
b = u.*P + 0.5.*eta;

if nargout < 2, return; end

%--- B'(eta) = 1/2 + eta * Q(u) ---
% Coefficients: d_j = 2j * c_j
% Q(u) = 2c1 + u*(4c2 + u*(6c3 + u*(8c4 + u*(10c5 + u*(12c6 + u*14c7)))))
Q = 14.0*c7;
Q = Q.*u + 12.0*c6;
Q = Q.*u + 10.0*c5;
Q = Q.*u +  8.0*c4;
Q = Q.*u +  6.0*c3;
Q = Q.*u +  4.0*c2;
Q = Q.*u +  2.0*c1;
bp = eta.*Q + 0.5;

if nargout < 3, return; end

%--- B''(eta) = R(u) ---
% Coefficients: e_j = 2j*(2j-1) * c_j
% Prefactors for j=1..7: 2, 12, 30, 56, 90, 132, 182
R = 182.0*c7;
R = R.*u + 132.0*c6;
R = R.*u +  90.0*c5;
R = R.*u +  56.0*c4;
R = R.*u +  30.0*c3;
R = R.*u +  12.0*c2;
R = R.*u +   2.0*c1;
bpp = R;

end
