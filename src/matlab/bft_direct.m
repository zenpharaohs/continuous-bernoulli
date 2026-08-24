function b = bft_direct(eta)
% BFT_DIRECT  Ground truth evaluation of B(eta) = log((exp(eta)-1)/eta)
%   using expm1 to avoid cancellation for small eta.
%
%   Valid for |eta| > ~1e-15. For smaller |eta| use bft_horner or bft_pf
%   which are MORE accurate than this function near zero.
%
%   This function serves as the regression oracle for |eta| in [0.01, 1].
%
%   See also: bft_horner, bft_pf, bft_test

b = log(expm1(eta) ./ eta);
end
