function b = bft_b(eta)
% BFT_B  B(eta) = log((exp(eta)-1)/eta), vectorized dispatch.
%
%   b = bft_b(eta)
%
%   SAMPLING HOT PATH.
%   bft_pf  for |eta| < 1  (3-pole partial fraction, machine precision)
%   direct  for |eta| >= 1 (expm1-based, machine precision)
%
%   Vectorized: accepts any size array.
%
%   See also: bft_pf, bft_all, bft_direct

b     = zeros(size(eta));
small = abs(eta) < 1.0;

if any(small(:))
    b(small) = bft_pf(eta(small));
end
large = ~small;
if any(large(:))
    et  = eta(large);
    pos = et > 0;
    bv  = zeros(size(et));
    if any(pos(:))
        q       = exp(-et(pos));
        bv(pos) = et(pos) + log1p(-q) - log(et(pos));
    end
    if any(~pos(:))
        en       = et(~pos);
        bv(~pos) = log(expm1(en) ./ en);
    end
    b(large) = bv;
end
end
