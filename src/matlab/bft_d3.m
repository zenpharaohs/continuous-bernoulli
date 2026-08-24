function bppp = bft_d3(eta)
% BFT_D3  Analytic B'''(eta) = d^3/deta^3 log((exp(eta)-1)/eta).
%
%   bppp = bft_d3(eta)
%
%   Derivation:
%     B''(eta) = R(eta^2)  where R(u) = sum_{k=0}^6 D2(k) * u^k
%     d/d(eta) R(eta^2) = 2*eta * R'(eta^2)
%     R'(u)    = sum_{k=0}^5 (k+1)*D2(k+1) * u^k  =: sum D3(k) * u^k
%
%   D3(k) = (k+1)*D2(k+1),  k=0..5.
%
%   Because B'''(0) = 0 (by odd symmetry of B'''), the leading D2(0) = 1/12
%   drops out and D3 coefficients are smaller than D2 coefficients.
%   Condition numbers across the CF regime (|eta*| < ~3): 1.0 to ~4.3.
%   Accuracy: ~2 eps (vs ~4e-14 for 5-point FD at optimal h, ~85x better).
%
%   For |eta| >= 1: direct formula via expm1.
%     B'''(eta) = exp(eta)*(exp(eta)+1)/(exp(eta)-1)^3 - 2/eta^3
%
%   Vectorized: accepts arrays.
%
%   See also: bft_all (B, B', B''), bft_horner (B'' Horner coefficients),
%             cb_sample_laplace_cf (uses B''' for CF skewness coefficient)

% Horner coefficients for R'(u):  D3(k) = (k+1)*D2(k+1)
% from exact Bernoulli numbers (same source as bft_horner.m / cb_bft.h).
D3 = [ 1*(-12.0/2880.0), ...        %  1*D2(1) = -1/240
        2*( 30.0/181440.0), ...       %  2*D2(2) =  1/3024
        3*(-56.0/9676800.0), ...      %  3*D2(3) = -7/403200
        4*( 90.0/479001600.0), ...    %  4*D2(4)
        5*(-132.0*691.0/15692092416000.0), ...   % 5*D2(5)
        6*( 182.0*7.0/7322976460800.0) ];        % 6*D2(6)

bppp = zeros(size(eta));

% --- |eta| < 1: Horner in u = eta^2, then multiply by 2*eta ---
small = abs(eta) < 1.0;
if any(small(:))
    e = eta(small);
    u = e .* e;
    Q = D3(6);
    Q = Q.*u + D3(5);
    Q = Q.*u + D3(4);
    Q = Q.*u + D3(3);
    Q = Q.*u + D3(2);
    Q = Q.*u + D3(1);
    bppp(small) = 2.0 .* e .* Q;
end

% --- |eta| >= 1: direct via expm1 ---
if any(~small(:))
    e    = eta(~small);
    em1  = expm1(e);           % exp(eta) - 1
    ex   = em1 + 1.0;          % exp(eta)
    bppp(~small) = ex .* (ex + 1.0) ./ (em1 .* em1 .* em1) ...
                   - 2.0 ./ (e .* e .* e);
end
end
