function [a, b] = cb_to_beta_data_mm(chi, nu)
% CB_TO_BETA_DATA_MM  Beta(a,b) matching CB data moments via mean value parameterization.
%
%   [a, b] = cb_to_beta_data_mm(chi, nu)
%
%   Finds Beta(a,b) such that:
%       E[X | Beta(a,b)]   =  mu   =  B'(eta*)   =  E[X | CB(theta)]
%       Var[X | Beta(a,b)] =  v    =  B''(eta*)  =  Var[X | CB(theta)]
%
%   where eta* is the mode of the CB posterior p(eta | chi, nu), satisfying
%   B'(eta*) = chi/nu = Z_bar.
%
%   This is "what moment matching should have been" for the CB family:
%   match the moments of the SUFFICIENT STATISTIC T(x) = x at the
%   OBSERVATION level, using the natural exponential family structure.
%
%   DERIVATION
%   ----------
%   The mean value parameter mu = B'(eta*) and variance v = B''(eta*)
%   are the first two cumulants of the CB data distribution.
%   For Beta(a,b):
%       E[X]   = a / (a+b)
%       Var[X] = a*b / ((a+b)^2 * (a+b+1))
%   Setting E[X] = mu, Var[X] = v gives:
%       s = mu*(1-mu)/v - 1        [concentration = a+b]
%       a = mu * s
%       b = (1-mu) * s
%
%   COST: O(1) -- requires only two evaluations of the B-function family
%   (already computed by cb_mode + bft_all).  No numerical integration.
%
%   COMPARISON WITH OTHER APPROXIMATIONS
%   -----------------------------------------------------------------------
%   Method            Matches             Cost          Notes
%   Cheesy            Z_bar (wrong!)      O(1)          Bernoulli confusion
%   Data-level MM     E[X], Var[X]        O(1)     <--- this function
%   Parameter-level   E[theta],Var[theta] O(quad)       cb_to_beta_num
%   Hellinger-opt     H^2 directly        O(DOPRI5)     cb_to_beta_hellinger
%   -----------------------------------------------------------------------
%   Inequality: H2_hellinger <= H2_param_mm <= H2_data_mm <= H2_cheesy.
%
%   The data-level approach uses Legendre duality:
%     eta (natural parameter) <--> mu = B'(eta) (mean value parameter)
%   and is the correct exponential-family reasoning for moment matching.
%   See Brown (1986) "Fundamentals of Statistical Exponential Families",
%   ISI Lecture Notes Vol. 9, Chapter 1.
%
%   INPUTS
%     chi    sum of observed rewards (sufficient statistic, > 0)
%     nu     pull count (> 0)
%
%   OUTPUTS
%     a, b   Beta parameters with E[X|Beta] = B'(eta*), Var[X|Beta] = B''(eta*)
%
%   EXAMPLE
%     % 20 observations with average loss 0.30
%     [a, b] = cb_to_beta_data_mm(6, 20);
%     fprintf('Data-level MM: Beta(%.4f, %.4f)\n', a, b);
%     % Compare: cheesy is Beta(7, 15); this is much closer to the CB posterior.
%
%   SEE ALSO: cb_to_beta_num, cb_to_beta_hellinger, cb_mode, bft_all
%
% MIT License.  Andrew Mullhaupt, Stony Brook University AMS/QF, 2026.

%% Degenerate cases
if nu <= 0 || chi <= 0 || chi >= nu
    a = max(chi + 0.5, 1e-4);
    b = max(nu - chi + 0.5, 1e-4);
    return;
end

%% Find posterior mode eta* and read off B'(eta*), B''(eta*)
% B'(eta*) = chi/nu is the mean value parameter mu (observed average loss).
% B''(eta*) is the variance of X under CB(theta*).
eta_star = cb_mode(chi, nu);
[~, mu, v] = bft_all(eta_star);   % mu = B'(eta*),  v = B''(eta*)

%% Moment-match to Beta
% E[X|Beta] = a/(a+b) = mu
% Var[X|Beta] = a*b/((a+b)^2*(a+b+1)) = v
% => s = a+b = mu*(1-mu)/v - 1  (concentration parameter)
%    a = mu*s,  b = (1-mu)*s
s = mu * (1 - mu) / v - 1;

% Guard: if v >= mu*(1-mu) (impossible for CB since B''<B'*(1-B') only fails
% in degenerate limits), clamp to a safe minimum concentration.
if s <= 0
    % v is too large to match with any Beta -- degenerate limit.
    % Fall back to uniform Beta(1,1).
    a = 1;  b = 1;
    return;
end

a = mu * s;
b = (1 - mu) * s;

% Final guard: ensure a,b > 0 (always satisfied if s > 0 and 0 < mu < 1)
a = max(a, 1e-6);
b = max(b, 1e-6);
end
