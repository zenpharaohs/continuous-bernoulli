function mu = cb_theta_to_mean(theta)
% CB_THETA_TO_MEAN  CB canonical parameter -> mean value parameter.
%
%   mu = cb_theta_to_mean(theta)
%
%   Returns mu = E[X | CB(theta)] = B'(logit(theta)), the expected value of
%   a CB-distributed observation when the arm parameter is theta.
%
%   This is the mean value parameter (expectation parameter) of the CB
%   exponential family, dual to the natural parameter eta = logit(theta)
%   via the Legendre-Fenchel transform.  See Brown (1986) "Fundamentals of
%   Statistical Exponential Families", ISI Lecture Notes Vol 9, Chapter 1.
%
%   IMPORTANT: mu != theta (except at theta = 0.5).
%   The CB distribution has a nontrivial normalizer C(lambda), so its mean
%   B'(eta) differs from the canonical parameter sigmoid(eta).
%   Contrast with Bernoulli, where E[X|p] = p (the two coincide).
%
%   EXAMPLE
%     mu = cb_theta_to_mean(0.3)   % -> 0.4302 (NOT 0.3)
%     mu = cb_theta_to_mean(0.5)   % -> 0.5    (symmetric point, they agree)
%
%   INVERSE: cb_mean_to_theta(mu)
%
%   SEE ALSO: cb_mean_to_theta, cb_data_sample, bft_all
%
% MIT License.  Andrew Mullhaupt, Stony Brook University AMS/QF, 2026.

theta = theta(:);
mu = zeros(size(theta));
for k = 1:numel(theta)
    th = theta(k);
    if th <= 0 || th >= 1
        mu(k) = NaN;
    elseif abs(th - 0.5) < 1e-12
        mu(k) = 0.5;
    else
        eta = log(th / (1 - th));   % logit(theta)
        [~, bp, ~] = bft_all(eta);  % B'(eta) = E[X | eta]
        mu(k) = bp;
    end
end
end
