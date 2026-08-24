function x = cb_data_sample(theta, n)
% CB_DATA_SAMPLE  Draw n samples from the CB likelihood CB(theta).
%
%   x = cb_data_sample(theta, n)
%
%   Returns n iid samples x ~ CB(theta), with x in [0,1].
%
%   The CB distribution has density:
%     p(x|theta) = C(theta) * theta^x * (1-theta)^{1-x},  x in [0,1]
%   with normalizer C(theta) = logit(theta)/(theta - 1 + 1/theta)...
%   equivalently, in natural parameter eta = logit(theta):
%     p(x|eta) = exp(x*eta - B(eta)),  B(eta) = log((e^eta-1)/eta)
%
%   EXACT INVERSE-CDF SAMPLER
%   The CDF in natural parameter form is:
%     F(x|eta) = (e^{x*eta} - 1) / (e^eta - 1)   for eta != 0
%     F(x|0)   = x                                 (eta=0, theta=0.5: Uniform)
%
%   The quantile function (inverse CDF) is:
%     F^{-1}(u|eta) = log(1 + u*(e^eta - 1)) / eta   for eta != 0
%     F^{-1}(u|0)   = u
%
%   Implemented stably using expm1 and log1p:
%     x = log1p(u * expm1(eta)) / eta
%
%   NOTE: this is a FORWARD sampler from the CB data distribution.
%   It is NOT the conjugate posterior sampler (see cb_sample, cb_stream).
%
%   NOTE ON PARAMETERIZATION:
%     theta is the CB CANONICAL parameter, not the expected value.
%     E[X | CB(theta)] = B'(logit(theta)) != theta  in general.
%     To specify an arm by its expected loss mu, use:
%       x = cb_data_sample(cb_mean_to_theta(mu), n)
%
%   EXAMPLE
%     % Arm with canonical parameter theta = 0.3
%     x = cb_data_sample(0.3, 1e6);
%     fprintf('E[X|theta=0.3] sample mean: %.4f\n', mean(x));   % ~0.430
%     fprintf('Theory:                     %.4f\n', cb_theta_to_mean(0.3));
%
%     % Arm with EXPECTED loss mu = 0.3
%     theta_arm = cb_mean_to_theta(0.3);  % ~0.0646
%     x = cb_data_sample(theta_arm, 1e6);
%     fprintf('E[X] sample mean: %.4f\n', mean(x));   % ~0.300
%
%   SEE ALSO: cb_sample (posterior sampler), cb_mean_to_theta, cb_theta_to_mean
%
% MIT License.  Andrew Mullhaupt, Stony Brook University AMS/QF, 2026.

if nargin < 2, n = 1; end

if theta <= 0 || theta >= 1
    error('cb_data_sample: theta must be in (0,1), got %.6g', theta);
end

eta = log(theta / (1 - theta));   % logit(theta) = natural parameter

u = rand(n, 1);   % uniform samples for inverse-CDF

if abs(eta) < 1e-10
    % theta ~ 0.5: CB -> Uniform[0,1]
    x = u;
else
    % Stable: log1p(u * expm1(eta)) / eta
    % For large |eta|: expm1(eta) ~ exp(eta), log1p(...) ~ eta*x, so x~Exp
    x = log1p(u .* expm1(eta)) / eta;
end

% Clip to [0,1] to guard against tiny floating-point excursions
x = max(0.0, min(1.0, x));
end
