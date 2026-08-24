function theta = cb_mean_to_theta(mu)
% CB_MEAN_TO_THETA  CB mean value parameter -> canonical parameter.
%
%   theta = cb_mean_to_theta(mu)
%
%   Returns theta = sigmoid(eta) where B'(eta) = mu.  Equivalently, theta
%   is the CB arm parameter whose expected observations have mean mu.
%
%   This is the inverse of the mean value mapping mu = B'(eta):
%     1. Find eta* such that B'(eta*) = mu  (Newton solve via cb_mode)
%     2. Return theta = sigmoid(eta*)
%
%   The mapping B': R -> (0,1) is strictly increasing and bijective, so
%   the inverse is unique.
%
%   IMPORTANT: theta != mu (except at mu = 0.5).
%   If you observe n pulls with average loss Z_bar = chi/nu, the CB arm
%   parameter that generated those observations is approximately
%   cb_mean_to_theta(Z_bar), NOT Z_bar itself.
%
%   EXAMPLE
%     theta = cb_mean_to_theta(0.3)   % -> 0.0646 (arm param for mean loss 0.3)
%     theta = cb_mean_to_theta(0.5)   % -> 0.5    (symmetric)
%
%     % Verify round-trip:
%     cb_theta_to_mean(cb_mean_to_theta(0.3))   % -> 0.3
%
%   INVERSE: cb_theta_to_mean(theta)
%
%   RELATIONSHIP TO cb_mode:
%     cb_mode(chi, nu) returns eta* where B'(eta*) = chi/nu.
%     cb_mean_to_theta(mu) = sigmoid(cb_mode(mu, 1)).
%
%   SEE ALSO: cb_theta_to_mean, cb_mode, bft_all
%
% MIT License.  Andrew Mullhaupt, Stony Brook University AMS/QF, 2026.

mu = mu(:);
theta = zeros(size(mu));
for k = 1:numel(mu)
    m = mu(k);
    if m <= 0 || m >= 1
        theta(k) = NaN;
    elseif abs(m - 0.5) < 1e-12
        theta(k) = 0.5;
    else
        % Newton solve B'(eta) = m, then theta = sigmoid(eta)
        % cb_mode(chi=m, nu=1) gives the mode eta* satisfying B'(eta*)=m/1=m
        eta_star = cb_mode(m, 1);
        theta(k) = 1 / (1 + exp(-eta_star));
    end
end
end
