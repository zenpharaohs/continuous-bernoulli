function s = cb_stream_mean(mu, n, varargin)
% CB_STREAM_MEAN  Create a CB stream using the mean value parameterization.
%
%   s = cb_stream_mean(mu, n)
%   s = cb_stream_mean(mu, n, 'seed', uint64(42), ...)
%
%   Equivalent to cb_stream(mu*n, n, ...).
%
%   In the mean value parameterization, the sufficient statistics are:
%     mu = chi/nu = observed mean of CB data (= Z_bar)
%     n  = nu     = number of observations
%   so chi = mu * n.
%
%   This is the natural way to specify a posterior for users who think
%   in terms of "n observations with average value mu", rather than
%   "sufficient statistics chi and nu".
%
%   The mean value parameter mu = E_eta[X] = B'(eta) is dual to the
%   natural parameter eta via the Legendre transform.  See Brown (1986)
%   "Fundamentals of Statistical Exponential Families", ISI Lecture Notes.
%
%   INPUTS
%     mu    observed mean of CB data, in (0,1)
%     n     number of observations (nu), > 0
%
%   All additional arguments are passed to cb_stream.
%
%   EXAMPLE
%     % 20 observations with average loss 0.30
%     s = cb_stream_mean(0.30, 20);
%     theta_draws = s.draw(1000);
%     fprintf('Posterior mean theta: %.4f\n', mean(theta_draws));
%     % -> ~0.0646  (the CB arm parameter, not 0.30)
%     % See cb_mean_to_theta(0.30) for the asymptotic value.
%     s.delete();
%
%   SEE ALSO: cb_stream, cb_mean_to_theta, cb_theta_to_mean
%
% MIT License.  Andrew Mullhaupt, Stony Brook University AMS/QF, 2026.

if mu <= 0 || mu >= 1
    error('cb_stream_mean: mu must be in (0,1), got %.6g', mu);
end
if n <= 0
    error('cb_stream_mean: n must be > 0, got %.6g', n);
end

chi = mu * n;
nu  = n;
s   = cb_stream(chi, nu, varargin{:});
end
