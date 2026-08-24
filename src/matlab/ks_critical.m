function d_crit = ks_critical(n, alpha)
%KS_CRITICAL  Exact critical value for the one-sample KS test.
%
%   d_crit = ks_critical(n, alpha)
%
%   Returns d_crit such that P(D_n > d_crit) = alpha exactly,
%   i.e. the upper-alpha quantile of D_n under H_0.
%
%   Uses ks_exact_cdf (MTW exact for n <= 10000, asymptotic Kolmogorov
%   for n > 10000) with 60-step bisection for the inverse CDF.
%   Precision in d: ~ 2^{-60} * d_asym  (better than 1e-15 for all cases).
%
%   BRACKET CONSTRUCTION
%   Initial bracket from asymptotic formula d_asym = sqrt(-log(alpha/2)/(2n)).
%   Then verified: the bracket expansion walks hi upward and lo downward
%   until [lo, hi] is guaranteed to straddle the target quantile.
%   The lo expansion uses lo/2 steps to avoid the t <= 0.15 early-return
%   floor in ks_exact_cdf, which returns 0 for d <= 0.15/sqrt(n).
%   At standard alpha (0.01 or 0.05), d_asym >> 0.15/sqrt(n) for all n,
%   so this floor is never hit in practice; the guard is for robustness
%   at extreme alpha values.
%
%   INPUTS
%     n      positive integer: sample size
%     alpha  significance level in (0,1), e.g. 0.01 or 0.05
%
%   OUTPUT
%     d_crit  exact upper-alpha quantile of D_n
%
%   SEE ALSO: ks_exact_cdf
%
% MIT License.  Andrew Mullhaupt, Stony Brook University AMS/QF, 2026.

target = 1.0 - alpha;

%% Initial bracket from asymptotic formula
d_asym = sqrt(-log(alpha/2) / (2*n));

% Floor for lo: stay above the t=0.15 early-return in ks_exact_cdf.
% Below d_floor, ks_exact_cdf returns 0 regardless of actual CDF value.
d_floor = 0.16 / sqrt(n);   % slight margin above 0.15/sqrt(n)

lo = max(d_asym * 0.5, d_floor);
hi = min(1.0, d_asym * 2.0);

%% Ensure bracket is valid: lo gives p < target,  hi gives p >= target
while ks_exact_cdf(n, lo) >= target
    lo = max(lo / 2, d_floor);
    if lo <= d_floor + 1e-15,  lo = d_floor;  break;  end
end
while ks_exact_cdf(n, hi) < target
    hi = min(1.0, hi * 1.5);
    if hi >= 1.0,  break;  end
end

%% Bisection: 60 steps, precision < 2^{-60} * (hi-lo) < 1e-18
for iter = 1:60
    mid = 0.5 * (lo + hi);
    if ks_exact_cdf(n, mid) < target
        lo = mid;
    else
        hi = mid;
    end
end

d_crit = 0.5 * (lo + hi);

end  % ks_critical
