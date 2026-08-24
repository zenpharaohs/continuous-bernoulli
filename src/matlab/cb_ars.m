function eta = cb_ars(chi, nu, eta_star, n)
% CB_ARS  Adaptive Rejection Sampler with Gilks-Wild squeeze.
%
%   eta = cb_ars(chi, nu, eta_star, n)
%
%   Target (unnormalized log-density):
%     f(eta) = chi*eta - nu*B(eta),   eta in R
%
%   f is strictly log-concave (f'' = -nu*B'' < 0 everywhere).
%
%   Algorithm: vectorized overbooking + chord squeeze.
%
%   For each batch of proposals:
%     1. Draw eta_prop from upper hull H(eta)
%     2. Draw log_u = log(Uniform(0,1)) once per proposal
%     3. CHEAP TEST:  if log_u <= S(eta) - H(eta):  accept  [no bft_b call]
%     4. EXPENSIVE:   evaluate f(eta) = chi*eta - nu*B(eta)
%                     if log_u <= f(eta) - H(eta): accept, else reject
%
%   S(eta) = chord lower bound between adjacent hull points p_k, p_{k+1}
%   Valid by strict log-concavity of f.  Tail proposals always expensive.
%
%   Hull point count is adaptive in post_std = 1/sqrt(nu*B''(eta*)):
%     post_std <= 1.5:  5 pts (narrow posterior, few points needed)
%     post_std <= 3.0:  7 pts (moderate, tighter chord spacing)
%     post_std >  3.0:  9 pts (broad posterior, weakest squeeze case)
%
%   In all cases points are placed at uniform sigma multiples up to ~3.5 sigma.
%   The chord gap kappa*Delta^2/8 is approximately equalized across intervals
%   at the mode curvature kappa = nu*B''(eta*).
%
%   See also: cb_mode, cb_sample, bft_b, bft_all

OVERBOOK  = 1.2;
MIN_BATCH = 32;

% --- Build initial hull: adaptive point count based on post_std ---
[~, ~, bpp_star] = bft_all(eta_star);
kappa = nu * bpp_star;
sig   = 1.0 / sqrt(max(kappa, 1e-6));   % posterior std in eta-space

% Choose hull point offsets in sigma units.
% Target: chord gap kappa*Delta^2/8 ~ 0.15 per interval (squeeze rate ~86%).
% Delta_target = sqrt(8*0.15) * sig = sqrt(1.2) * sig ~ 1.10 * sig.
% We cap outer extent at 3.5 sigma for tail probability control.
if sig <= 1.5
    % Narrow: 5 points, Delta ~ 1.5 sigma
    offsets = [-3.0; -1.5; 0; 1.5; 3.0];
elseif sig <= 3.0
    % Moderate: 7 points, Delta ~ 1.0 sigma in interior
    offsets = [-3.5; -2.25; -1.0; 0; 1.0; 2.25; 3.5];
else
    % Broad: 9 points, Delta ~ 0.75 sigma in interior
    offsets = [-3.5; -2.5; -1.75; -0.875; 0; 0.875; 1.75; 2.5; 3.5];
end

pts  = eta_star + sig * offsets;
hull = hull_build(pts, chi, nu);

% --- Vectorized overbooking + squeeze loop ---
eta   = zeros(n, 1);
drawn = 0;

while drawn < n
    need       = n - drawn;
    batch_size = max(MIN_BATCH, ceil(OVERBOOK * need));

    % Draw proposals from upper hull
    [eta_prop, log_env] = hull_sample_batch(hull, batch_size);

    % Draw log-uniforms once per proposal (shared by both tests)
    log_u = log(rand(batch_size, 1));

    % --- Cheap squeeze test (no bft_b call) ---
    log_sq = chord_squeeze(hull, eta_prop, log_env);
    cheap_accept = log_u <= log_sq;

    % --- Expensive test only for proposals failing cheap test ---
    accept = cheap_accept;
    need_eval = ~cheap_accept;
    if any(need_eval)
        log_f = chi*eta_prop(need_eval) - nu*bft_b_vec(eta_prop(need_eval));
        accept(need_eval) = log_u(need_eval) <= log_f - log_env(need_eval);
    end

    % Compact accepted samples
    eta_acc = eta_prop(accept);
    take    = min(length(eta_acc), need);
    eta(drawn+1 : drawn+take) = eta_acc(1:take);
    drawn   = drawn + take;
end
end % cb_ars


% =========================================================================
%  Chord squeeze lower bound
% =========================================================================

function log_sq = chord_squeeze(hull, eta_prop, log_env)
% S(eta_prop) - H(eta_prop) for each proposal.
% S is the chord between adjacent hull points bracketing eta_prop.
% Returns -Inf for tail proposals.

m      = length(hull.pts);
np     = length(eta_prop);
k_vec  = sum(eta_prop > hull.pts', 2);   % bracket index in {0,...,m}
log_sq = -Inf(np, 1);

valid = k_vec >= 1 & k_vec <= m-1;
if any(valid)
    kv = k_vec(valid);
    ep = eta_prop(valid);
    dp = hull.pts(kv+1)   - hull.pts(kv);
    df = hull.fvals(kv+1) - hull.fvals(kv);
    S  = hull.fvals(kv) + (df./dp).*(ep - hull.pts(kv));
    log_sq(valid) = S - log_env(valid);
end
end


% =========================================================================
%  Vectorized hull sampler
% =========================================================================

function [eta_prop, log_env] = hull_sample_batch(hull, m)
n_segs = length(hull.pts);
probs  = exp(hull.logwts - hull.logZ);
probs  = probs / sum(probs);
edges  = [0; cumsum(probs(:))];
u_cat  = rand(m, 1);
segs   = sum(u_cat > edges(2:end-1)', 2) + 1;
segs   = min(segs, n_segs);

s_vec = hull.slopes(segs);
f_vec = hull.fvals(segs);
p_vec = hull.pts(segs);
a_vec = -Inf(m, 1);
b_vec =  Inf(m, 1);
for k = 1:m
    seg = segs(k);
    if seg > 1,      a_vec(k) = hull.zpts(seg-1); end
    if seg < n_segs, b_vec(k) = hull.zpts(seg);   end
end
eta_prop = sample_trunc_exp_vec(s_vec, a_vec, b_vec);
log_env  = f_vec + s_vec .* (eta_prop - p_vec);
end


% =========================================================================
%  Vectorized B(eta)
% =========================================================================

function b = bft_b_vec(eta)
b     = zeros(size(eta));
small = abs(eta) < 1.0;
if any(small(:)), b(small) = bft_pf(eta(small)); end
if any(~small(:))
    et = eta(~small);
    b(~small) = log(expm1(et) ./ et);
end
end


% =========================================================================
%  Vectorized truncated exponential sampler
% =========================================================================

function x = sample_trunc_exp_vec(s, a, b)
m   = length(s); x = zeros(m,1); u = rand(m,1); tol = 1e-8;
flat       = abs(s) < tol;
left_tail  = ~flat & s > 0 & isinf(a);
right_tail = ~flat & s < 0 & isinf(b);
fin_pos    = ~flat & s > 0 & ~isinf(a);
fin_neg    = ~flat & s < 0 & ~isinf(b);
if any(flat),       x(flat)       = a(flat) + u(flat).*(b(flat)-a(flat)); end
if any(left_tail),  x(left_tail)  = b(left_tail) + log(u(left_tail))./s(left_tail); end
if any(right_tail), x(right_tail) = a(right_tail) + log1p(-u(right_tail))./s(right_tail); end
if any(fin_pos)
    sp=s(fin_pos); ap=a(fin_pos); bp=b(fin_pos); up=u(fin_pos);
    x(fin_pos) = ap + log1p(up.*expm1(sp.*(bp-ap)))./sp;
end
if any(fin_neg)
    sn=s(fin_neg); an=a(fin_neg); bn=b(fin_neg); un=u(fin_neg);
    x(fin_neg) = an + log1p(-un.*(-expm1(sn.*(bn-an))))./sn;
end
end


% =========================================================================
%  Hull build
% =========================================================================

function hull = hull_build(pts, chi, nu)
pts = sort(pts(:)); m = length(pts);
fvals = zeros(m,1); slopes = zeros(m,1);
for i = 1:m
    [b, bp]   = bft_all(pts(i));
    fvals(i)  = chi*pts(i) - nu*b;
    slopes(i) = chi - nu*bp;
end

zpts = zeros(m-1, 1);
for i = 1:m-1
    ds = slopes(i) - slopes(i+1);
    if abs(ds) < 1e-14
        zpts(i) = 0.5*(pts(i)+pts(i+1));
    else
        zpts(i) = (fvals(i+1)-fvals(i) ...
                   + slopes(i)*pts(i) - slopes(i+1)*pts(i+1)) / ds;
    end
end

if slopes(1) <= 0
    error('cb_ars:hull_build:badLeftSlope', 'Left slope <= 0 (%.3e).', slopes(1));
end
if slopes(m) >= 0
    error('cb_ars:hull_build:badRightSlope', 'Right slope >= 0 (%.3e).', slopes(m));
end

logwts    = zeros(m,1);
logwts(1) = fvals(1) + slopes(1)*(zpts(1)  -pts(1)) - log( slopes(1));
logwts(m) = fvals(m) + slopes(m)*(zpts(m-1)-pts(m)) - log(-slopes(m));
for i = 2:m-1
    logwts(i) = log_seg_wt(fvals(i), slopes(i), pts(i), zpts(i-1), zpts(i));
end

hull.pts    = pts;
hull.fvals  = fvals;
hull.slopes = slopes;
hull.zpts   = zpts;
hull.logwts = logwts;
hull.logZ   = my_logsumexp(logwts);
end


function lw = log_seg_wt(f_i, s, p_i, a, b)
tol = 1e-8;
if abs(s) < tol, lw = f_i + log(b-a);
elseif s > 0
    arg = s*(a-b);
    if arg < -30, lw = f_i + s*(b-p_i) - log(s);
    else,         lw = f_i + s*(b-p_i) - log(s) + log1p(-exp(arg)); end
else
    arg = s*(b-a);
    if arg < -30, lw = f_i + s*(a-p_i) - log(-s);
    else,         lw = f_i + s*(a-p_i) - log(-s) + log1p(-exp(arg)); end
end
end


function lse = my_logsumexp(v)
m = max(v); lse = m + log(sum(exp(v-m)));
end
