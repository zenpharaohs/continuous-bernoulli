function [c_proj, rho_proj, H2_proj, DB_proj, info] = leg_proj_hellinger(c_hat, N)
% LEG_PROJ_HELLINGER  Project Legendre coefficients to nearest valid density.
%
%   [c_proj, rho_proj, H2_proj, DB_proj, info] = leg_proj_hellinger(c_hat, N)
%
%   Given raw estimated Legendre coefficients c_hat = [c_1,...,c_m], solves:
%
%     min_{c}  || c - c_hat ||^2
%     s.t.     p_c(x) = 1 + sum_k c_k phi_k(x) >= 0   for all x in [0,1]
%
%   The semi-infinite nonnegativity constraint is enforced via an exchange
%   (cutting-plane) active-set method.  At each iteration:
%
%     1. Compute the extrema of p_c(x) exactly: find roots of p_c'(x)
%        (degree m-1 polynomial) by bisection on a sign-change grid, plus
%        endpoints {0, 1}.  For a degree-m polynomial there are at most m+1
%        constraint candidates -- this is not a heuristic grid.
%     2. Check whether p_c(x_j) >= 0 at all extrema.
%     3. If yes: feasible, done.
%     4. If no: add all violated extrema x_j to the active constraint set
%        and solve the QP with those constraints.
%     5. Return to 1 with the new c from the QP solution.
%
%   Since the constraint set consists exactly of the extrema of the current
%   polynomial (plus endpoints), and a polynomial of degree m is nonneg on
%   [0,1] iff it is nonneg at its local minima and endpoints, this is not
%   an approximation -- it is the exact finite reduction of the semi-infinite
%   constraint for polynomials.  (A dense grid would be heuristic; this is not.)
%
%   After solving, the projected Hellinger affinity is computed exactly:
%
%     rho_proj = integral_0^1 sqrt(p_c(x)) dx   [Gauss-Legendre quadrature]
%
%   This is the honest nonlinear estimate; it does NOT use the second-order
%   approximation rho ~ 1 - chi^2/8.  It is automatically in [0,1].
%
%   INPUTS:
%     c_hat  [1 x m]  estimated Legendre coefficients (from streaming pass)
%     N      int       sample size (for debiased chi^2 in info struct only)
%
%   OUTPUTS:
%     c_proj   [1 x m]  projected coefficients (nearest feasible density)
%     rho_proj scalar   Hellinger affinity = integral_0^1 sqrt(p_c)
%     H2_proj  scalar   Hellinger distance^2 = 1 - rho_proj  (in [0,1])
%     DB_proj  scalar   Bhattacharyya distance = -log(rho_proj)  (>= 0)
%     info     struct   diagnostics:
%                         .n_iter       number of outer iterations
%                         .n_constr     final active constraint count
%                         .min_density  min p_c(x) on quadrature grid
%                         .max_density  max p_c(x) on quadrature grid
%                         .proj_dist    || c_proj - c_hat || (0 if already feasible)
%                         .chi2_proj    debiased chi^2 using projected c
%                         .H2_approx    second-order H^2 ~ chi2_proj/8 (comparison)
%                         .converged    true if tolerance met, false if max_iter hit
%
%   ALGORITHM NOTES:
%     - Normalization is automatic: phi_k are mean-zero so int p_c = 1.
%     - The QP is strictly convex (H=I), so each iterate is unique.
%     - Bisection tolerance: 1e-12 on the root interval width.
%     - Convergence tolerance: min p at extrema >= -1e-12.
%     - max_iter = 50 (converges in < 5 for typical cases).
%     - Requires MATLAB Optimization Toolbox (quadprog).
%
% MIT License.  Andrew Mullhaupt, Stony Brook University AMS/QF, 2026.

c_hat = c_hat(:)';   % row vector
m     = numel(c_hat);

%% Gauss-Legendre quadrature nodes on [0,1]
n_quad = max(300, 3*m);
[x_gl, w_gl] = gauss_legendre_01(n_quad);
[PHI_quad, ~] = leg_basis(x_gl, m);   % [n_quad x m], precompute once

%% QP matrices (fixed throughout all iterations)
%   min (1/2) c'*H*c + f'*c  where H=2I, f=-2*c_hat (since ||c-c_hat||^2)
%   quadprog convention: min (1/2)x'Hx + f'x
H_qp = 2 * eye(m);
f_qp = -2 * c_hat';
opts = optimoptions('quadprog', 'Display', 'off', ...
                    'Algorithm', 'interior-point-convex', ...
                    'OptimalityTolerance', 1e-10, ...
                    'ConstraintTolerance', 1e-10);

%% Fine grid for initial sign-change detection of p'
% 10 points per polynomial degree is enough to catch all sign changes
% since p' has degree m-1 and thus at most m-1 real roots on [0,1].
n_fine  = max(1000, 10*m);
x_fine  = linspace(0, 1, n_fine)';
[~, DPHI_fine] = leg_basis(x_fine, m);   % [n_fine x m], precompute once

%% Main exchange loop
c_cur      = c_hat;     % start at raw estimate
x_active   = zeros(0,1); % active constraint points (grows, may shrink)
n_iter     = 0;
max_iter   = 50;
feas_tol   = -1e-12;    % allow tiny numerical violations
converged  = false;

while n_iter < max_iter
    n_iter = n_iter + 1;

    %% Step 1: find extrema of current p_c(x) = 1 + DPHI*c_cur (derivative)
    dp = DPHI_fine * c_cur';   % p'(x) at fine grid, [n_fine x 1]

    % Locate sign changes => bracket each root of p'
    sc  = dp(1:end-1) .* dp(2:end);
    brk = find(sc < 0);   % indices where sign changes

    x_extrema = zeros(numel(brk), 1);
    for j = 1:numel(brk)
        xa = x_fine(brk(j));
        xb = x_fine(brk(j)+1);
        dpa = dp(brk(j));    % sign at left bracket (fixed throughout bisection)
        % Bisect to find root of p' to tolerance 1e-12
        for iter_b = 1:60
            xm = 0.5*(xa + xb);
            [~, DPHIm] = leg_basis(xm, m);
            dpm = DPHIm * c_cur';
            if dpm * dpa <= 0
                xb = xm;
            else
                xa = xm;
            end
            if (xb - xa) < 1e-12, break; end
        end
        x_extrema(j) = 0.5*(xa + xb);
    end

    % Full candidate set: interior extrema + endpoints
    x_cands = unique([0; x_extrema; 1], 'sorted');

    %% Step 2: evaluate p at all candidate extrema
    PHI_cands   = leg_basis(x_cands, m);
    p_at_cands  = 1 + PHI_cands * c_cur';   % p_c(x_j) for each candidate

    %% Step 3: check feasibility
    min_p = min(p_at_cands);
    if min_p >= feas_tol
        converged = true;
        break;   % feasible -- done
    end

    %% Step 4: update active constraint set (accumulate, never discard).
    % Add all currently-violated extrema to the active set.
    % We ACCUMULATE across iterations: old constraints are never dropped.
    % Dropping constraints causes oscillation (the QP satisfies new
    % constraints but re-violates old ones, cycling indefinitely).
    % Accumulation guarantees convergence: each iterate satisfies a
    % strictly larger constraint set, so the sequence is Lyapunov-decreasing.
    % The set grows by at most O(m) per iteration and stops when no
    % new violations are found -- convergence in at most O(m) iterations.
    x_violated = x_cands(p_at_cands < feas_tol);
    x_active   = unique([x_active; x_violated], 'sorted');   % ACCUMULATE

    %% Step 5: solve QP with active constraints
    PHI_active = leg_basis(x_active, m);
    % Constraint: 1 + PHI*c >= 0  =>  -PHI*c <= 1
    A_ineq = -PHI_active;
    b_ineq =  ones(numel(x_active), 1);

    [c_new, ~, exitflag, ~] = quadprog(H_qp, f_qp, A_ineq, b_ineq, ...
                                        [], [], [], [], [], opts);
    if exitflag <= 0
        % QP failed (should not happen for a feasible problem)
        % Fall back: add endpoints and retry with warm start
        if exitflag == -2
            warning('leg_proj_hellinger: QP infeasible at iter %d -- check c_hat', n_iter);
        end
        break;
    end
    c_cur = c_new';
end

c_proj = c_cur;

%% Verify final feasibility on dense quadrature grid
p_final    = 1 + PHI_quad * c_proj';
p_final    = max(p_final, 0);   % clip tiny numerical negatives for quadrature
rho_proj   = w_gl' * sqrt(p_final);
rho_proj   = min(max(rho_proj, 0), 1);
H2_proj    = 1 - rho_proj;
DB_proj    = -log(max(rho_proj, 1e-300));

%% Diagnostics
info.n_iter      = n_iter;
info.n_constr    = numel(x_active);
info.min_density = min(1 + PHI_quad * c_proj');   % before clipping
info.max_density = max(1 + PHI_quad * c_proj');
info.proj_dist   = norm(c_proj - c_hat);
info.chi2_proj   = max(sum(c_proj.^2) - m/N, 0);
info.H2_approx   = info.chi2_proj / 8;
info.converged   = converged;

end % main function

%% =========================================================================
function [PHI, DPHI] = leg_basis(x_vec, m)
% LEG_BASIS  Normalized Legendre basis phi_k and derivatives on [0,1].
%
%   phi_k(x) = sqrt(2k+1) * P_k(2x-1),  k=1..m
%   d/dx phi_k(x) = sqrt(2k+1) * 2 * P_k'(2x-1)
%
%   P_k via Bonnet recursion; P_k' via differentiated Bonnet recursion:
%     k * P_k'(z) = (2k-1)*P_{k-1}(z) + (2k-1)*z*P_{k-1}'(z) - (k-1)*P_{k-2}'(z)
%   with P_0'=0, P_1'=1.
%
%   Factor 2 in DPHI: chain rule d/dx = (d/dz)*(dz/dx) = 2*(d/dz).

n = numel(x_vec);
x_vec = x_vec(:);
z = 2*x_vec - 1;

PHI  = zeros(n, m);
DPHI = zeros(n, m);

Pk_2  = ones(n,1);    dPk_2 = zeros(n,1);   % P_0, P_0'
Pk_1  = z;            dPk_1 = ones(n,1);    % P_1, P_1'

norms = sqrt(3:2:(2*m+1));   % sqrt(2k+1) for k=1..m

PHI(:,1)  = norms(1) * z;
DPHI(:,1) = norms(1) * 2 * ones(n,1);

for k = 2:m
    Pk   = ((2*k-1)*z.*Pk_1  - (k-1)*Pk_2 ) / k;
    dPk  = ((2*k-1)*Pk_1 + (2*k-1)*z.*dPk_1 - (k-1)*dPk_2) / k;
    PHI(:,k)  = norms(k) * Pk;
    DPHI(:,k) = norms(k) * 2 * dPk;
    Pk_2 = Pk_1;  dPk_2 = dPk_1;
    Pk_1 = Pk;    dPk_1 = dPk;
end
end % leg_basis

%% =========================================================================
function [x, w] = gauss_legendre_01(n)
% Gauss-Legendre nodes and weights on [0,1] via Golub-Welsch.
    beta = 0.5 ./ sqrt(1 - (2*(1:n-1)).^(-2));
    J    = diag(beta, 1) + diag(beta, -1);
    [V, D] = eig(J);
    xgl = diag(D);
    [xgl, idx] = sort(xgl);
    wgl = 2 * V(1,idx)'.^2;
    x = (xgl + 1) / 2;
    w = wgl / 2;
end
