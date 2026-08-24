function [PHI, DPHI] = leg_basis(x_vec, m)
% LEG_BASIS  Evaluate normalized Legendre basis and derivatives on [0,1].
%
%   [PHI, DPHI] = leg_basis(x_vec, m)
%
%   phi_k(x) = sqrt(2k+1) * P_k(2x-1),  k=1..m
%   phi_k'(x) = sqrt(2k+1) * 2 * dP_k/dz|_{z=2x-1}
%
%   Uses three-term recurrences for P_k and dP_k/dz.
%   Derivative recurrence (from differentiating Bonnet's recursion):
%     k * P_k'(z) = (2k-1)*P_{k-1}(z) + (2k-1)*z*P_{k-1}'(z) - (k-1)*P_{k-2}'(z)
%   with P_0'=0, P_1'=1.
%
%   Returns:
%     PHI  [n x m]  basis values
%     DPHI [n x m]  basis derivatives w.r.t. x (not z)
%
% MIT License.  Andrew Mullhaupt, Stony Brook University AMS/QF, 2026.

n = numel(x_vec);
x_vec = x_vec(:);
z = 2*x_vec - 1;   % map [0,1] -> [-1,1]

PHI  = zeros(n, m);
DPHI = zeros(n, m);

P_prev2  = ones(n,1);    % P_0 = 1
P_prev1  = z;             % P_1 = z
dP_prev2 = zeros(n,1);   % P_0' = 0
dP_prev1 = ones(n,1);    % P_1' = 1

% k=1: phi_1 = sqrt(3)*P_1(z) = sqrt(3)*z
PHI(:,1)  = sqrt(3)*z;
DPHI(:,1) = sqrt(3)*2*ones(n,1);  % d/dx phi_1 = sqrt(3)*2 (chain rule: d/dx = 2*d/dz)

for k = 2:m
    % Value recurrence: k*P_k = (2k-1)*z*P_{k-1} - (k-1)*P_{k-2}
    P_cur = ((2*k-1)*z.*P_prev1 - (k-1)*P_prev2) / k;
    % Derivative recurrence: k*P_k' = (2k-1)*P_{k-1} + (2k-1)*z*P_{k-1}' - (k-1)*P_{k-2}'
    dP_cur = ((2*k-1)*P_prev1 + (2*k-1)*z.*dP_prev1 - (k-1)*dP_prev2) / k;
    PHI(:,k)  = sqrt(2*k+1) * P_cur;
    DPHI(:,k) = sqrt(2*k+1) * 2 * dP_cur;  % factor 2: d/dx = 2*(d/dz)
    P_prev2  = P_prev1;  P_prev1  = P_cur;
    dP_prev2 = dP_prev1; dP_prev1 = dP_cur;
end
end
