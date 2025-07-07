function [x,flag, relres, iter, resvec] = pjm(A,b,tol,maxit)
%PJM - Solve system of linear equations - Parallel Jacobi Method
%   This MATLAB function attempts to solve the system of linear equations
%   A*x = b for x using a parallel implementation of the Jacobi Method.
%
%Syntax
%   x = PJM(A,b)
%   x = PJM(A,b,tol)
%   x = PJM(A,b,tol,maxit)
%   [x,flag] = PJM(___)
%   [x,flag,relres] = PJM(___)
%   [x,flag,relres,iter] = PJM(___)
%   [x,flag,relres,iter,resvec] = PJM(___)
%
%Input Arguments
%   A - Coefficient matrix
%     matrix
%   b - Right side of linear equation
%     column vector
%   tol - Method tolerance
%     [] or 1e-6 (default) | positive scalar
%   maxit - Maximum number of iterations
%     [] or min(size(A,1),100) (default) | positive scalar integer
%
%Output Arguments
%   x - Linear system solution
%     column vector
%   flag - Convergence flag
%     scalar
%   relres - Relative residual error
%     scalar
%   iter - Iteration number
%     scalar
%   resvec - Residual error
%     vector
%
%See also parpool, gcp, distributed, gather

arguments (Input)
    A {mustBeMatrix, mustBeFloat, mustBeSquared}
    b {mustBeColumn, mustBeFloat}
    tol (1,1) {mustBeFloat, mustBePositive} = PjmHelper.DefaultTol
    maxit (1,1) {mustBeInteger, mustBePositive} = PjmHelper.DefaultMaxit
end

arguments (Output)
    x (:,1) {mustBeFloat}
    flag (1,1) {mustBeMember(flag, [0, 3])}
    relres (1,1) {mustBeFloat, mustBeBetween(relres, 0, 1)}
    iter (1,1) {mustBeInteger, mustBePositive}
    resvec (:,1) {mustBeFloat, mustBeNonnegative}
end

nargoutchk(PjmHelper.MinOutArgs,PjmHelper.MaxOutArgs);

n = size(A,1);
if n ~= length(b)
    errID = 'pjm:DimensionMismatch';
    msg = ['Right-hand side must be a column vector of length 1000 to match the coefficient ' ...
        'matrix'];
    throw(MException(errID,msg));
end

maxit = min(n, maxit);

if isempty(gcp('nocreate'))
    parpool;
end

x_dist = distributed(zeros(n,1));
b_dist = distributed(b);
norm_b_dist = norm(b_dist);

if norm_b_dist == 0
    x = gather(x_dist); flag = 0; relres = 0; iter = 0; resvec = 0;
    if(nargout < 2)
        fprintf(['The right hand side vector is all zero so pjm returned an all zero solution ' ...
            '            without iterating.']);
    end
    return;
end

A_dist = distributed(A);
P_dist = distributed(diag(A));

if(any(P_dist == 0))
    warning('pjm:ZeroDiagonal', 'The coefficient matrix has at least one zero on its main diagonal. ' + ...
        'This can lead to unexpected output.')
end

r_dist = b_dist;
norm_r_dist = norm_b_dist;

relres = 1;
resvec = zeros(maxit+1, 1);
resvec(1) = norm_r_dist;
k = 0;
flag = 1;

while (relres > tol && k < maxit)

    k = k+1;
    z_dist = r_dist ./ P_dist;
    x_dist = x_dist + z_dist;
    r_dist = r_dist - A_dist * z_dist;

    norm_r = norm(r_dist);
    resvec(k+1) = norm_r;

    relres = norm_r / norm_b_dist;
end

x = gather(x_dist);
resvec = resvec(1:k+1);
iter = k;

if(relres<=tol)
    flag = 0;
    if(nargout < 2)
        fprintf("pjm converged at iteration %u to a solution with relative residual %.4e.", k, ...
            relres);
    end
else
    if(nargout < 2)
        fprintf("pjm stopped at iteration %u without converging to the desired tolerance %.4e " + ...
            "because the maximum number of iterations %u was reached.\n The iterate returned " + ...
            "(number %u) has relative residual %.4e.", k,tol,maxit,relres);
    end
end

end

function mustBeSquared(A)
if(diff(size(A)))
    errID = 'pjm:NotSquared';
    msg = 'Coefficient matrix must be squared.';
    throw(MException(errID,msg));
end
end

