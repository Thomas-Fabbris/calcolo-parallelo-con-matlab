%PJM - Solve system of linear equations - Parallel Jacobi Method
%    This MATLAB function attempts to solve the system of linear equations
%    A*x = b for x using a parallel implementation of the Jacobi Method.
%
%Syntax
%    x = PJM(A,b)
%    x = PJM(A,b,tol)
%    x = PJM(A,b,tol,maxit)
%    x = PJM(A,b,tol,maxit,x0)
%    [x,flag] = PJM(___)
%    [x,flag,relres] = PJM(___)
%    [x,flag,relres,iter] = PJM(___)
%    [x,flag,relres,iter,resvec] = PJM(___)
%
%Input Arguments
%    A - Coefficient matrix
%        matrix
%    b - Right side of linear equation
%        column vector
%    tol - Method tolerance
%        [] or 1e-6 (default) | positive scalar
%    maxit - Maximum number of iterations
%        [] or 100 (default) | positive scalar integer
%    x0 - Initial Guess
%        [] or column vector of zeros (default) | column vector
%
%Output Arguments
%    x - Linear system solution
%        column vector
%    flag - Convergence flag
%        scalar
%    relres - Relative residual error
%        scalar
%    iter - Iteration number
%        scalar
%    resvec - Residual error
%        vector
%
%See also parpool, distributed, gather

function [x,flag, relres, iter, resvec] = pjm(A,b,tol,maxit,x0)
arguments (Input)
    A {mustBeMatrix, mustBeFloat, mustBeReal, mustBeSquared}
    b {mustBeColumn, mustBeFloat, mustBeReal, mustBeSameSize(b,A,'Right-hand side')}
    tol {mustBeScalarOrEmpty, mustBeFloat, mustBePositive} = 1e-6
    maxit {mustBeScalarOrEmpty, mustBeInteger, mustBePositive} = 1000
    x0 {mustBeColumn, mustBeFloat, mustBeSameSize(x0,A,'Initial guess')} = zeros(size(A,1),1)
end

arguments (Output)
    x (:,1) {mustBeColumn, mustBeFloat}
    flag (1,1) {mustBeMember(flag, [0,1,2])}
    relres (1,1) {mustBeFloat, mustBeBetween(relres,0,1)}
    iter (1,1) {mustBeInteger, mustBeNonnegative}
    resvec (:,1) {mustBeColumn, mustBeFloat, mustBeNonnegative}
end

nargoutchk(0,5);
isInputDistributed = isa(A, 'distributed') || isa(b, 'distributed') || isa(x0, 'distributed');

normB = norm(b);

if (normB == 0)
    x = x0; 
    flag = 0; 
    relres = 0; 
    iter = 0; 
    resvec = zeros(1,1);
    
    if(nargout < 2)
        fprintf(['The right-hand side vector is all zero so pjm returned the initial solution ' ...
            ' without iterating.']);
    end
    
    return;
end

A_dist = distributed(A);
b_dist = distributed(b);
x_dist = distributed(x0);
d_dist = diag(A_dist);
r_dist = b_dist - A_dist*x_dist;

norm_r_dist = norm(r_dist);
relres = norm_r_dist / normB;
resvec = zeros(maxit+1, 1);
resvec(1) = norm_r_dist;
iter = 0;
flag = 1;

while (relres > tol && iter < maxit)
    iter = iter+1;

    x_dist = x_dist + r_dist ./d_dist;
    r_dist = b_dist - A_dist * x_dist;
    norm_r_dist = norm(r_dist);

    if(isnan(norm_r_dist) || isinf (norm_r_dist))
        flag = 2;
        break;
    end
    
    resvec(iter+1) = norm_r_dist;
    relres = norm_r_dist / normB;
end

if ~isInputDistributed
    x = gather(x_dist);
else
    x = x_dist;
end

resvec = resvec(1:iter + 1);

if(relres<=tol)
    flag = 0;
end

if(nargout < 2)
    switch flag
        case 0
            fprintf("pjm converged at iteration %u to a solution with relative residual %.4e." ...
                , iter,relres);
        case 1
            fprintf("pjm stopped at iteration %u without converging to the desired tolerance " + ...
                "%.4e because the maximum number of iterations %u was reached.\n The iterate " + ...
                "returned (number %u) has relative residual %.4e.", iter,tol,maxit,relres);
        case 2
            fprintf("pjm stopped at iteration %u without converging to the desired tolerance " + ...
                "%.4e a scalar quantity became too small or too large to continue computing.\n " + ...
                "The iterate returned (number %u) has relative residual %.4e.", iter,tol,relres);
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

function mustBeSameSize(v,A,name)
n = size(A,1);
if (~isequal(n, size(v,1)))
    errID = 'pjm:DimensionMismatch';
    msg = sprintf(['%s must be a column vector of length %u to match the ' ...
        'problem size'],name,n);
    throw(MException(errID,msg));
end
end