function [x,flag, relres, iter, resvec] = pjm(A,b,tol,maxit)
% PJM Solve system of linear equations using the Jacobi Method
%Syntax:
%   'x = pcg(A,b)' attempts to solve the system of linear equations 'A*x = b' for 'x' using the Jacobi
%   Method. When the attempt is successful, pjm displays a message to state the convergence.
%   If pjm fails to converge after the maximum number of iterations, it displays a diagnostic
%   message, reporting the relative residual 'norm(b-A*x)/norm(b)' and the iteration number at which
%   the method stopped.
%
%   'x = pcg(A,b,tol)' specifies a tolerance for the method. The default value is '1e-6'.
%
%   'x = pcg(A,b,tol,maxit)' specifies the maximum number of iterations to use.
%   pjm displays a diagnostic message if it fails to converge within maxit iterations.
%
%   '[x,flag] = pjm(___)' returns a flag that specifies whether the method has converged.
%   When flag = 0, convergence was successful. When the flag output is specified, pjm does not
%   display any diagnostic messages.
%
%   '[x,flag,relres] = pjm(___)' also returns the relative residual norm(b-A*x)/norm(b).
%
%   '[x,flag,relres,iter] = pjm(___)' also returns the iteration number iter at which the algorithm
%   stopped
%
%
%   '[x,flag,relres,iter,resvec] = pjm(___)' also returns a vector of the residual norm at each
%   iteration.
%
% Input arguments:
%   - A (matrix) -
%   Coefficient matrix, specified as a symmetric positive definite matrix. This matrix is the
%   coefficient matrix in the linear system 'A*x = b'.
%
%   - b (column vector) -
%   Right side of linear equation, specified as a column vector.
%   The length of b must be equal to 'size(A,1)'.
%
%   - tol (positive scalar, optional) -
%   Method tolerance, specified as a positive scalar. 'pjm' must meet the tolerance within the number
%   of allowed iterations to be successful. A smaller value of 'tol' means the answer must be more
%   precise for the calculation to be successful.
%
%   - maxit (positive integer, optional) -
%   Maximum number of iterations, specified as a positive scalar integer.
%   Increase the value of maxit to allow more iterations for 'pjm' in order to achieve the tolerance
%   desired. Generally, a smaller value of 'tol' means more iterations are required by the method
%   to converge.

arguments (Input)
    A {mustBeMatrix, mustBeSquared, mustBeSymmetric, mustBeDefinite}
    b {mustBeColumn}
    tol (1,1) {mustBeReal, mustBePositive} = 1e-6
    maxit (1,1) {mustBeInteger, mustBePositive} = 10000
end

if(size(A,1) ~= length(b))
    errID = 'pjm:NotEqualSize';
    msg = 'Coefficient matrix A and constant terms column vector b must have compatible sizes.';
    throw(MException(errID,msg));
end

minArgs = 1;
maxArgs = 5;
nargoutchk(minArgs,maxArgs);

if isempty(gcp('nocreate'))
    parpool;
end

A_dist = distributed(A);
b_dist = distributed(b);
P_dist = distributed(diag(diag(A)));
x_dist = distributed(zeros(n,1));

r_dist = b_dist - A_dist * x_dist;
res = norm(r_dist) / norm(b_dist);

resvec = norm(r_dist);

while (res > tol && k < maxit)

    k = k+1;
    z_dist = P_dist \ r_dist;
    alpha = 1;
    x_dist = x_dist + alpha * z_dist;
    r_dist = r_dist - alpha * A_dist * z_dist;

    res = norm(r_dist) / norm(b_dist);

    resvec = [resvec norm(r_dist)];
end

relres = res;

if(nargout >= 2)
    if(k<=maxit && relres<=tol)
        flag = 0;
    elseif(k==maxit && relres>tol)
        flag = 1;
    end
end

if(nargout >= 4)
    iter = k;
end

end

function mustBeSymmetric(A)
simm = issymmetric(A);
if(simm ~= 1)
    errID = 'pjm:NotSymmetric';
    msg = 'Coefficient matrix A must be symmetric.';
    throw(MException(errID,msg));
end
end

function mustBeDefinite(A)
d = eig(A);
if (~(all(d > 0)))
    errID = 'pjm:NotDefinite';
    msg = 'Coefficient matrix A must be positive definite.';
    throw(MException(errID,msg));
end
end

function mustBeSquared(A)
if(diff(size(A)))
    errID = 'pjm:NotSquared';
    msg = 'Coefficient matrix A must be squared.';
    throw(MException(errID,msg));
end
end
