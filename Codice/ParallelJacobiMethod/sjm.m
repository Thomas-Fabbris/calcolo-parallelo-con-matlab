function [x,flag, relres, iter, resvec] = sjm(A,b,tol,maxit)
arguments (Input)
    A {mustBeMatrix, mustBeFloat, mustBeSquared}
    b {mustBeColumn, mustBeFloat}
    tol (1,1) {mustBeFloat, mustBePositive} = PjmHelper.DefaultTol
    maxit (1,1) {mustBeInteger, mustBePositive} = PjmHelper.DefaultMaxit
end

arguments (Output)
    x (:,1) {mustBeFloat}
    flag (1,1) {mustBeMember(flag, [0, 1])}
    relres (1,1) {mustBeFloat, mustBeBetween(relres, 0, 1)}
    iter (1,1) {mustBeInteger, mustBeNonNegative}
    resvec (:,1) {mustBeFloat, mustBeNonnegative}
end

nargoutchk(PjmHelper.MinOutArgs,PjmHelper.MaxOutArgs);

n = size(A,1);
if n ~= length(b)
    errID = 'sjm:DimensionMismatch';
    msg = ['Right-hand side must be a column vector of length 1000 to match the coefficient ' ...
        'matrix'];
    throw(MException(errID,msg));
end

x=zeros(n,1);
norm_b=norm(b);

if norm_b == 0
    flag = 0; relres = 0; iter = 0; resvec = 0;
    if(nargout < 2)
        fprintf(['The right hand side vector is all zero so pjm returned an all zero solution ' ...
            '            without iterating.']);
    end
    return;
end

P = diag(diag(A));

r = b;
norm_r = norm_b;

relres = 1;
resvec = zeros(maxit+1, 1);
resvec(1) = norm_r;
k = 0;
flag = 1;

while (relres > tol && k < maxit)
    k = k+1;
    z = P\r;
    x = x + z;
    r = r - A * z;

    norm_r = norm(r);
    resvec(k+1) = norm_r;

    relres = norm_r / norm_b;
end

resvec = resvec(1:k+1);
iter = k;

if(relres<=tol)
    flag = 0;
    if(nargout < 2)
        fprintf("sjm converged at iteration %u to a solution with relative residual %.4e.", k, ...
            relres);
    end
else
    if(nargout < 2)
        fprintf("sjm stopped at iteration %u without converging to the desired tolerance %.4e " + ...
            "because the maximum number of iterations %u was reached.\n The iterate returned " + ...
            "(number %u) has relative residual %.4e.", k,tol,maxit,relres);
    end
end

end

function mustBeSquared(A)
if(diff(size(A)))
    errID = 'sjm:NotSquared';
    msg = 'Coefficient matrix must be squared.';
    throw(MException(errID,msg));
end
end

