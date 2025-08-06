%JACOBI - Solve system of linear equations - Jacobi Method
%    This MATLAB function attempts to solve the system of linear equations
%    A*x = b for x using an implementation of the Jacobi Method.
%    The right hand side column vector b must have a compatible length with
%    the n-by-n coefficient matrix A.
%
%    Syntax
%        x = JACOBI(A,b)
%        x = JACOBI(A,b,tol)
%        x = JACOBI(A,b,tol,maxit)
%        x = JACOBI(A,b,tol,maxit,x0)
%        [x,flag] = JACOBI(___)
%        [x,flag,relres] = JACOBI(___)
%        [x,flag,relres,iter] = JACOBI(___)
%        [x,flag,relres,iter,resvec] = JACOBI(___)
%
%    Input Arguments
%        A - Coefficient matrix
%            matrix
%        b - Right side of linear equation
%            column vector
%        tol - Method tolerance
%            [] or 1e-6 (default) | positive scalar
%        maxit - Maximum number of iterations
%            [] or 100 (default) | positive scalar integer
%        x0 - Initial guess
%            [] or column vector of zeros (default) | column vector
%
%    Output Arguments
%        x - Linear system solution
%            column vector
%        flag - Convergence flag
%            scalar
%        relres - Relative residual error
%            scalar
%        iter - Iteration number
%            scalar
%        resvec - Residual error
%            vector
%

%    Copyright 2025 Thomas Fabbris

function [x,flag, relres, iter, resvec] = jacobi(A,b,tol,maxit,x0)
arguments (Input)
    A {mustBeMatrix, mustBeFloat, mustBeReal, mustBeSquared}
    b {mustBeColumn, mustBeFloat, mustBeReal, mustBeSameSize(b,A,'Right-hand side')}
    tol {mustBeScalarOrEmpty, mustBeFloat, mustBePositive} = []
    maxit {mustBeScalarOrEmpty, mustBeInteger, mustBeNonnegative} = 20
    x0 {mustBeColumn, mustBeFloat, mustBeSameSize(x0,A,'Initial guess')} = zeros(size(A,1),1)
end

arguments (Output)
    x (:,1) {mustBeColumn, mustBeFloat}
    flag (1,1) {mustBeMember(flag, [0,1,2,3])}
    relres (1,1) {mustBeFloat, mustBeBetween(relres,0,1)}
    iter (1,1) {mustBeInteger, mustBeNonnegative}
    resvec (:,1) {mustBeColumn, mustBeFloat, mustBeNonnegative}
end

maxit = max(maxit, 0);

x = x0;

useSingle = matlab.internal.feature("SingleSparse") && (isUnderlyingType(A,'single') || ...
    isUnderlyingType(b,'single') || isUnderlyingType(x,'single'));

if (useSingle)
    % Cast b and x to single
    b = single(b);
    x = single(x);
end

% Helper arrays should be dense, real, and of the same underlying type than b
prototype = real(full(zeros("like", b)));

if (isempty(tol))
    if (useSingle)
        tol = 1e-3;
    else
        tol = 1e-6;
    end
end
epsT = eps("like", prototype);
if (tol <= epsT)
    warning('MATLAB:jacobi:tooSmallTolerance', ['Tolerance may not be achievable. ' ...
        'Use a larger tolerance']);
    tol = epsT;
elseif (tol>= 1)
    warning('MATLAB:jacobi:tooBigTolerance', ['Tolerance is greater than 1. ' ...
        'Use a smaller tolerance']);
    tol = 1 - epsT;
end

% Check for all zero right hand side vector
norm2b = norm(b);                             % norm of right hand side vector b
if (norm2b == 0)                              % if b is null
    n = size(A,1);
    x = zeros(n,1,"like", prototype);         % then solution is all zeros
    flag = 0;                                 % a valid solution has been obtained
    relres = zeros("like", prototype);        % the relative residual is set to zero for convenience
    iter = 0;                                 % no iterations needed
    resvec = zeros(n,1,"like", prototype);    % resvec(1) = norm(b-A*x) = norm(0)
    if (nargout < 2)
        fprintf(['The right-hand side vector is all zero so jacobi returned the initial solution ' ...
            ' without iterating.']);
    end
    return
end

% Initialize variables for the method
flag = 1;                                     % assume convergence until failure
tolb = tol * norm2b;                          % relative tolerance
r = b - A * x;
normr = norm(r);                              % norm of residual vector r

% Initial guess is a good solution
if (normr <= tolb)
    flag = 0;
    relres = normr / norm2b;
    iter = 0;
    resvec = normr;
    if (nargout < 2)
        fprintf(['The initial guess has relative residual %0.2g which is within the desired ' ...
            'tolerance %0.2g so jacobi returned it without iterating.'], relres, tol);
    end
    return
end

d = diag(A);

if (~any(d))
    flag = 2;
    relres = normr / norm2b;
    iter = 0;
    resvec = normr;
    if (nargout < 2)
        fprintf("jacobi stopped at iteration %u without converging to the desired tolerance " + ...
            "%0.2g a scalar quantity became too small or too large to continue computing.\n" + ...
            "The iterate returned (number %u) has relative residual %0.2g.", iter,tol,relres);
    end
    return;
end

resvec = zeros(maxit+1,1,"like", prototype);  % preallocate vector for norm of residuals
resvec(1,:) = normr;                          % resvec(1) = norm(b-A*x0)
stag = 0;                                     % stagnation of the method
maxstagsteps = 3;

% Loop over maxit iterations (unless convergence or failure)

for ii = 1 : maxit
    if (isinf(normr))
        flag = 2;
        iter = ii;
        break
    end
    
    x_update = r./ d;
    
    % Check for stagnation of the method
    if (norm(x_update) < epsT * norm(x))
        stag = stag + 1;
    else
        stag = 0;
    end
    
    x = x + x_update;                         % compute new iterate
    r = b - A * x;
    normr = norm(r);
    resvec(ii+1,1) = normr;
    
    % Check for convergence
    if (normr <= tolb)
        flag = 0;
        iter = ii;
        break
    end
    if (stag >= maxstagsteps)
        flag = 3;
        iter = ii;
        break
    end
end                                       

% Return the solution
relres = normr / norm2b;                      % calculate the relative residual

if (flag == 1)
    iter = ii;
end

% Truncate the zeros from resvec
if ((flag <= 1) || (flag == 3))
    resvec = resvec(1:ii+1,:);
else
    resvec = resvec(1:ii,:);
end

% Display a message if the output flag is missing
if (nargout < 2)
    switch flag
        case 0
            fprintf("jacobi converged at iteration %u to a solution with relative residual %0.2g." ...
                , iter,relres);
        case 1
            fprintf("jacobi stopped at iteration %u without converging to the desired tolerance " + ...
                "%0.2g because the maximum number of iterations %u was reached.\nThe iterate " + ...
                "returned has relative residual %0.2g.", iter,tol,relres);
        case 2
            fprintf("jacobi stopped at iteration %u without converging to the desired tolerance " + ...
                "%0.2g because a scalar quantity became too small or too large to continue computing.\n " + ...
                "The iterate returned has relative residual %0.2g.", iter,tol,relres);
        case 3
            fprintf("jacobi stopped at iteration %u without converging to the desired tolerance " + ...
                "%0.2g because the method stagnated.\n" + ...
                "The iterate returned has relative residual %0.2g.",iter,tol,relres);
    end
end
end

function mustBeSquared(A)
[m, n] = size(A);
if (m ~= n)
    error('MATLAB:jacobi:NonSquareMatrix','Coefficient matrix must be squared.');
end
end

function mustBeSameSize(v,A,name)
[m,n] = size(A);
if (~isequal([m,1], size(v)))
    error('MATLAB:jacobi:DimensionMismatch', ['%s must be a column vector of length %u to match the ' ...
        'problem size.'],name,n);
end
end