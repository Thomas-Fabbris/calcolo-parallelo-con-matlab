function [x,flag, relres, iter, resvec] = jacobi(A,b,tol,maxit,x0)
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

if (nargin < 2)
    error('jacobi:NotEnoughInputs', 'Not enough input arguments.');
end

support = matlab.internal.feature("SingleSparse");

% A must be a floating point real matrix
if (~isfloat(A) || ~isreal(A))
   error('jacobi:InvalidInput', 'Argument must be a floating-point matrix of real elements.');
end

[m, n] = size(A);

if (m ~= n)
    error('jacobi:NonSquareMatrix', 'Matrix must be square.');
end
if (~isequal(size(b),[m,1]))
  error('jacobi:RHSsizeMatchCoeffMatrix', ['Right-hand side must be a column vector of length %u to' ...
      ' match the coefficient matrix.'], m);
end
useSingle = isUnderlyingType(A,'single');

% b must be a floating point real array
if (~isfloat(b) || ~isreal(b))
    error('jacobi:RHSInvalidClass', ['Right-hand side must be a floating-point vector with real ' ...
        'entries.']);
end

useSingle = useSingle || isUnderlyingType(b,'single');

% Assign default values to optional parameters, if omitted
if (nargin < 4) || isempty(maxit)
    maxit = min(n,100);
end
maxit = max(maxit, 0);

if ((nargin == 5) && ~isempty(x0))
    if (~isequal(size(x0), [n,1]))
       error('jacobi:WrongInitGuessSize', ['Initial guess must be a column vector of length %u to ' ...
           'match the problem size.'], n) 
    else 
        x = x0;
        useSingle = useSingle || isUnderlyingType(x,'single');
    end
else
    x = zeros(n,1);
end

if (nargin > 5)
     error('jacobi:ToomanyInputs', 'Too many input arguments.');
end

useSingle = support && useSingle;

if (useSingle)
    % Cast b and x to single
    b = single(b);
    x = single(x);
end

% Helper arrays should be dense, real, and of the same underlying type than b
prototype = real(full(zeros("like", b)));

if ((nargin < 3) ||isempty(tol))
    if (useSingle)
        tol = 1e-3;
    else
        tol = 1e-6;
    end
end
epsT = eps("like", prototype);
if (tol <= epsT)
    warning('jacobi:tooSmallTolerance', ['Tolerance may not be achievable. ' ...
        'Use a larger tolerance']);
    tol = epsT;
elseif (tol>= 1)
    warning('jacobi:tooBigTolerance', ['Tolerance is greater than 1. ' ...
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
            'tolerance %0.2g so jacobi returned it without iterating.'], gather(relres), tol);
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
            "The iterate returned (number %u) has relative residual %0.2g.", iter,tol,gather(relres));
    end
    return;
end

resvec = zeros(maxit+1,1,"like", prototype);  % preallocate vector for norm of residuals
resvec(1,:) = normr;                          % resvec(1) = norm(b-A*x0)
stag = 0;                                     % stagnation of the method
maxstagsteps = 10;

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
end                                           % for ii = 1: maxit

if (isempty(ii)) 
    ii = 0;
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
                , iter,gather(relres));
        case 1
            fprintf("jacobi stopped at iteration %u without converging to the desired tolerance " + ...
                "%0.2g because the maximum number of iterations %u was reached.\nThe iterate " + ...
                "returned has relative residual %0.2g.", iter,tol,maxit, gather(relres));
        case 2
            fprintf("jacobi stopped at iteration %u without converging to the desired tolerance " + ...
                "%0.2g because a scalar quantity became too small or too large to continue computing.\n " + ...
                "The iterate returned has relative residual %0.2g.", iter,tol,gather(relres));
        case 3
            fprintf("jacobi stopped at iteration %u without converging to the desired tolerance " + ...
                "%0.2g because the method stagnated.\n" + ...
                "The iterate returned has relative residual %0.2g.",iter,tol,gather(relres));
    end
end
end