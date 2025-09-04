function [varargout] = jacobi(varargin)
%JACOBI - Jacobi Method for distributed array.
%
%   Syntax
%      x = JACOBI(A,b)
%      x = JACOBI(A,b,tol)
%      x = JACOBI(A,b,tol,maxit)
%      x = JACOBI(A,b,tol,maxit,x0)
%      [x,flag] = JACOBI(___)
%      [x,flag,relres] = JACOBI(___)
%      [x,flag,relres,iter] = JACOBI(___)
%      [x,flag,relres,iter,resvec] = JACOBI(___)
%
%   Example
%   Using JACOBI to solve a system of distributed array data:
%      A = gallery("poisson",50));
%      b = full(sum(A,2));
%      dA = distributed(A);
%      db = distributed(b);
%      tol = 1e-9;
%      maxit = 500;
%      dX = jacobi(dA,db,tol,maxit);
%   
%   See also DISTRIBUTED, JACOBI.

%   Author: Thomas Fabbris   
%   Version: 1.0

varargout = cell(1, max(nargout,1));
[varargout{:}] = distributedutil.distributedSpmdWrapper(@jacobi, ...
                                                        varargin{:});
end