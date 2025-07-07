classdef(Sealed = true) PjmHelper
    %PjmHelper - Helper class for pjm
    %   PJMHELPER is the helper class for pjm, providing costant properties used in the parallel 
    %   implementation of Jacobi Method for solving systems of linear equations.
    %
    %PjmHelper Properties:
    %   DefaultMaxit - Default maximum number of iterations for pjm
    %      100 (default) | positive scalar
    %   DefaultTol - Default method tolerance for pjm
    %      1e-6 (default) | positive scalar
    %   MinOutArgs - Minimum number of output arguments for pjm
    %      0 (default) | nonnegative scalar integer 
    %   MaxOutArgs - Maximum number of output arguments for pjm
    %      5 (default) | positive scalar integer
    %
    %See also pjm
    properties(Constant)
        % DefaultMaxit - Default maximum number of iterations for pjm
        %    100 (default) | positive scalar
        DefaultMaxit = 100
        %   DefaultTol - Default method tolerance for pjm
        %      1e-6 (default) | positive scalar
        DefaultTol = 1e-6
        %   MinOutArgs - Minimum number of output arguments for pjm
        %      0 (default) | nonnegative scalar integer 
        MinOutArgs = 0
        %   MaxOutArgs - Maximum number of output arguments for pjm
        %      5 (default) | positive scalar integer
        MaxOutArgs = 5
    end

     methods (Access = private)
        function obj = PjmHelper()
            % Private constructor
        end
     end
end