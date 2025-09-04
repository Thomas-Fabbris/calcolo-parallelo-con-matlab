classdef testJacobi < matlab.unittest.TestCase
   %testJacobi - Unit case testing for jacobi function
   
   properties
      A
      b
      xTrue
      ADiv
      bDiv
   end
   
   methods(TestClassSetup)
      function setup(testCase)
         n = 100;
         testCase.A = rand(n, n) * 10;
         sommaDiag = sum(abs(testCase.A), 2) - abs(diag(testCase.A));
         nuovaDiagonale = sommaDiag + rand(n, 1) * 5;
         testCase.A = testCase.A - diag(diag(testCase.A)) + diag(nuovaDiagonale);
         testCase.xTrue = rand(n,1);
         testCase.b = testCase.A * testCase.xTrue;
         
         testCase.ADiv = rand(n,n) * 10;
         testCase.bDiv = testCase.ADiv * testCase.xTrue;
      end
   end
   
   methods(Test)
      function testNotEnoughInputs(testCase)
         testCase.verifyError(@() jacobi(testCase.A), 'jacobi:NotEnoughInputs');
      end
      
      function testNonSquareMatrix(testCase)
         AWrong = [1, 2, 3; 4, 5, 6];
         bWrong = [1; 2];
         testCase.verifyError(@() jacobi(AWrong,bWrong), 'jacobi:NonSquareMatrix');
      end
      
      function testWrongRHSSize(testCase)
         bWrong = [1; 2];
         testCase.verifyError(@() jacobi(testCase.A,bWrong), 'jacobi:RHSsizeMatchCoeffMatrix');
      end
      
      function testWrongInitialGuessSize(testCase)
         x0Wrong = [1; 2];
         testCase.verifyError(@() jacobi(testCase.A, testCase.b, [], [], x0Wrong), 'jacobi:WrongInitGuessSize');
      end
      
      function testTooManyInputs(testCase)
         testCase.verifyError(@() jacobi(testCase.A, testCase.b, 1e-6, 100, testCase.xTrue, 'extra'), 'jacobi:TooManyInputs');
      end
      
      function testZeroRHS(testCase)
         [x, flag, relres, iter] = jacobi(testCase.A, zeros(100,1));
         testCase.verifyEqual(x, zeros(100,1));
         testCase.verifyEqual(flag, 0);
         testCase.verifyEqual(iter, 0);
         testCase.verifyEqual(relres, 0);
      end
      
      function testGoodInitialGuess(testCase)
         [x, flag, relres, iter] = jacobi(testCase.A, testCase.b, [], [], testCase.xTrue);
         testCase.verifyEqual(x, testCase.xTrue);
         testCase.verifyEqual(flag, 0);
         testCase.verifyEqual(iter, 0);
         testCase.verifyLessThan(relres, 1e-6);
      end
      
      function testZeroOnDiagonal(testCase)
         AZeroDiag = testCase.A;
         AZeroDiag(2,2) = 0;
         [~, flag, ~, iter] = jacobi(AZeroDiag, testCase.b);
         testCase.verifyEqual(flag, 2);
         testCase.verifyEqual(iter, 0);
      end
      
      
      %% 4. Test sul Percorso di Esecuzione Standard (Ciclo)
      function testConvergence(testCase)
         [x, flag, relres, iter, resvec] = jacobi(testCase.A, testCase.b, 1e-9, 10000);
         testCase.verifyEqual(flag, 0, 'Il sistema dovrebbe convergere (flag=0).');
         testCase.verifyGreaterThan(iter, 0, 'Dovrebbe impiegare almeno un''iterazione.');
         testCase.verifyLessThan(iter, 10000, 'Non dovrebbe raggiungere il massimo delle iterazioni.');
         testCase.verifyLessThan(relres, 1e-9, 'Il residuo relativo dovrebbe essere inferiore alla tolleranza.');
         testCase.verifyEqual(x, testCase.xTrue, 'AbsTol', 1e-8);
         testCase.verifyEqual(numel(resvec), iter + 1);
      end
      
      function testMaxIterationsReached(testCase)
         maxit = 5;
         [~, flag, ~, iter] = jacobi(testCase.A, testCase.b, [], maxit);
         testCase.verifyEqual(flag, 1, 'Dovrebbe raggiungere il numero massimo di iterazioni (flag=1).');
         testCase.verifyEqual(iter, maxit, 'Il contatore di iterazioni dovrebbe essere uguale a maxit.');
      end
      
      function testToleranceTooSmallWarning(testCase)
         testCase.verifyWarning(@() jacobi(testCase.A, testCase.b, 0), 'jacobi:tooSmallTolerance');
      end
      
      function testToleranceTooBigWarning(testCase)
         testCase.verifyWarning(@() jacobi(testCase.A, testCase.b, 1.5), 'jacobi:tooBigTolerance');
      end
      
      %% 6. Test sulla Precisione Singola
      function testSinglePrecision(testCase)
         ASingle = single(testCase.A);
         bSingle = single(testCase.b);
         
         [x, flag] = jacobi(ASingle, bSingle, 10000, []);
         testCase.verifyEqual(flag, 0);
         testCase.verifyClass(x, 'single');
      end
      
      function testNoOutputConvergence(testCase)
         testCase.verifyWarningFree(@() jacobi(testCase.A, testCase.b, 1e-4));
      end
      
      function testNoOutputMaxIterations(testCase)
         
         testCase.verifyWarningFree(@() jacobi(testCase.A, testCase.b, 1e-12, 5));
      end
      
      function testNoOutputDivergence(testCase)
         % Esegue per coprire il messaggio di fallimento (es. zero sulla diagonale)
         A_zero_diag = testCase.A;
         A_zero_diag(1,1) = 0;
         testCase.verifyWarningFree(@() jacobi(A_zero_diag, testCase.b));
      end
      
      function testNoOutputZeroRHS(testCase)
         testCase.verifyWarningFree(@() jacobi(testCase.A, zeros(100,1)));
      end
      
      function testNoOutputGoodInitialGuess(testCase)
         testCase.verifyWarningFree(@() jacobi(testCase.A, testCase.b, 1e-6, 100, testCase.xTrue));
      end
      
   end
end