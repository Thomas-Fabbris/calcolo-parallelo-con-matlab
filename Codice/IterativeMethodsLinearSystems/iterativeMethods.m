clc
clear
% Metodi iterativi - metodo del gradiente coniugato (CG)

% Generazione matrice di Wathen
n = 400;
N = 3*n^2+4*n+1;
A = distributed(gallery('wathen', n, n));
fprintf("\nDimensione matrice dei coefficienti A: %u \n\n", N);

% Generazione vettore-colonna dei termini noti
b = sum(A,2);
% Soluzione esatta del sistema
xExact = ones(N,1,'distributed');

% I tentativo - Risoluzione del sistema con CG default
[xCG_1,flagCG_1,relres_CG1,iterCG_1,resvecCG_1] = pcg(A,b);
% Calcolo dell'errore assoluto
errCG_1 = abs(xExact - xCG_1);
% Rappresentazione grafica errore assoluto
figure(1)
semilogy(errCG_1, 'o');
title('System of Linear Equations with Sparse Matrix');
ylabel('Absolute Error');
xlabel('Element in x');

if flagCG_1 == true
    disp("[CG_1]: La soluzione approssimata non converge");
end

% II tentativo - metodo del gradiente coniugato (CG) personalizzato

tolerance = 1e-12;
maxit = N;      

tCG = tic;
[xCG_2,flagCG_2,relresCG_2,iterCG_2,resvecCG_2] = pcg(A,b,tolerance,maxit);
tCG = toc(tCG);

if flagCG_2 == false
    disp("[CG_2]: La soluzione approssimata converge");
end

errCG_2 = abs(xExact - xCG_2);

figure(2)
semilogy(errCG_1,'o');
hold on
semilogy(errCG_2,'diamond');
title('Comparison of Absolute Error');
ylabel('Absolute Error');
xlabel('Element in x');
legend('Default tolerance and iterations','Improved tolerance and iterations');
hold off

relresvecCG = resvecCG_2./resvecCG_2(1);

figure(3)
f=semilogy(relresvecCG);
hold on
semilogy(f.Parent.XLim,[1e-6 1e-6],'--')
semilogy([20 20], f.Parent.YLim,'--')
semilogy(f.Parent.XLim,[1e-12 1e-12],'--')
px = 20;
py = relresvecCG(px);

plot(px, py, 's')
title('Evolution of Relative Residual');
ylabel('Relative Residual');
xlabel('Iteration Step');
legend('Residuals of CG','Default Tolerance','Default Number of Steps','Custom Tolerance')
hold off

% III tentativo - metodo del gradiente coniugato precondizionato (PCG)
% Matrice di precondizionamento M, scelta come la diagonale principale 
% di A, in quanto A possiede per costruzione pochi elementi non nulli fuori dalla diagonale
M = spdiags(spdiags(A,0),0,N,N);

tPCG = tic;
[xPCG,flagPCG,relresPCG,iterPCG,resvecPCG] = pcg(A,b,tolerance,maxit,M);
tPCG = toc(tPCG);

figure(4)
hold off;
semilogy(relresvecCG)
hold on;
semilogy(resvecPCG./resvecPCG(1))
semilogy(f.Parent.XLim,[1e-12 1e-12],'--')
title('Evolution of Relative Residual');
ylabel('Relative Residual');
xlabel('Iteration Step');
legend('Residuals of CG','Residuals of PCG with M \approx diag(A)', 'Custom tolerance')

% PCG converge ad una soluzione per il sistema in un numero molto minore di
% iterazioni rispetto a CG, con conseguenze sul tempo di risoluzione

fprintf([...
    '\n[CG_2 =  CG]:  Tempo di computazione:  %d s', ...
    '\n[PCG]: Tempo di computazione: %d s\n'],tCG,tPCG);

fprintf([...
    '\n[CG]:  Numero di iterazioni richieste per la convergenza  %u s', ...
    '\n[PCG]: Numero di iterazioni richieste per la convergenza: %u s\n'],iterCG_2,iterPCG);

% PCG trova una soluzione più accurata rispetto a CG

errPCG = abs(xExact-xPCG);
figure(5)
hold off
semilogy(errCG_1,'o');
hold on
semilogy(errCG_2,'d');
semilogy(errPCG,'x');
title('Comparison of absolute error');
ylabel('Absolute error');
xlabel('Element in x');
legend('CG default','CG custom','PCG');

delete(gcp('nocreate'))