clear;
clc;

fprintf('Inizio del test di performance per il metodo di Jacobi...\n\n');

num_sistemi = 100;
n = 1000;               
tol = 1e-6;             
maxit = 100000;           

tempi_soluzione = zeros(num_sistemi, 1);
stati_uscita = zeros(num_sistemi, 1);
iterazioni_finali = zeros(num_sistemi, 1);
residui_relativi = zeros(num_sistemi, 1);

for i = 1:num_sistemi
    if mod(i, 10) == 0
        A = rand(n, n) * 10;
        fprintf('Sistema %d/%d: Generata matrice casuale (test di non-convergenza).\n', i, num_sistemi);
    else
        A = rand(n, n) * 10;
        somma_off_diag = sum(abs(A), 2) - abs(diag(A));
        nuova_diagonale = somma_off_diag + rand(n, 1) * 5;
        A = A - diag(diag(A)) + diag(nuova_diagonale);
    end
    
    x_vera = rand(n, 1);
    b = A * x_vera;
    tic;
    [~, flag, relres, iter, ~] = jacobi(A, b, tol, maxit);
    tempi_soluzione(i) = toc;
    
    stati_uscita(i) = flag;
    iterazioni_finali(i) = iter;
    residui_relativi(i) = relres;
    
    fprintf('Sistema %d/%d risolto in %.4f secondi. Stato: %d, Iterazioni: %d, Residuo Relativo: %0.2g\n', ...
            i, num_sistemi, tempi_soluzione(i), stati_uscita(i), iterazioni_finali(i), residui_relativi(i));
end

fprintf('\n...Risoluzione di tutti i sistemi completata.\n');
fprintf('Generazione delle statistiche e dei grafici in corso...\n');

num_convergenti = sum(stati_uscita == 0);
num_max_iter = sum(stati_uscita == 1);
num_falliti_stallo = sum(stati_uscita == 3);
num_falliti_altro = sum(stati_uscita ~= 0 & stati_uscita ~= 1 & stati_uscita ~= 3);
tempo_totale = sum(tempi_soluzione);

fprintf('\n--- STATISTICHE FINALI ---\n');
fprintf('Numero totale di sistemi: %d\n', num_sistemi);
fprintf('  - Convergenti (Flag 0): %d\n', num_convergenti);
fprintf('  - Max Iterazioni (Flag 1): %d\n', num_max_iter);
fprintf('  - Fallimento Calcolo (Flag 2): %d\n', num_falliti_altro);
fprintf('  - Stagnazione (Flag 3): %d\n', num_falliti_stallo);
fprintf('Tempo totale di esecuzione: %.2f secondi\n', tempo_totale);
fprintf('Tempo medio per sistema: %.4f secondi\n', mean(tempi_soluzione));
fprintf('--------------------------\n');


figure('Name', 'Riepilogo Esecuzioni Jacobi', 'NumberTitle', 'off');
categorie_stati = categorical(stati_uscita, [0, 1, 2, 3], {'Convergenza', 'Max Iterazioni', 'Fallimento (NaN)', 'Stagnazione'});
histogram(categorie_stati, 'BarWidth', 0.5);
title('Distribuzione degli Stati di Uscita del Metodo di Jacobi');
xlabel('Stato Finale');
ylabel('Numero di Sistemi');
grid on;

figure('Name', 'Analisi Performance Jacobi', 'NumberTitle', 'off');
gscatter(iterazioni_finali, tempi_soluzione, categorie_stati, 'brgkc', 'o*xsd');
title('Performance: Tempo di Risoluzione vs. Iterazioni');
xlabel('Numero di Iterazioni');
ylabel('Tempo di Risoluzione (secondi)');
legend('Location', 'northwest');
grid on;

fprintf('\nGrafici generati con successo. Fine dello script.\n');