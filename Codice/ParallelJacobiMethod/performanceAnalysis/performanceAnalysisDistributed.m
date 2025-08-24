clear; clc; close all;

num_sistemi = 100;
dimensione_matrice = 1000;
densita_matrice = 0.01;  
sistemi_convergenti = 90;
max_iterazioni = 100000;
tolleranza = 1e-6;

tempi_esecuzione = zeros(num_sistemi, 1);
conteggio_iterazioni = zeros(num_sistemi, 1);
stato_convergenza = zeros(num_sistemi, 1);
residui_relativi = zeros(num_sistemi, 1);

if isempty(gcp('nocreate'))
    parpool;
end
fprintf('Test avviato su %d worker paralleli.\n\n', gcp().NumWorkers);

fprintf('Inizio risoluzione di %d sistemi sparsi (N=%d, densità=%.2f%%)...\n', ...
    num_sistemi, dimensione_matrice, densita_matrice*100);

for i = 1:num_sistemi
    fprintf(' -> Sistema %d/%d... ', i, num_sistemi);

    if i <= sistemi_convergenti
        A = sprand(dimensione_matrice, dimensione_matrice, densita_matrice);
        
        somma_off_diag = sum(abs(A), 2) - abs(diag(A));
        nuova_diagonale = somma_off_diag + rand(dimensione_matrice, 1);
        
        A = A - spdiags(diag(A), 0, dimensione_matrice, dimensione_matrice) + ...
            spdiags(nuova_diagonale, 0, dimensione_matrice, dimensione_matrice);
    else
        A = sprand(dimensione_matrice, dimensione_matrice, densita_matrice);
    end
    
    b = rand(dimensione_matrice, 1);
    A_dist = distributed(A);
    b_dist = distributed(b);

    tic;
    try
        [~, flag, relres, iter, ~] = jacobi(A_dist, b_dist, tolleranza, max_iterazioni);
        tempi_esecuzione(i) = toc;
        conteggio_iterazioni(i) = iter;
        stato_convergenza(i) = flag;
        residui_relativi(i) = relres;
        
        switch flag
            case 0
                fprintf('CONVERGENTE in %d iterazioni. Tempo: %.4f s. Residuo relativo: %.2g\n', iter, tempi_esecuzione(i), residui_relativi(i));
            case {1, 2, 3}
                fprintf('NON convergente (flag=%d). Tempo: %.4f s.\n', flag, tempi_esecuzione(i));
        end

    catch ME
        tempi_esecuzione(i) = toc;
        fprintf('ERRORE durante l''esecuzione: %s. Tempo: %.4f s.\n', ME.message, tempi_esecuzione(i));
        stato_convergenza(i) = -1;
    end
end

indici_convergenti = find(stato_convergenza == 0);
tempi_convergenti = tempi_esecuzione(indici_convergenti);
iterazioni_convergenti = conteggio_iterazioni(indici_convergenti);

fprintf('\n\n--- STATISTICHE SUI SISTEMI SPARSI CONVERGENTI ---\n');
if ~isempty(indici_convergenti)
    num_convergenti = length(indici_convergenti);
    fprintf('Sistemi convergenti: %d su %d\n', num_convergenti, num_sistemi);
    fprintf('Tempo di esecuzione medio:   %.4f s\n', mean(tempi_convergenti));
    fprintf('Tempo di esecuzione mediano:  %.4f s\n', median(tempi_convergenti));
    fprintf('Tempo minimo di esecuzione:   %.4f s\n', min(tempi_convergenti));
    fprintf('Tempo massimo di esecuzione:  %.4f s\n', max(tempi_convergenti));
else
    fprintf('Nessun sistema ha raggiunto la convergenza (flag = 0).\n');
end
fprintf('-----------------------------------------------------\n');

figure('Name', 'Analisi Tempi di Esecuzione (Matrici Sparse)', 'NumberTitle', 'off', 'WindowState', 'maximized');
b_h = bar(tempi_esecuzione, 'FaceColor', 'flat');
hold on;
colori = zeros(num_sistemi, 3);
colori(stato_convergenza == 0, :) = repmat([0.30, 0.75, 0.93], sum(stato_convergenza == 0), 1);
colori(stato_convergenza > 0, :) = repmat([0.9, 0.25, 0.25], sum(stato_convergenza > 0), 1);
b_h.CData = colori;
h_conv = patch(NaN, NaN, [0.30, 0.75, 0.93]);
h_non_conv = patch(NaN, NaN, [0.9, 0.25, 0.25]);
legend([h_conv, h_non_conv], 'Convergente (flag=0)', 'Non convergente', 'Location', 'northwest');
title('Tempo di Esecuzione per ogni Sistema Sparso');
xlabel('Indice del Sistema Casuale');
ylabel('Tempo di Esecuzione (secondi)');
grid on; box on; hold off;

if ~isempty(indici_convergenti)
    figure('Name', 'Correlazione Iterazioni-Tempo (Matrici Sparse)', 'NumberTitle', 'off', 'WindowState', 'maximized');
    scatter(iterazioni_convergenti, tempi_convergenti, 72, 'filled', 'MarkerFaceColor', [0, 0.45, 0.74], 'MarkerFaceAlpha', 0.7);
    hold on;
    p = polyfit(iterazioni_convergenti, tempi_convergenti, 1);
    yfit = polyval(p, iterazioni_convergenti);
    plot(iterazioni_convergenti, yfit, 'r--', 'LineWidth', 2);
    title('Correlazione Iterazioni vs. Tempo (Sistemi Convergenti)');
    xlabel('Numero di Iterazioni');
    ylabel('Tempo di Esecuzione (secondi)');
    legend('Dati', 'Linea di tendenza', 'Location', 'northwest');
    grid on; box on; hold off;
end