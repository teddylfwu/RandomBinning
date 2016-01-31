% load data from all methods
Randbin = importdata('RandBin_OneVsAll_a9a_Perf.plotdata', ' ', 1);
Fourier = importdata('FOURIER_OneVsAll_Laplace_a9a_Perf.plotdata', ' ', 1);
Nystrom = importdata('NYSTROM_OneVsAll_Laplace_a9a_Perf.plotdata', ' ', 1);
Blockdiag = importdata('BLOCKDIAG_OneVsAll_Laplace_a9a_Perf.plotdata', ' ', 1);

%Fig 1-1: L2R_Ridge, Accuracy VS Rank
Rank = [8 16 32 64 128 256 512 1024];
Randbin_best = zeros(length(Rank),1); % for classification
Fourier_best = zeros(length(Rank),1); % for classification
Nystrom_best = zeros(length(Rank),1); % for classification
Blockdiag_best = zeros(length(Rank),1); % for classification
for j = 1:length(Rank)
    for i = 1:size(Randbin.data,1)
        if Randbin.data(i,1) == Rank(j)
            if Randbin_best(j) < Randbin.data(i,5) % for classification
                Randbin_best(j) = Randbin.data(i,5);
            end
        end
    end
    for i = 1:size(Fourier.data,1)
        if Fourier.data(i,1) == Rank(j)
            if Fourier_best(j) < Fourier.data(i,5) % for classification
                Fourier_best(j) = Fourier.data(i,5);
            end
        end
    end
    for i = 1:size(Nystrom.data,1)
        if Nystrom.data(i,1) == Rank(j)
            if Nystrom_best(j) < Nystrom.data(i,5) % for classification
                Nystrom_best(j) = Nystrom.data(i,5);
            end
        end
    end
    for i = 1:size(Blockdiag.data,1)
        if Blockdiag.data(i,1) == Rank(j)
            if Blockdiag_best(j) < Blockdiag.data(i,5) % for classification
                Blockdiag_best(j) = Blockdiag.data(i,5);
            end
        end
    end
end
set(0,'defaultaxesfontsize',15);
h1 = figure(1);
semilogx(Rank,Randbin_best,'-rx','LineWidth',2,'MarkerSize',10)
hold on
semilogx(Rank,Fourier_best,'-b.','LineWidth',2,'MarkerSize',10)
semilogx(Rank,Nystrom_best,'-c+','LineWidth',2,'MarkerSize',10)
semilogx(Rank,Blockdiag_best,'-m*','LineWidth',2,'MarkerSize',10)
xlim([5 2000])
xlabel('Rank')
ylabel('Accuracy %')
title('a9a: Accuracy VS Rank')
legend('Rand Bin', 'Fourier', 'Nystrom', 'BlockDiag','Location','SouthEast');
saveas(h1,'a9a_L2R_Ridge_Accu_Rank','fig');
saveas(h1,'a9a_L2R_Ridge_Accu_Rank','eps');