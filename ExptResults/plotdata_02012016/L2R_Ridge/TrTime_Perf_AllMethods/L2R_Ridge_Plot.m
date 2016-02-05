% load data from all methods
Randbin = importdata('RandBin_OneVsAll_SUSY_Perf.plotdata', ' ', 1);
Fourier = importdata('FOURIER_OneVsAll_Laplace_SUSY_Perf.plotdata', ' ', 1);
Nystrom = importdata('NYSTROM_OneVsAll_Laplace_SUSY_Perf.plotdata', ' ', 1);
Blockdiag = importdata('BLOCKDIAG_OneVsAll_Laplace_SUSY_Perf.plotdata', ' ', 1);

%Fig 1-3: L2R_Ridge, TrainTime VS Accuracy
set(0,'defaultaxesfontsize',15);
h3 = figure(3);
semilogy(Randbin.data(:,5),Randbin.data(:,6),'-rx','LineWidth',2,'MarkerSize',10)
hold on
semilogy(Fourier.data(:,5),Fourier.data(:,6),'-bo','LineWidth',2,'MarkerSize',10)
semilogy(Nystrom.data(:,5),Nystrom.data(:,6),'-c+','LineWidth',2,'MarkerSize',10)
semilogy(Blockdiag.data(:,5),Blockdiag.data(:,6),'-m*','LineWidth',2,'MarkerSize',10)
hold off
minperf = min([min(Randbin.data(:,5)),min(Fourier.data(:,5)),...
    min(Nystrom.data(:,5)),min(Blockdiag.data(:,5))]);
maxperf = max([max(Randbin.data(:,5)),max(Fourier.data(:,5)),...
    max(Nystrom.data(:,5)),max(Blockdiag.data(:,5))])+4;
% minperf =min(Randbin.data(:,5));
% maxperf =max(Randbin.data(:,5));

xlim([(minperf) (maxperf)])
xlabel('Test Accuracy %')
ylabel('Train Time (Seconds)')
title('SUSY: Train Time VS Accuracy')
legend('RandBin', 'Fourier', 'Nystrom', 'BlockDiag', 'Location','NorthWest');
saveas(h3,'SUSY_L2R_Ridge_Traintime_Accu','fig');
saveas(h3,'SUSY_L2R_Ridge_Traintime_Accu','pdf');

%Fig 1-4: L2R_Ridge, MemoryEst VS Accuracy
set(0,'defaultaxesfontsize',15);
h4 = figure(4);
semilogy(Randbin.data(:,5),Randbin.data(:,8),'-rx','LineWidth',2,'MarkerSize',10)
hold on
semilogy(Fourier.data(:,5),Fourier.data(:,8),'-bo','LineWidth',2,'MarkerSize',10)
semilogy(Nystrom.data(:,5),Nystrom.data(:,8),'-c+','LineWidth',2,'MarkerSize',10)
semilogy(Blockdiag.data(:,5),Blockdiag.data(:,8),'-m*','LineWidth',2,'MarkerSize',10)
hold off

xlim([(minperf) (maxperf)])
xlabel('Test Accuracy %')
ylabel('Normalized Memory Estimate')
title('SUSY: Memory Estimate VS Accuracy')
legend('RandBin', 'Fourier', 'Nystrom', 'BlockDiag', 'Location','NorthWest');
saveas(h4,'SUSY_L2R_Ridge_MemEst_Accu','fig');
saveas(h4,'SUSY_L2R_Ridge_MemEst_Accu','pdf');
