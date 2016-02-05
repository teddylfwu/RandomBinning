% load data from all methods
Randbin = importdata('RandBin_OneVsAll_cadata_Perf.plotdata', ' ', 1);
Fourier = importdata('FOURIER_OneVsAll_Laplace_cadata_Perf.plotdata', ' ', 1);
Nystrom = importdata('NYSTROM_OneVsAll_Laplace_cadata_Perf.plotdata', ' ', 1);
Blockdiag = importdata('BLOCKDIAG_OneVsAll_Laplace_cadata_Perf.plotdata', ' ', 1);
Standard = importdata('STANDARD_OneVsAll_Laplace_cadata_Perf.plotdata', ' ', 1);

%Fig 1-1: L2R_Ridge, Accuracy VS R
set(0,'defaultaxesfontsize',15);
h1 = figure(1);
semilogx(Randbin.data(:,1),Randbin.data(:,5),'-rx','LineWidth',2,'MarkerSize',10)
hold on
semilogx(Fourier.data(:,1),Fourier.data(:,5),'-bo','LineWidth',2,'MarkerSize',10)
semilogx(Nystrom.data(:,1),Nystrom.data(:,5),'-c+','LineWidth',2,'MarkerSize',10)
semilogx(Blockdiag.data(:,1),Blockdiag.data(:,5),'-m*','LineWidth',2,'MarkerSize',10)
semilogx(Randbin.data(:,1),ones(1,length(Randbin.data(:,1)))*Standard.data(4),...
    '-k.','LineWidth',2,'MarkerSize',10)
hold off
xlim([5 2000])
xlabel('R')
ylabel('Test Error %')
title('cadata: Test Error VS R')
legend('Rand Bin', 'Fourier', 'Nystrom', 'BlockDiag','Kernel','Location','NorthEast');
saveas(h1,'cadata_L2R_Ridge_Error_R','fig');
saveas(h1,'cadata_L2R_Ridge_Error_R','pdf');

%Fig 1-2: L2R_Ridge, TrainTime VS R
set(0,'defaultaxesfontsize',15);
h2 = figure(2);
loglog(Randbin.data(:,1),Randbin.data(:,6),'-rx','LineWidth',2,'MarkerSize',10)
hold on
loglog(Fourier.data(:,1),Fourier.data(:,6),'-bo','LineWidth',2,'MarkerSize',10)
loglog(Nystrom.data(:,1),Nystrom.data(:,6),'-c+','LineWidth',2,'MarkerSize',10)
loglog(Blockdiag.data(:,1),Blockdiag.data(:,6),'-m*','LineWidth',2,'MarkerSize',10)
semilogx(Randbin.data(:,1),ones(1,length(Randbin.data(:,1)))*Standard.data(5),...
    '-k.','LineWidth',2,'MarkerSize',10)
hold off
xlim([5 2000])
xlabel('R')
ylabel('Train Time (Seconds)')
title('cadata: Train Time VS R')
legend('Rand Bin', 'Fourier', 'Nystrom', 'BlockDiag','Kernel','Location', 'SouthEast');
saveas(h2,'cadata_L2R_Ridge_Traintime_R','fig');
saveas(h2,'cadata_L2R_Ridge_Traintime_R','pdf');
