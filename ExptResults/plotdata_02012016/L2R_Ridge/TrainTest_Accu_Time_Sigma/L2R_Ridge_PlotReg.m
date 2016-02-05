% load data from all methods
Randbin_test = importdata('RandBin_OneVsAll_cadata_TestPerf.plotdata', ' ', 1);
Randbin_train = importdata('RandBin_OneVsAll_cadata_TrainPerf.plotdata', ' ', 1);
Standard = importdata('STANDARD_OneVsAll_Laplace_cadata_Perf.plotdata', ' ', 1);

%Fig 1-1: L2R_Ridge, Train and Test Error VS Sigma, fixed R
set(0,'defaultaxesfontsize',15);
h1 = figure(1);
ind_train = find(Randbin_train.data(:,1) == 128);
ind_test = find(Randbin_test.data(:,1) == 128);
semilogx(Randbin_train.data(ind_train,3),Randbin_train.data(ind_train,5),'bo','LineWidth',2,'MarkerSize',10)
hold on
semilogx(Randbin_test.data(ind_test,3),Randbin_test.data(ind_test,5),'rx','LineWidth',2,'MarkerSize',10)
semilogx(Randbin_train.data(ind_train,3),ones(1,length(Randbin_train.data(ind_train,3)))...
    *Standard.data(4), '-k.','LineWidth',2,'MarkerSize',10)
hold off
xlabel('\sigma','FontSize',24, 'FontWeight','bold')
ylabel('Error %')
title('cadata: Train and Test Error VS \sigma')
legend('Train data', 'Test data', 'Peak Accuracy', 'Location','NorthWest');
saveas(h1,'cadata_L2R_Ridge_Error_Sigma_128','fig');
saveas(h1,'cadata_L2R_Ridge_Error_Sigma_128','pdf');

%Fig 1-1: L2R_Ridge, Train and Test Time VS Sigma, fixed R
set(0,'defaultaxesfontsize',15);
h2 = figure(2);
minperf = min(Randbin_test.data(ind_test,5));
minperf_ind = find(Randbin_test.data(:,5) == minperf);
loglog(Randbin_train.data(ind_train,3),Randbin_train.data(ind_train,6),'-bo','LineWidth',2,'MarkerSize',10)
hold on
loglog(Randbin_test.data(ind_test,3),Randbin_test.data(ind_test,6),'-rx','LineWidth',2,'MarkerSize',10)
plot(Randbin_train.data(minperf_ind,3), Randbin_test.data(minperf_ind,6),'ks','LineWidth',2,'MarkerSize',18)
hold off
xlabel('\sigma','FontSize',24, 'FontWeight','bold')
ylabel('Time (Seconds)')
title('cadata: Train Time VS \sigma')
legend('Train data', 'Test data', 'Peak Accuracy', 'Location','NorthWest');
saveas(h2,'cadata_L2R_Ridge_Time_Sigma_128','fig');
saveas(h2,'cadata_L2R_Ridge_Time_Sigma_128','pdf');

%Fig 1-7: L2R_Ridge, Train Accuracy VS R, fixed Sigma
set(0,'defaultaxesfontsize',15);
h7 = figure(7);
Sigma = Randbin_train.data(:,3);
ind_train_sigma1 = find(Randbin_train.data(:,3) == Sigma(1));
ind_train_sigma2 = find(Randbin_train.data(:,3) == Sigma(2));
ind_train_sigma3 = find(Randbin_train.data(:,3) == Sigma(3));
semilogx(Randbin_train.data(ind_train_sigma1,1),Randbin_train.data(ind_train_sigma1,5),'-bo','LineWidth',2,'MarkerSize',10)
hold on
semilogx(Randbin_train.data(ind_train_sigma2,1),Randbin_train.data(ind_train_sigma2,5),'-rx','LineWidth',2,'MarkerSize',10)
semilogx(Randbin_train.data(ind_train_sigma3,1),Randbin_train.data(ind_train_sigma3,5),'-c+','LineWidth',2,'MarkerSize',10)
hold off
xlabel('R')
ylabel('Accuracy %')
title('acoustic: Train Accuracy VS R')
legend(['\sigma:' num2str(Sigma(1))], ['\sigma:' num2str(Sigma(2))], ...
    ['\sigma:' num2str(Sigma(3))], 'Location','NorthEast');
saveas(h7,'acoustic_L2R_Ridge_TrAccu_R','fig');
saveas(h7,'acoustic_L2R_Ridge_TrAccu_R','pdf');
