% load data from all methods
Randbin_train = importdata('RandBin_OneVsAll_SUSY_TrainPerf.plotdata', ' ', 1);
Randbin_test = importdata('RandBin_OneVsAll_SUSY_TestPerf.plotdata', ' ', 1);
% Standard = importdata('STANDARD_OneVsAll_Laplace_SUSY_Perf.plotdata', ' ', 1);

%Fig 1-5: L2R_Ridge, Train and Test Accuracy VS Sigma, fixed R
set(0,'defaultaxesfontsize',15);
h5 = figure(5);
ind_train = find(Randbin_train.data(:,1) == 128);
ind_test = find(Randbin_test.data(:,1) == 128);
semilogx(Randbin_train.data(ind_train,3),Randbin_train.data(ind_train,5),'bo','LineWidth',2,'MarkerSize',10)
hold on
semilogx(Randbin_test.data(ind_test,3),Randbin_test.data(ind_test,5),'rx','LineWidth',2,'MarkerSize',10)
% semilogx(Randbin_train.data(ind_train,3),ones(1,length(Randbin_train.data(ind_train,3)))...
%     *Standard.data(4), '-k.','LineWidth',2,'MarkerSize',10)
hold off
xlabel('\sigma','FontSize',24, 'FontWeight','bold')
ylabel('Accuracy %')
title('SUSY: Train and Test Accuracy VS \sigma')
legend('Train data', 'Test data', 'Location','SouthWest');
% legend('Train data', 'Test data', 'Peak Accuracy', 'Location','SouthWest');
saveas(h5,'SUSY_L2R_Ridge_Accu_Sigma_128','fig');
saveas(h5,'SUSY_L2R_Ridge_Accu_Sigma_128','pdf');

%Fig 1-6: L2R_Ridge, Train and Test Time VS Sigma, fixed R
set(0,'defaultaxesfontsize',15);
h6 = figure(6);
maxperf =max(Randbin_test.data(ind_test,5));
maxperf_ind = find(Randbin_test.data(:,5) == maxperf);
loglog(Randbin_train.data(ind_train,3),Randbin_train.data(ind_train,6),'-bo','LineWidth',2,'MarkerSize',10)
hold on
loglog(Randbin_test.data(ind_test,3),Randbin_test.data(ind_test,6),'-rx','LineWidth',2,'MarkerSize',10)
plot(Randbin_train.data(maxperf_ind,3), Randbin_test.data(maxperf_ind,6),'ks','LineWidth',2,'MarkerSize',18)
hold off
xlabel('\sigma','FontSize',24, 'FontWeight','bold')
ylabel('Time (Seconds)')
title('SUSY: Train Time VS \sigma')
legend('Train data', 'Test data', 'Peak Accuracy', 'Location','SouthEast');
saveas(h6,'SUSY_L2R_Ridge_Time_Sigma_128','fig');
saveas(h6,'SUSY_L2R_Ridge_Time_Sigma_128','pdf');

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
title('SUSY: Train Accuracy VS R')
legend(['\sigma:' num2str(Sigma(1))], ['\sigma:' num2str(Sigma(2))], ...
    ['\sigma:' num2str(Sigma(3))], 'Location','SouthEast');
saveas(h7,'SUSY_L2R_Ridge_TrAccu_R','fig');
saveas(h7,'SUSY_L2R_Ridge_TrAccu_R','pdf');


