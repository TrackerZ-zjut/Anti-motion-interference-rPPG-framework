clear;
workDir = 'G:\HWR\Multi-scale rPPG';
NEWworkDir = 'G:\HWR\ZMH';
addpath([NEWworkDir '\utils']);

fps_Real = 30;
fps_Est = 30;
nSubject = 49;

load('UBFC_train_val_test_idx_list.mat'); 
nSub = length(sub_idx_test);

SNRs = zeros(nSubject+1,1); 
MAEs = zeros(nSubject+1,1);
RMSEs = zeros(nSubject+1,1);

sum_SNR = 0;
sum_MAE = 0;
sum_RMSE = 0;
n_SNR = 0;
n_MAE = 0;
n_RMSE = 0;

for iSub = 1 :nSub

    subName = sub_idx_test{iSub};

    vidDir = ['E:\Han\UBFC_DATASET\DATASET_2\' subName ];

    PulseEst_File = [ vidDir '\PhysNet3D_UBFC_0721.mat' ]; % PulseEst_PhysNet
    real_pulse_File = [ vidDir '\ground_truth.txt' ];
    ground_truth = dlmread( real_pulse_File );
    gt_HR = ground_truth( 1, : );  

    load(PulseEst_File); % 

    nor_ppgClipped = normalizeSignal(gt_HR');
    instPulse_Real = instantPulseFFT(nor_ppgClipped,fps_Real,false);
    gtHR = instPulse_Real;   

    nor_PulseEst = normalizeSignal( PulseEst_PhysNet );  %@@@@@@@@@@@@@@@@@@@@@@@
    instPulse_Est = instantPulseFFT(nor_PulseEst,fps_Est,false);  % 获取预测心率

    minLen = min( length(gtHR) , length(instPulse_Est) );
    gtHR = gtHR(1:minLen);
    instPulse_Est = instPulse_Est(1:minLen);

%         figure(2); clf; plot(pulseEst); xlim([1 length(pulseEst)]);

    SNRs(iSub,1) = eval_SNR(mean(gtHR), nor_PulseEst, fps_Est);  %获取SNR信噪比，保存在SNRs矩阵中
    MAEs(iSub,1) = sum(abs(gtHR - instPulse_Est))/length(gtHR);  % 获取MAR，保存MAEs矩阵中
    RMSEs(iSub,1) = sqrt(mean((gtHR - instPulse_Est).^2));  %获取RMSE，保存在RMSEs矩阵中

    sum_SNR = sum_SNR + SNRs(iSub,1);
    sum_MAE = sum_MAE + MAEs(iSub,1);
    sum_RMSE = sum_RMSE + RMSEs(iSub,1);
    n_SNR = n_SNR + 1;
    n_MAE = n_MAE + 1;
    n_RMSE = n_RMSE + 1;
end
%         figure(1); clf; plot(gtHR); xlim([1 length(instPulse_Est)]);   %@@@@@@@@@@@@@@@@@@@@@
%         figure(2); clf; plot(instPulse_Est); xlim([1 length(instPulse_Est)]);  %@@@@@@@@@@@@@@@@@@
%         figure(3); clf; plot(nor_PulseEst); xlim([1 length(nor_PulseEst)]);
%         figure(4); clf; plot(nor_ppgClipped); xlim([1 length(nor_ppgClipped)]);

SNRs(iSub+1,1) = sum_SNR/n_SNR;
MAEs(iSub+1,1) = sum_MAE/n_MAE;
RMSEs(iSub+1,1) = sum_RMSE/n_RMSE;
