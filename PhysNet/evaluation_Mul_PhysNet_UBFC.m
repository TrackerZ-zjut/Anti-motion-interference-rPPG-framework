clear;
workDir = 'G:\ZMH\Multi-scale rPPG';
addpath([workDir '\utils']);

fps_Real = 30;
fps_Est = 30;
nSubject = 49;

SNRs = zeros(nSubject+1,1);
MAEs = zeros(nSubject+1,1);
RMSEs = zeros(nSubject+1,1);

for iVersion = 1
    
    sum_SNR = 0;
    sum_MAE = 0;
    sum_RMSE = 0;
    n_SNR = 0;
    n_MAE = 0;
    n_RMSE = 0;
    
    for iSub = 1 :nSubject
        subName = [ 'subject' num2str(iSub) ];
        vidDir = [workDir '\Result\UBFC_DATASET\DATASET_2\' subName ];
        filePath = [vidDir '\vid.avi'];
        ResultDir = [workDir '\Result\UBFC_DATASET\DATASET_2\' subName  ];
        PulseEst_File = [ ResultDir '\newMulRPN_UBFC_1012_PhysNet3D.mat' ];% predicted data
        
        if ~exist(filePath, 'file')
            disp( [  subName ' does not exist' ] )
            continue;
        end
        
        real_pulse_File = [ vidDir '\ground_truth.txt' ];
        ground_truth = dlmread( real_pulse_File );
        gt_HR = ground_truth( 1, : );   %  real waveform
        
        if ~exist(PulseEst_File, 'file')
            disp( [  subName ' does not exist' ] )
            continue;
        end
        
        load(PulseEst_File);
        
        nor_ppgClipped = normalizeSignal(gt_HR');
        instPulse_Real = instantPulseFFT(nor_ppgClipped,fps_Real,false);
        gtHR = instPulse_Real;   %  real HR
        
        %         pulseEst = PulseEst_GS;
        pulseEst = PulseEst_Uni;
        nor_PulseEst = normalizeSignal(pulseEst);
        instPulse_Est = instantPulseFFT(nor_PulseEst,fps_Est,false);  %   predicted HR
        
        %  ensure same length
        minLen = min( length(gtHR) , length(instPulse_Est) );
        gtHR = gtHR(1:minLen);
        instPulse_Est = instPulse_Est(1:minLen);
        
        SNRs(iSub,iVersion) = eval_SNR(mean(gtHR),nor_PulseEst,fps);  %get SNR
        MAEs(iSub,iVersion) = sum(abs(gtHR - est_HR))/length(gtHR);  % get MAE
        RMSEs(iSub,iVersion) = sqrt(mean((gtHR - est_HR).^2));  %get RMSE
        
        sum_SNR = sum_SNR + SNRs(iSub,iVersion);
        sum_MAE = sum_MAE + MAEs(iSub,iVersion);
        sum_RMSE = sum_RMSE + RMSEs(iSub,iVersion);
        n_SNR = n_SNR + 1;
        n_MAE = n_MAE + 1;
        n_RMSE = n_RMSE + 1;
    end
    
    SNRs(iSub+1,iVersion) = sum_SNR/n_SNR; %Average SNRs
    MAEs(iSub+1,iVersion) = sum_MAE/n_MAE; %Average MAEs
    RMSEs(iSub+1,iVersion) = sum_RMSE/n_RMSE; %Average RMSEs
end
