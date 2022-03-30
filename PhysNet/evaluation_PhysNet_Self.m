clear;
workDir = 'G:\ZMH\Multi-scale rPPG';
addpath([workDir '\utils'])

fps_Real = 60;
fps_Est = 30;
nVersion = 3;
nSub = 35;
SNRs = zeros(nSub+1,nVersion);
MAEs = zeros(nSub+1,nVersion);
RMSEs = zeros(nSub+1,nVersion);


for iVersion = 1:nVersion
    
    sum_SNR = 0;
    sum_MAE = 0;
    sum_RMSE = 0;
    n_SNR = 0;
    n_MAE = 0;
    n_RMSE = 0;
    for iSub = 1:nSub
        subID = [num2str(iVersion,'%02d') '-' num2str(iSub,'%02d')];
        PluseEst_Fold = [workDir '\Result\self rPPG\' subID ];
        PluseEst_File = [ PluseEst_Fold '\newSingle_1012_Self_PhysNet3D.mat' ];% predicted data
        real_pulse_Fold = [workDir '\Result\self rPPG\' subID];
        real_pulse_File = [ real_pulse_Fold '\ppgClipped.mat' ];%   real data
        if ~exist(destfold,'dir')
            mkdir(destfold);
        end
        
        if ~exist(real_pulse_File,'file')
            disp( [ subID '  does not exist!' ] )
            continue;
        end
        
        if ~exist(PluseEst_File,'file')
            disp( [ subID '  does not exist!' ] )
            continue;
        end
        
        load(PluseEst_File);  % PulseEst_PhysNet
        load(real_pulse_File);  %  ppgClipped
        
        nor_ppgClipped = normalizeSignal(ppgClipped);%  % real waveform
        gtHR = instantPulseFFT(nor_ppgClipped,fps_Real,false); %  real HR
        
        nor_PulseEst = normalizeSignal(PulseEst_PhysNet); %predicted waveform
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


