clear;
workDir = 'G:\ZMH\Multi-scale rPPG';
addpath([workDir '\utils']);

fps_gt = 60;
fps = 30;
nVersion = 6;
nSub = 10;
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
        subID = [num2str(iSub, '%02d') '-' num2str(iVersion,'%02d')];
        real_pulse_File = [workDir '\Result\PURE_ROI@truth\' subID '\data.mat' ];%   real data
        
        PluseEst_Fold    = [NEWworkDir '\Result\Single_PhysNet_result\PURE\' subID];
        PluseEst_File = [ PluseEst_Fold '\PhysNet3D_PURE_1008_epoch03.mat' ];% predicted data
        
        if ~exist(real_pulse_File,'file')
            disp( [ subID '  real_pulse_File does not exist!' ] )
            continue;
        end
        
        if ~exist(PluseEst_File,'file')
            disp( [ subID '  PluseEst_File does not exist!' ] )
            continue;
        end
        
        load(PluseEst_File);  % PulseEst_PhysNet
        load(real_pulse_File);  % waveform
        
        real_pulse = waveform(1:2:end);
        real_pulse = double(real_pulse);% real waveform
        
        nor_PulseEst = normalizeSignal(PulseEst_PhysNet); %predicted waveform
        est_HR = instantPulseFFT(nor_PulseEst,fps,false); %   predicted HR
        
        nor_real_pulse = normalizeSignal(real_pulse);
        gtHR = instantPulseFFT(nor_real_pulse,fps,false);%  real HR
        
        %  ensure same length
        minLen = min( length(gtHR) , length(est_HR) );
        gtHR = gtHR(1:minLen);
        est_HR = est_HR(1:minLen);
        
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

