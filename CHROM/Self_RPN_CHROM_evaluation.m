clear;
workDir = 'G:\ZMH\Multi-scale rPPG';
addpath([workDir '\utils']);

nVersion = 3;
nSub = 35;
fps_Real = 60;
fps_Est = 30;

SNRs = zeros(nSub+1,nVersion);
MAEs = zeros(nSub+1,nVersion);
RMSEs = zeros(nSub+1,nVersion);

for iVersion = 1 :nVersion
    
    sum_SNR = 0;
    sum_MAE = 0;
    sum_RMSE = 0;
    n_SNR = 0;
    n_MAE = 0;
    n_RMSE = 0;
    
    for iSub = 1 :nSub
        subID = [num2str(iVersion,'%02d') '-' num2str(iSub, '%02d')];
        vidDir = [workDir '\Result\self rPPG\' subID];
        ResultDir = [workDir '\Result\self rPPG\' subID ];
        PluseEst_File = [ ResultDir '\newMulRPN_0919_CHROM_S7.mat' ];  %  predicted data
        real_pulse_File = [ vidDir '\ppgClipped.mat' ];  % real data
        
        if ~exist(real_pulse_File,'file')
            disp( [ subID ' does not exist' ] )
            continue;
        end
        
        if ~exist(PluseEst_File,'file')
            disp( [ subID 'does not exist' ] )
            continue;
        end
        
        load(PluseEst_File);  %  load predicted waveform
        load(real_pulse_File);  %  load real waveform
        
        nor_ppgClipped = normalizeSignal(ppgClipped); %  real waveform
        gtHR = instantPulseFFT(nor_ppgClipped,fps_Real,false);  %  real HR
        
        %         pulseEst = PulseEst_GS;
        pulseEst = PulseEst_Uni;
        nor_PulseEst = normalizeSignal(pulseEst);  %   predicted waveform
        est_HR = instantPulseFFT(nor_PulseEst,fps_Est,false);  %   predicted HR
        
        % ensure same length
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
