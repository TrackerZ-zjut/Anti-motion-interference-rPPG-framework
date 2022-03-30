function snr = eval_SNR(refHR,pulseEst, fps)
pulseEst = pulseEst - mean(pulseEst);
    
mFreq = refHR/60; %
harmFreq = 2*mFreq;  
harmFreq2 = 3*mFreq;
halfWinLength = 0.3; % Hz

p = nextpow2(length(pulseEst));
n = 2^p;
Y_pulse = fft(pulseEst,n);
powsPulse = abs(Y_pulse);
freqPulse = (0:n/2)*fps/n;
% figure(22); clf; plot(freqPulse,powsPulse(1:length(freqPulse)));

freqRange = ((freqPulse <= mFreq+halfWinLength) & (freqPulse >= mFreq-halfWinLength))...
    | ((freqPulse <= harmFreq+halfWinLength) & (freqPulse >= harmFreq-halfWinLength))...
    | ((freqPulse <= harmFreq2+halfWinLength) & (freqPulse >= harmFreq2-halfWinLength));
% powsPulse2 = powsPulse(1:length(freqPulse));
% figure(22); hold on; plot(freqPulse(freqRange),powsPulse2(freqRange)); 

freqRangeComp = (freqPulse <= 5) & (freqPulse >= 0.7) & (~freqRange);

snr = 10*log10(sum(powsPulse(freqRange).^2)/sum(powsPulse(freqRangeComp).^2)); 
end