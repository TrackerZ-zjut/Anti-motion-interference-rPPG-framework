function snr = get_SNR(p, fps) 
pulseEst = p - mean(p);
X = nextpow2(length(pulseEst));  
n = 2^X;  
Y_pulse = fft(pulseEst,n);
powsPulse = abs(Y_pulse);  
freqPulse = (0:n/2)*fps/n;  

% figure(22); clf; plot(freqPulse,powsPulse(1:length(freqPulse)));

freqRange = (freqPulse >= 0.8) & (freqPulse <= 3.5); % signal range 0.8 - 3.5 Hz

% powsPulse2 = powsPulse(1:length(freqPulse));
% figure(22); hold on; plot(freqPulse(freqRange),powsPulse2(freqRange)); 

freqRangeComp = ( (freqPulse > 0) & (freqPulse < 0.8) )   |   ( (freqPulse > 3.5) & (freqPulse < 15) ); % noise range 0-0.8Hz , 3.5-15Hz

snr = 10*log10(sum(powsPulse(freqRange).^2)/sum(powsPulse(freqRangeComp).^2)); % get snr
end