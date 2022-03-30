function HR = getHR(sig, fps)

if size(sig,1) == 1
    sig = sig';
end
N = 60*fps;
m2 = 2^nextpow2(N);
amp = fft(sig,m2,1);
pows = abs(amp).^2;
freqs = fps*(0:(m2/2))/m2;

pows(freqs < 0.8) = 0;
pows(freqs > 5) = 0;
[~, idx] = max(pows(1:length(freqs)));
HR = freqs(idx);
HR = ceil(HR*60);

% figure(22); clf; plot(sig);
% figure(33); clf; plot(freqs,pows(1:length(freqs)));
end