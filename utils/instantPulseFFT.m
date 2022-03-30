% instant pulse
function instPulse = instantPulseFFT(pulseEst,fps,useECG)

pulseEst = pulseEst - mean(pulseEst);
winLength = 10*fps; % in seconds*fps
stepSize = 1*fps; 
traceLength = length(pulseEst);
tStart = winLength:stepSize:traceLength;
instPulse = zeros(length(tStart),1);
tx = (0:winLength-1)/fps;
for i = 1:length(tStart)
    t = tStart(i);
    y = pulseEst(t-winLength+1:t);
    if ~useECG
        instPulse(i) = getHR(y, fps);
    else
        instPulse(i) = getHR_ECG(y, fps);
    end
end

end