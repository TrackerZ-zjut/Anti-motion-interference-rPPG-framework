function sigOut = normalizeSignal(sigIn)
% normalize signals to zero mean and unit variance

[traceLength, chn] = size(sigIn);
mSig = mean(sigIn);
sigOut = sigIn - mSig;
sigOut = sigOut/std(sigOut);
end