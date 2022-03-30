function HR = getHR_ECG(sig, fps)

idx = find(sig > 70);
diff = idx(2:end) - idx(1:end-1);
locs = find(diff>10);
idxStart = locs(1);
idxEnd = locs(end)+1;

timeInterval = (idx(idxEnd) - idx(idxStart))/fps;
nInterval = length(find(diff>10));

HR = (nInterval)/timeInterval;
HR = ceil(HR*60);
end