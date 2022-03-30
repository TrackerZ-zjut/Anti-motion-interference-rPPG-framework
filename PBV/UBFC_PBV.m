clear;
workDir = 'G:\ZMH\Multi-scale rPPG';
addpath([workDir '\utils']);


nSubject = 49;
PUREfps = 30;
winLength = 150;
stepSize = winLength/2;
hannW = hann(winLength);
load('uspeusig.mat');
u_sig = u(:,2);

for iSubject = 1:nSubject
    
    subName = [ 'subject' num2str(iSubject) ];
    disp(['processing ' subName ]);
    vidDir = [workDir '\Result\UBFC_DATASET\DATASET_2\' subName ];
    roisaveFile = [ vidDir '\roi_facedetector.mat' ];%  ROI coordinates tracked by KLT
    filePath = [vidDir '\vid.avi'];
    ResultDir = [workDir '\Result\UBFC_DATASET\DATASET_2\' subName ];
    file2Save = [ResultDir '\new_single_PBV_1220.mat'];
    
    if ~exist(filePath,'file')
        disp( [  subName ' does not exist' ] )
        continue;
    end
    
    load(roisaveFile);
    currentVideo = VideoReader(filePath); %  read video
    nImages = currentVideo.NumberOfFrames; %  get all frames
    Num_k = floor( nImages/stepSize );
    nImages = Num_k * stepSize;
    
    traces = zeros(3,nImages);
    for iImage =1:nImages
        currImage = read(currentVideo, iImage);  % read video frame
        bbox0 = rect_klt(iImage,:);     % ROI coordinates
        imgcrop = imcrop ( currImage, bbox0 ); % get  RGB traces
    end
    
    traceLength = size(traces,2);
    win_pulseEst = zeros( 1, winLength );
    PulseEst = zeros(1, traceLength);
    
    for n = winLength:stepSize:traceLength
        % PBV algorithm
        raw_trace = traces( : , n-winLength+1:n);
        mean_trace = mean(raw_trace,2);
        ntraces = raw_trace./repmat(mean_trace,[1,size(raw_trace,2)]);
        ntraces = ntraces - ones(3,winLength);
        p = u_sig'*((ntraces*ntraces')\ntraces);
        p = p - mean(p);
        p = p/std(p);
        win_pulseEst = p;%  windows signal extracted by PBV
        win_fusion_pulseEst = win_pulseEst.*(hannW)';
        % Overlap and add to complete signal
        PulseEst(n-winLength+1:n) = PulseEst(n-winLength+1:n) + win_fusion_pulseEst;
    end
    save( file2Save, 'PulseEst');
end
disp(' PluseEst complete');
