

nSub = 35;
nVersion =  3;
PUREfps = 30;
winLength = 150;
stepSize = winLength/2;
hannW = hann(winLength);
skinM = diag([0.7682, 0.5121, 0.3841]);


for iVersion = 1 :nVersion
    for iSub = 1 :nSub
        subID = [num2str(iVersion,'%02d') '-' num2str(iSub,'%02d')];
        disp(['processing ' subID ]);
        
        vidDir = [workDir '\Result\self rPPG\' subID];
        filePath = [ vidDir '\video.avi' ];
        roi_File = [ vidDir '\roi_facedetector.mat' ];  %  ROI coordinates tracked by KLT
        ResultDir = [workDir '\Result\self rPPG\' subID ];
        file2Save = [ResultDir '\new_single_CHROM_1220.mat'];
        
        
        if ~exist(vidDir,'dir')
            disp([ subID 'does not exist'])
            continue;
        end
        
        if ~exist(ResultDir,'dir')
            mkdir(ResultDir);
        end
        
        load(roi_File)
        currentVideo = VideoReader(filePath);  %  read video
        nImages = currentVideo.NumberOfFrames;  %  get all frames
        nImages = nImages - 200;
        Num_k = floor( nImages/stepSize );
        nImages = Num_k * stepSize;
        
        traces = zeros( 3, nImages );
        
        for iImage =1:nImages
            currImage = read(currentVideo, iImage);  % read video frame
            bbox0 = rect_klt(iImage,:);
            imgcrop = imcrop ( currImage, bbox0 );
            traces(:,iImage)  =  mean(mean(imgcrop),2);%get RGB trace
            
        end
        traceLength = size(traces,2);
        win_pulseEst = zeros( 1, winLength );
        PulseEst = zeros(1, traceLength);
        
        for n = winLength:stepSize:traceLength
            % CHROM algorithm
            raw_trace = traces( : , n-winLength+1:n);
            mean_trace = mean(raw_trace,2);
            ntraces = raw_trace./repmat(mean_trace,[1,size(raw_trace,2)]);
            ntraces = skinM*ntraces;
            S = [3 -2 0; 1.5 1 -1.5]*ntraces;
            p = S(1,:) - S(2,:)*std(S(1,:))/std(S(2,:));
            p = p - mean(p);
            p = p/std(p);
            win_pulseEst = p;  %windows signal extracted by CHROM
            win_fusion_pulseEst = win_pulseEst.*(hannW)';
            % Overlap and add to complete signal
            PulseEst(n-winLength+1:n) = PulseEst(n-winLength+1:n) + win_fusion_pulseEst;
        end
        save( file2Save, 'PulseEst' );
    end
end

disp('PluseEst complete');

