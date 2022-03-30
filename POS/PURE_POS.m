clear;
workDir = 'G:\ZMH\Multi-scale rPPG';
addpath([workDir '\utils']);

nSub = 10;
nVersion = 6;
PUREfps = 30;
winLength = 150;
stepSize = winLength/2;
hannW = hann(winLength);


for  iSub = 1:nSub
    for iVersion = 1:nVersion
        subID = [num2str(iSub,'%02d') '-' num2str(iVersion,'%02d')];
        disp(['processing ' subID ]);
        vidDir = [ 'E:\PURE\Data\' subID];
        ResultDir = [workDir '\Result\PURE\' subID ];
        roi_File = [ ResultDir '\roi_facedetector.mat' ];  %  ROI coordinates tracked by KLT
        NewResultDir = [NEWworkDir '\Result\PURE\' subID ];
        file2Save = [NewResultDir '\new_single_POS_1220.mat'];
        
        if ~exist(vidDir,'dir')
            disp([ subID 'does not exist'])
            continue;
        end
        
        imageList = dir(vidDir);
        load(roi_File);  % rect_klt
        nImages = length(imageList)-2;
        Num_k = floor( nImages/stepSize );
        nImages = Num_k * stepSize;
        
        traces = zeros( 3, nImages );
        for iImage =1:nImages
            imageName = imageList(iImage+2).name;
            imagePath = [vidDir '\' imageName];
            currImage = imread(imagePath);
            
            bbox0 = rect_klt(iImage,:);  % ROI coordinates
            imgcrop = imcrop ( currImage, bbox0 );
            traces(:,iImage)  =  mean(mean(imgcrop),2);   % get  RGB traces
        end
        
        traceLength = size(traces,2);
        win_pulseEst = zeros( 1, winLength );
        PulseEst = zeros(1, traceLength);
        
        for n = winLength:stepSize:traceLength
            % POS algorithm
            raw_trace = traces( : , n-winLength+1:n);
            mean_trace = mean(raw_trace,2);
            ntraces = raw_trace./repmat(mean_trace,[1,size(raw_trace,2)]);
            S = [0 1 -1; -2 1 1]*ntraces;
            p = S(1,:) + S(2,:)*std(S(1,:))/std(S(2,:));
            p = p - mean(p);
            p = p/std(p);
            win_pulseEst = p; %  windows signal extracted by POS
            win_fusion_pulseEst = win_pulseEst.*(hannW)';
            % Overlap and add to complete signal
            PulseEst(n-winLength+1:n) = PulseEst(n-winLength+1:n) + win_fusion_pulseEst;
        end
        save( file2Save, 'PulseEst' );
    end
end

disp('PluseEst complete');

