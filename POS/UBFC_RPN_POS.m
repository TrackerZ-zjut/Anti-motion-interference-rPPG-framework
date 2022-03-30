clear;
workDir = 'G:\ZMH\Multi-scale rPPG';
addpath([workDir '\utils']);

nSubject = 49;
PUREfps = 30;
winLength = 150;
stepSize = winLength/2;
hannW = hann(winLength);
k = 0.9;
offset = 1;

for iSubject = 1:nSubject
    
    subName = [ 'subject' num2str(iSubject) ];
    disp(['processing ' subName ]);
    vidDir = [workDir '\Result\UBFC_DATASET\DATASET_2\' subName ];
    roisaveFile = [ vidDir '\roi_facedetector.mat' ];%  ROI coordinates tracked by KLT
    filePath = [vidDir '\vid.avi'];
    ResultDir = [workDir '\Result\UBFC_DATASET\DATASET_2\' subName ];
    file2Save = [ResultDir '\newMulRPN_0919_POS_S1.mat'];
    
    if ~exist(filePath,'file')
        disp( [  subName ' does not exist' ] )
        continue;
    end
    
    if ~exist(ResultDir,'dir')
        mkdir(ResultDir);
    end
    
    load(roisaveFile)
    currentVideo = VideoReader(filePath);  %  read video
    nImages = currentVideo.NumberOfFrames; %  get all frames
    Num_k = floor( nImages/stepSize );
    nImages = Num_k * stepSize;
    vidHeight = currentVideo.Height; %get video height
    vidWidth = currentVideo.Width; %Get video width
    
    ROI25 = zeros(25,4,nImages );
    tracesRPN25 = zeros(25,3,nImages);
    
    ROI9 = zeros(9,4,nImages);
    tracesROI9 = zeros( 9 , 3 , nImages );
    
    ROI5 = zeros(5,4,nImages);
    tracesROI5 = zeros( 5 , 3 , nImages );
    
    for iImage =1:nImages
        currImage = read(currentVideo, iImage);  % read video frame
        
        bbox0 = rect_klt(iImage,:);%original ROI coordinates
        x0 = bbox0(1);
        y0 = bbox0(2);
        w0 = bbox0(3);
        h0 = bbox0(4);
        
        % get 25 ROI coordinates
        w70 = ceil(0.70*w0);
        w75 = ceil(0.75*w0);
        w80 = ceil(0.80*w0);
        w85 = ceil(0.85*w0);
        w90 = ceil(0.90*w0);
        
        h70_11 = ceil(w70 * 1.1);
        h70_12 = ceil(w70 * 1.2);
        h70_13 = ceil(w70 * 1.3);
        h70_14 = ceil(w70 * 1.4);
        h70_15 = ceil(w70 * 1.5);
        h75_11 = ceil(w75 * 1.1);
        h75_12 = ceil(w75 * 1.2);
        h75_13 = ceil(w75 * 1.3);
        h75_14 = ceil(w75 * 1.4);
        h75_15 = ceil(w75 * 1.5);
        h80_11 = ceil(w80 * 1.1);
        h80_12 = ceil(w80 * 1.2);
        h80_13 = ceil(w80 * 1.3);
        h80_14 = ceil(w80 * 1.4);
        h80_15 = ceil(w80 * 1.5);
        h85_11 = ceil(w85 * 1.1);
        h85_12 = ceil(w85 * 1.2);
        h85_13 = ceil(w85 * 1.3);
        h85_14 = ceil(w85 * 1.4);
        h85_15 = ceil(w85 * 1.5);
        h90_11 = ceil(w90 * 1.1);
        h90_12 = ceil(w90 * 1.2);
        h90_13 = ceil(w90 * 1.3);
        h90_14 = ceil(w90 * 1.4);
        h90_15 = ceil(w90 * 1.5);
        
        x70 = ceil(x0+w0/2-w70/2);
        x75 = ceil(x0+w0/2-w75/2);
        x80 = ceil(x0+w0/2-w80/2);
        x85 = ceil(x0+w0/2-w85/2);
        x90 = ceil(x0+w0/2-w90/2);
        
        y70_11 = ceil(y0+h0/2-h70_11/2);
        y70_12 = ceil(y0+h0/2-h70_12/2);
        y70_13 = ceil(y0+h0/2-h70_13/2);
        y70_14 = ceil(y0+h0/2-h70_14/2);
        y70_15 = ceil(y0+h0/2-h70_15/2);
        y75_11 = ceil(y0+h0/2-h75_11/2);
        y75_12 = ceil(y0+h0/2-h75_12/2);
        y75_13 = ceil(y0+h0/2-h75_13/2);
        y75_14 = ceil(y0+h0/2-h75_14/2);
        y75_15 = ceil(y0+h0/2-h75_15/2);
        y80_11 = ceil(y0+h0/2-h80_11/2);
        y80_12 = ceil(y0+h0/2-h80_12/2);
        y80_13 = ceil(y0+h0/2-h80_13/2);
        y80_14 = ceil(y0+h0/2-h80_14/2);
        y80_15 = ceil(y0+h0/2-h80_15/2);
        y85_11 = ceil(y0+h0/2-h85_11/2);
        y85_12 = ceil(y0+h0/2-h85_12/2);
        y85_13 = ceil(y0+h0/2-h85_13/2);
        y85_14 = ceil(y0+h0/2-h85_14/2);
        y85_15 = ceil(y0+h0/2-h85_15/2);
        y90_11 = ceil(y0+h0/2-h90_11/2);
        y90_12 = ceil(y0+h0/2-h90_12/2);
        y90_13 = ceil(y0+h0/2-h90_13/2);
        y90_14 = ceil(y0+h0/2-h90_14/2);
        y90_15 = ceil(y0+h0/2-h90_15/2);
        
        % Boundary restrictions
        if y70_11 +  h70_11 >= vidHeight
            h70_11= vidHeight - y70_11;
        end
        
        if y70_12 +  h70_12 >= vidHeight
            h70_12= vidHeight - y70_12;
        end
        
        if y70_13 +  h70_13 >= vidHeight
            h70_13= vidHeight - y70_13;
        end
        
        if y70_14 +  h70_14 >= vidHeight
            h70_14= vidHeight - y70_14;
        end
        
        if y70_15 +  h70_15 >= vidHeight
            h70_15= vidHeight - y70_15;
        end
        
        if y75_11 +  h75_11 >= vidHeight
            h75_11= vidHeight - y75_11;
        end
        
        if y75_12 +  h75_12 >= vidHeight
            h75_12= vidHeight - y75_12;
        end
        
        if y70_13 +  h70_13 >= vidHeight
            h70_13= vidHeight - y70_13;
        end
        
        if y75_14 +  h75_14 >= vidHeight
            h75_14= vidHeight - y75_14;
        end
        
        if y75_15 +  h75_15 >= vidHeight
            h75_15= vidHeight - y75_15;
        end
        
        if y80_11 +  h80_11 >= vidHeight
            h80_11= vidHeight - y80_11;
        end
        
        if y80_12 +  h80_12 >= vidHeight
            h80_12= vidHeight - y80_12;
        end
        
        if y80_13 +  h80_13 >= vidHeight
            h80_13= vidHeight - y80_13;
        end
        
        if y80_14 +  h80_14 >= vidHeight
            h80_14= vidHeight - y80_14;
        end
        
        if y80_15 +  h80_15 >= vidHeight
            h80_15= vidHeight - y80_15;
        end
        
        if y85_11 +  h85_11 >= vidHeight
            h85_11= vidHeight - y85_11;
        end
        
        if y85_12 +  h85_12 >= vidHeight
            h85_12= vidHeight - y85_12;
        end
        
        if y85_13 +  h85_13 >= vidHeight
            h85_13= vidHeight - y85_13;
        end
        
        if y85_14 +  h85_14 >= vidHeight
            h85_14= vidHeight - y85_14;
        end
        
        if y85_15 +  h85_15 >= vidHeight
            h85_15= vidHeight - y85_15;
        end
        
        if y90_11 +  h90_11 >= vidHeight
            h90_11= vidHeight - y90_11;
        end
        
        if y90_12 +  h90_12 >= vidHeight
            h90_12= vidHeight - y90_12;
        end
        
        if y90_13 +  h90_13 >= vidHeight
            h90_13= vidHeight - y90_13;
        end
        
        if y90_14 +  h90_14 >= vidHeight
            h90_14= vidHeight - y90_14;
        end
        
        if y90_15 +  h90_15 >= vidHeight
            h90_15= vidHeight - y90_15;
        end
        % first 25 ROI coordinates
        ROI25(1, : , iImage) = [x70, y70_11, w70, h70_11] ;
        ROI25(2, : , iImage) = [x70, y70_12, w70, h70_12] ;
        ROI25(3, : , iImage) = [x70, y70_13, w70, h70_13] ;
        ROI25(4, : , iImage) = [x70, y70_14, w70, h70_14] ;
        ROI25(5,: , iImage) = [x70, y70_15, w70, h70_15] ;
        ROI25(6,: , iImage) = [x75, y75_11, w75, h75_11] ;
        ROI25(7,: , iImage) = [x75, y75_12, w75, h75_12] ;
        ROI25(8,: , iImage) = [x75, y75_13, w75, h75_13] ;
        ROI25(9,: , iImage) = [x75, y75_14, w75, h75_14] ;
        ROI25(10,: , iImage) = [x75, y75_15, w75, h75_15] ;
        ROI25(11,: , iImage) = [x80, y80_11, w80, h80_11] ;
        ROI25(12,: , iImage) = [x80, y80_12, w80, h80_12] ;
        ROI25(13,: , iImage) = [x80, y80_13, w80, h80_13] ;
        ROI25(14,: , iImage) = [x80, y80_14, w80, h80_14] ;
        ROI25(15,: , iImage) = [x80, y80_15, w80, h80_15] ;
        ROI25(16,: , iImage) = [x85, y85_11, w85, h85_11] ;
        ROI25(17,: , iImage) = [x85, y85_12, w85, h85_12] ;
        ROI25(18,: , iImage) = [x85, y85_13, w85, h85_13] ;
        ROI25(19,: , iImage) = [x85, y85_14, w85, h85_14] ;
        ROI25(20,: , iImage) = [x85, y85_15, w85, h85_15] ;
        ROI25(21,: , iImage) = [x90, y90_11, w90, h90_11] ;
        ROI25(22,: , iImage) = [x90, y90_12, w90, h90_12] ;
        ROI25(23,: , iImage) = [x90, y90_13, w90, h90_13] ;
        ROI25(24,: , iImage) = [x90, y90_14, w90, h90_14] ;
        ROI25(25,: , iImage) = [x90, y90_15, w90, h90_15] ;
        
        imgcrop1 = imcrop ( currImage, ROI25(1, : , iImage) );
        imgcrop2 = imcrop ( currImage, ROI25(2, : , iImage) );
        imgcrop3 = imcrop ( currImage, ROI25(3, : , iImage) );
        imgcrop4 = imcrop ( currImage, ROI25(4, : , iImage) );
        imgcrop5 = imcrop ( currImage, ROI25(5, : , iImage) );
        imgcrop6 = imcrop ( currImage, ROI25(6, : , iImage) );
        imgcrop7 = imcrop ( currImage, ROI25(7, : , iImage) );
        imgcrop8 = imcrop ( currImage, ROI25(8, : , iImage) );
        imgcrop9 = imcrop ( currImage, ROI25(9, : , iImage) );
        imgcrop10 = imcrop ( currImage, ROI25(10, : , iImage) );
        imgcrop11 = imcrop ( currImage, ROI25(11, : , iImage) );
        imgcrop12 = imcrop ( currImage, ROI25(12, : , iImage) );
        imgcrop13 = imcrop ( currImage, ROI25(13, : , iImage) );
        imgcrop14 = imcrop ( currImage, ROI25(14, : , iImage) );
        imgcrop15 = imcrop ( currImage, ROI25(15, : , iImage) );
        imgcrop16 = imcrop ( currImage, ROI25(16, : , iImage) );
        imgcrop17 = imcrop ( currImage, ROI25(17, : , iImage) );
        imgcrop18 = imcrop ( currImage, ROI25(18, : , iImage) );
        imgcrop19 = imcrop ( currImage, ROI25(19, : , iImage) );
        imgcrop20 = imcrop ( currImage, ROI25(20, : , iImage) );
        imgcrop21 = imcrop ( currImage, ROI25(21, : , iImage) );
        imgcrop22 = imcrop ( currImage, ROI25(22, : , iImage) );
        imgcrop23 = imcrop ( currImage, ROI25(23, : , iImage) );
        imgcrop24 = imcrop ( currImage, ROI25(24, : , iImage) );
        imgcrop25 = imcrop ( currImage, ROI25(25, : , iImage) );
        
        % get 25 RGB traces
        tracesRPN25(1,:,iImage)  =  mean(mean(imgcrop1),2);
        tracesRPN25(2,:,iImage)  =  mean(mean(imgcrop2),2);
        tracesRPN25(3,:,iImage)  =  mean(mean(imgcrop3),2);
        tracesRPN25(4,:,iImage)  =  mean(mean(imgcrop4),2);
        tracesRPN25(5,:,iImage)  =  mean(mean(imgcrop5),2);
        tracesRPN25(6,:,iImage)  =  mean(mean(imgcrop6),2);
        tracesRPN25(7,:,iImage)  =  mean(mean(imgcrop7),2);
        tracesRPN25(8,:,iImage)  =  mean(mean(imgcrop8),2);
        tracesRPN25(9,:,iImage)  =  mean(mean(imgcrop9),2);
        tracesRPN25(10,:,iImage)  =  mean(mean(imgcrop10),2);
        tracesRPN25(11,:,iImage)  =  mean(mean(imgcrop11),2);
        tracesRPN25(12,:,iImage)  =  mean(mean(imgcrop12),2);
        tracesRPN25(13,:,iImage)  =  mean(mean(imgcrop13),2);
        tracesRPN25(14,:,iImage)  =  mean(mean(imgcrop14),2);
        tracesRPN25(15,:,iImage)  =  mean(mean(imgcrop15),2);
        tracesRPN25(16,:,iImage)  =  mean(mean(imgcrop16),2);
        tracesRPN25(17,:,iImage)  =  mean(mean(imgcrop17),2);
        tracesRPN25(18,:,iImage)  =  mean(mean(imgcrop18),2);
        tracesRPN25(19,:,iImage)  =  mean(mean(imgcrop19),2);
        tracesRPN25(20,:,iImage)  =  mean(mean(imgcrop20),2);
        tracesRPN25(21,:,iImage)  =  mean(mean(imgcrop21),2);
        tracesRPN25(22,:,iImage)  =  mean(mean(imgcrop22),2);
        tracesRPN25(23,:,iImage)  =  mean(mean(imgcrop23),2);
        tracesRPN25(24,:,iImage)  =  mean(mean(imgcrop24),2);
        tracesRPN25(25,:,iImage)  =  mean(mean(imgcrop25),2);
        
        %         figure(1); clf; imshow(imgcrop1);
        %         figure(2); clf; imshow(imgcrop2);
        %         figure(3); clf; imshow(imgcrop3);
        %         figure(4); clf; imshow(imgcrop4);
        %         figure(5); clf; imshow(imgcrop5);
        %         figure(6); clf; imshow(imgcrop6);
        %         figure(7); clf; imshow(imgcrop7);
        %         figure(8); clf; imshow(imgcrop8);
        %         figure(9); clf; imshow(imgcrop9);
        %         figure(10); clf; imshow(imgcrop10);
        %         figure(11); clf; imshow(imgcrop11);
        %         figure(12); clf; imshow(imgcrop12);
        %         figure(13); clf; imshow(imgcrop13);
        %         figure(14); clf; imshow(imgcrop14);
        %         figure(15); clf; imshow(imgcrop15);
        %         figure(16); clf; imshow(imgcrop16);
        %         figure(17); clf; imshow(imgcrop17);
        %         figure(18); clf; imshow(imgcrop18);
        %         figure(19); clf; imshow(imgcrop19);
        %         figure(20); clf; imshow(imgcrop20);
        %         figure(21); clf; imshow(imgcrop21);
        %         figure(22); clf; imshow(imgcrop22);
        %         figure(23); clf; imshow(imgcrop23);
        %         figure(24); clf; imshow(imgcrop24);
        %         figure(25); clf; imshow(imgcrop25);
    end
    
    [num_ROIs,chn,traceLength] = size(tracesRPN25);
    PulseEst_Stg1 = zeros(1, traceLength);%  first select signal
    PulseEst_Stg2_W = zeros(1, traceLength);% second select signal
    PulseEst_GS = zeros(1, traceLength);  % Gaussian fusion signal
    PulseEst_Uni = zeros(1, traceLength); %Average fusion signal
    
    SNRs_25 = zeros(1,25);
    rank_25_SNRs = zeros(1,25);
    index_25_SNRs = zeros(1,25);
    
    SNRs_9 = zeros(1,9);
    rank_9_Weights = zeros(1,9);
    index_9_Weights = zeros(1,9);
    
    for n =  winLength:stepSize:traceLength
        % first signals selection
        win_pulseEst1 = zeros( 25, winLength );
        for iROI1 = 1:25
            % POS algorithm
            raw_trace1 = tracesRPN25(iROI1 , : , n-winLength+1:n);
            raw_trace1 = squeeze( raw_trace1 );
            mean_trace1 = mean(raw_trace1,2);
            ntraces1 = raw_trace1./repmat(mean_trace1,[1,size(raw_trace1,2)]);
            S1 = [0 1 -1; -2 1 1]*ntraces1;
            p1 = S1(1,:) + S1(2,:)*std(S1(1,:))/std(S1(2,:));
            p1 = p1 - mean(p1);
            p1 = p1/std(p1);
            win_pulseEst1(iROI1,:) = p1;  %  signals extracted by POS
            SNRs_25(1,iROI1) = get_SNR( win_pulseEst1( iROI1 , : ) , PUREfps );  % get approximate snr
        end
        
        [ rank_25_SNRs , index_25_SNRs ] = sort( SNRs_25, 'descend' );
        First_index_25_SNR = index_25_SNRs( 1 , 1 );
        ROI25_max = zeros( 4, nImages );
        ROI25_max( : , n-winLength+1:n ) = ROI25( First_index_25_SNR, : , n-winLength+1:n );
        pulseEst_Stg1 = win_pulseEst1(First_index_25_SNR,:); %first selected signal
        
        for iImage2 = n-winLength+1:n
            currImage2 = read(currentVideo, iImage2);
            
            x_90= ROI25_max( 1 , iImage2 );
            y_90= ROI25_max( 2 , iImage2 );
            w_90= ROI25_max( 3 , iImage2 );
            h_90= ROI25_max( 4 , iImage2 );
            
            ROI9( : , : , iImage2 ) = F_9anchors( x_90, y_90, w_90, h_90,offset );
            % get second ROI coordinates
            x_91 = ROI9( 1 , 1 , iImage2 );
            y_91 = ROI9( 1 , 2 , iImage2 );
            w_91 = ROI9( 1 , 3 , iImage2 );
            h_91 = ROI9( 1 , 4 , iImage2 );
            
            x_92 = ROI9( 2 , 1 , iImage2 );
            y_92 = ROI9( 2 , 2 , iImage2 );
            w_92 = ROI9( 2 , 3 , iImage2 );
            h_92 = ROI9( 2 , 4 , iImage2 );
            
            x_93 = ROI9( 3 , 1 , iImage2 );
            y_93 = ROI9( 3 , 2 , iImage2 );
            w_93 = ROI9( 3 , 3 , iImage2 );
            h_93 = ROI9( 3 , 4 , iImage2 );
            
            x_94 = ROI9( 4 , 1 , iImage2 );
            y_94 = ROI9( 4 , 2 , iImage2 );
            w_94 = ROI9( 4 , 3 , iImage2 );
            h_94 = ROI9( 4 , 4 , iImage2 );
            
            x_95 = ROI9( 5 , 1 , iImage2 );
            y_95 = ROI9( 5 , 2 , iImage2 );
            w_95 = ROI9( 5 , 3 , iImage2 );
            h_95 = ROI9( 5 , 4 , iImage2 );
            
            x_96 = ROI9( 6 , 1 , iImage2 );
            y_96 = ROI9( 6 , 2 , iImage2 );
            w_96 = ROI9( 6 , 3 , iImage2 );
            h_96 = ROI9( 6 , 4 , iImage2 );
            
            x_97 = ROI9( 7 , 1 , iImage2 );
            y_97 = ROI9( 7 , 2 , iImage2 );
            w_97 = ROI9( 7 , 3 , iImage2 );
            h_97 = ROI9( 7 , 4 , iImage2 );
            
            x_98 = ROI9( 8 , 1 , iImage2 );
            y_98 = ROI9( 8 , 2 , iImage2 );
            w_98 = ROI9( 8 , 3 , iImage2 );
            h_98 = ROI9( 8 , 4 , iImage2 );
            
            x_99 = ROI9( 9 , 1 , iImage2 );
            y_99 = ROI9( 9 , 2 , iImage2 );
            w_99 = ROI9( 9 , 3 , iImage2 );
            h_99 = ROI9( 9 , 4 , iImage2 );
            
            imgcrop9ROI_1 = imcrop ( currImage2, [x_91, y_91, w_91, h_91] );
            imgcrop9ROI_2 = imcrop ( currImage2, [x_92, y_92, w_92, h_92] );
            imgcrop9ROI_3 = imcrop ( currImage2, [x_93, y_93, w_93, h_93] );
            imgcrop9ROI_4 = imcrop ( currImage2, [x_94, y_94, w_94, h_94] );
            imgcrop9ROI_5 = imcrop ( currImage2, [x_95, y_95, w_95, h_95] );
            imgcrop9ROI_6 = imcrop ( currImage2, [x_96, y_96, w_96, h_96] );
            imgcrop9ROI_7 = imcrop ( currImage2, [x_97, y_97, w_97, h_97] );
            imgcrop9ROI_8 = imcrop ( currImage2, [x_98, y_98, w_98, h_98] );
            imgcrop9ROI_9 = imcrop ( currImage2, [x_99, y_99, w_99, h_99] );
            
            % get 9 RGB traces
            tracesROI9(1,:,iImage2)  =  mean(mean(imgcrop9ROI_1),2);
            tracesROI9(2,:,iImage2)  =  mean(mean(imgcrop9ROI_2),2);
            tracesROI9(3,:,iImage2)  =  mean(mean(imgcrop9ROI_3),2);
            tracesROI9(4,:,iImage2)  =  mean(mean(imgcrop9ROI_4),2);
            tracesROI9(5,:,iImage2)  =  mean(mean(imgcrop9ROI_5),2);
            tracesROI9(6,:,iImage2)  =  mean(mean(imgcrop9ROI_6),2);
            tracesROI9(7,:,iImage2)  =  mean(mean(imgcrop9ROI_7),2);
            tracesROI9(8,:,iImage2)  =  mean(mean(imgcrop9ROI_8),2);
            tracesROI9(9,:,iImage2)  =  mean(mean(imgcrop9ROI_9),2);
        end
        
        % second signals selection
        win_pulseEst2 = zeros( 9, winLength );
        for iROI2 = 1:9
            % POS algorithm
            raw_trace2 = tracesROI9( iROI2 , : , n-winLength+1:n);
            raw_trace2 = squeeze( raw_trace2 );
            mean_trace2 = mean(raw_trace2,2);
            ntraces2 = raw_trace2./repmat(mean_trace2,[1,size(raw_trace2,2)]);
            S2 = [0 1 -1; -2 1 1]*ntraces2;
            p2 = S2(1,:) + S2(2,:)*std(S2(1,:))/std(S2(2,:));
            p2 = p2 - mean(p2);
            p2 = p2/std(p2);
            win_pulseEst2(iROI2, : ) = p2 ; %  signals extracted by POS
        end
        SNRs_9 = getVarWeight(win_pulseEst2,winLength);
        [ rank_9_Weights, index_9_Weights ] = sort( SNRs_9, 'descend' );
        First_index_9_Weights = index_9_Weights( 1 , 1 );
        ROI9_Weight_max = zeros( 4 , nImages );
        ROI9_Weight_max( : , n-winLength+1:n ) = ROI9(First_index_9_Weights,  :  , n-winLength+1:n );
        pulseEst_Stg2_W = win_pulseEst2(First_index_9_Weights,:); %second selection signal
        
        for iImage3 = n-winLength+1:n
            currImage3 = read(currentVideo, iImage3);
            
            ImageW = size(currImage3, 2 );
            ImageH = size(currImage3,1 );
            
            x_9weight = ROI9_Weight_max(1 , iImage3);
            y_9weight = ROI9_Weight_max(2 , iImage3);
            w_9weight = ROI9_Weight_max(3 , iImage3);
            h_9weight = ROI9_Weight_max(4 , iImage3);
            
            ROI_Weight_mutiscale  = F_multisacle( x_9weight,y_9weight,w_9weight,h_9weight, k, ImageW, ImageH);
            
            imgcrop5ROI_1 = imcrop ( currImage3, ROI_Weight_mutiscale( 1 , : ) );
            imgcrop5ROI_2 = imcrop ( currImage3, ROI_Weight_mutiscale( 2 , : ) );
            imgcrop5ROI_3 = imcrop ( currImage3, ROI_Weight_mutiscale( 3 , : ) );
            imgcrop5ROI_4 = imcrop ( currImage3, ROI_Weight_mutiscale( 4 , : ) );
            imgcrop5ROI_5 = imcrop ( currImage3, ROI_Weight_mutiscale( 5 , : ) );
            % get 5 RGB traces
            tracesROI5(1,:,iImage3)  =  mean(mean(imgcrop5ROI_1),2);
            tracesROI5(2,:,iImage3)  =  mean(mean(imgcrop5ROI_2),2);
            tracesROI5(3,:,iImage3)  =  mean(mean(imgcrop5ROI_3),2);
            tracesROI5(4,:,iImage3)  =  mean(mean(imgcrop5ROI_4),2);
            tracesROI5(5,:,iImage3)  =  mean(mean(imgcrop5ROI_5),2);
        end
        
        for iROI3 = 1:5
            % POS algorithm
            raw_trace3 = tracesROI5( iROI3 , : , n-winLength+1:n );
            raw_trace3 = squeeze( raw_trace3 );
            mean_trace3 = mean(raw_trace3,2);
            ntraces3 = raw_trace3./repmat( mean_trace3, [1,size(raw_trace3,2)] );
            S3 = [0 1 -1; -2 1 1]*ntraces3;
            p3 = S3(1,:) + S3(2,:)*std(S3(1,:))/std(S3(2,:));
            p3 = p3 - mean(p3);
            p3 = p3/std(p3);
            win_pulseEst3( iROI3, : ) = p3; %  signals extracted by POS
        end
        
        %  Gaussian fusion signal
        win_fusion_pulseEst_GS = 0.05*win_pulseEst3(1,:) + 0.25*win_pulseEst3(2,:)+ 0.4*win_pulseEst3(3,:)...
            + 0.25*win_pulseEst3(4,:)+ 0.05*win_pulseEst3(5,:);
        %   Average fusion signal
        win_fusion_pulseEst_Uni = 0.2*win_pulseEst3(1,:) + 0.2*win_pulseEst3(2,:)+ 0.2*win_pulseEst3(3,:)...
            + 0.2*win_pulseEst3(4,:)+ 0.2*win_pulseEst3(5,:);
        
        win_fusion_pulseEst_Uni = win_fusion_pulseEst_Uni.*(hannW)';
        win_fusion_pulseEst_GS = win_fusion_pulseEst_GS.*(hannW)';
        pulseEst_Stg1 = pulseEst_Stg1.*(hannW)';
        pulseEst_Stg2_W = pulseEst_Stg2_W.*(hannW)';
        
        % Overlap and add to complete signals
        PulseEst_Stg2_W(1, n-winLength+1:n) = PulseEst_Stg2_W(1, n-winLength+1:n) + pulseEst_Stg2_W;
        PulseEst_Stg1(1, n-winLength+1:n) = PulseEst_Stg1(1, n-winLength+1:n) + pulseEst_Stg1;
        PulseEst_GS(1, n-winLength+1:n) = PulseEst_GS(1,n-winLength+1:n) + win_fusion_pulseEst_GS;
        PulseEst_Uni(1, n-winLength+1:n) = PulseEst_Uni(1,n-winLength+1:n) + win_fusion_pulseEst_Uni;
        
    end
    
    save( file2Save, 'PulseEst_GS','PulseEst_Uni','PulseEst_Stg1','PulseEst_Stg2_W');
    disp([ subID ' PluseEst complete' ]);
    
end


