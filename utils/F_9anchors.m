function [ Anchors9ROI ] = F_9anchors(x,y,w,h,offset)
       
        Anchors9ROI = zeros(9,4);

        Anchors9ROI(1,:) = [x,        y,        w, h ];
        Anchors9ROI(2,:) = [x,        y-offset, w, h ];
        Anchors9ROI(3,:) = [x+offset, y,        w, h ];
        Anchors9ROI(4,:) = [x,        y+offset, w, h ];
        Anchors9ROI(5,:) = [x-offset, y,        w, h ];
        Anchors9ROI(6,:) = [x+offset, y-offset, w, h ];
        Anchors9ROI(7,:) = [x+offset, y+offset, w, h ];
        Anchors9ROI(8,:) = [x-offset, y+offset, w, h ];
        Anchors9ROI(9,:) = [x-offset, y-offset, w, h ];

end
