function [ ROI_mutiscale ] = F_multisacle( x0,y0,w0,h0, k, ImageW, ImageH)

        k_2 = k^(-2);  
        k_1 = k^(-1);
        k1  = k^1;
        k2  = k^2;

        w_2 = w0*k_2;       % Level -2 
        h_2  = h0*k_2;
        x_2 = x0 + w0/2 - w_2/2 ;
        y_2 = y0 + h0/2 - h_2/2 ;
        if x_2 < 1; x_2 = 1; end
        if y_2 < 1; y_2 = 1; end
        if x_2+w_2 > ImageW; w_2 = ImageW-x_2; end
        if y_2+h_2 > ImageH; h_2 = ImageH-y_2; end
        bboxL_2 = [ floor(x_2),  floor(y_2), floor(w_2), floor(h_2) ];

        w_1 = w0*k_1;       % Level  -1 
        h_1  = h0*k_1;
        x_1 = x0 + w0/2 - w_1/2 ;
        y_1 = y0 + h0/2 - h_1/2 ;
        if x_1 < 1; x_1 = 1; end
        if y_1 < 1; y_1 = 1; end
        if x_1+w_1 > ImageW;  w_1 = ImageW-x_1; end
        if y_1+h_1 > ImageH;  h_1 = ImageH-y_1; end
        bboxL_1 = [  floor(x_1),  floor(y_1),  floor(w_1),  floor(h_1)];

        bboxL0 = [ floor(x0), floor(y0), floor(w0), floor(h0)];  % Level 0

        w1 = w0*k1;       % Level  1  
        h1  = h0*k1;
        x1 = x0 + w0/2 - w1/2 ; 
        y1 = y0 + h0/2 - h1/2 ;
        bboxL1 = [ floor(x1), floor(y1), floor(w1), floor(h1) ];

        w2 = w0*k2;       % Level  2 
        h2  = h0*k2;
        x2 = x0 + w0/2 - w2/2;
        y2 = y0 + h0/2 - h2/2;
        bboxL2 = [ floor(x2), floor(y2), floor(w2), floor(h2)];

        ROI_mutiscale = [ bboxL_2; bboxL_1; bboxL0; bboxL1; bboxL2 ];

end
