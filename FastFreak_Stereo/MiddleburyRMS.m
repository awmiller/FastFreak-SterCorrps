function [ RMS ] = MiddleburyRMS( dispc, dispt )
%MiddleburyRMS Calculates the RMS of the disparity errors using the Middlebury
%stereo methods at http://vision.middlebury.edu/stereo/ and in their paper
%given the input vector indicies refer to the disparity of the same 
%pixel coordinate for each index. ie:
%  disp1    disp2
% [ d11 ]  [ d21 ] ---> img (x1,y1)
% [ d12 ]  [ d22 ] ---> img (x2,y2)
% [ ... ]  [ ... ] ...
% [ d1N ]  [ d2N ] ---> img (xN,yN)
%
% note that there is no required ordering of x or y values.

err = dispc - dispt;
sqerr = err.^2;
agg = sum(sqerr) / length(dispc);
RMS = agg.^(1/2);

end

