function [ B ] = MiddleburyBadPixels( disp1,disp2, thresh)
%MiddleburyBadPixels Computes the number of bad pixels in two disparity
%maps, given the input vector indicies refer to the disparity of the same 
%pixel coordinate for each index. ie:
%  disp1    disp2
% [ d11 ]  [ d21 ] ---> img (x1,y1)
% [ d12 ]  [ d22 ] ---> img (x2,y2)
% [ ... ]  [ ... ] ...
% [ d1N ]  [ d2N ] ---> img (xN,yN)
%
% note that there is no required ordering of x or y values.

delta = abs(disp1-disp2);
match = delta>thresh;
total = sum(match);
B = total/length(disp1);

end

