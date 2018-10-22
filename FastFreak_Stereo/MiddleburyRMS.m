function [ RMS ] = MiddleburyRMS( dispc, dispt )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

err = dispc - dispt;
sqerr = err.^2;
agg = sum(sqerr) / length(dispc);
RMS = agg.^(1/2);

end

