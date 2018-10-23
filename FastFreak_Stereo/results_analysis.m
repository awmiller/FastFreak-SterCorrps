% import
ffm = importdata('FastFreakMatches.csv');
sfm = importdata('SiftMatches.csv');

% structify
for i=1:length(ffm.textdata)
    goodname = regexprep(ffm.textdata{i},'\W','');
    ffm.(goodname) = ffm.data(:,i);
end

for i=1:length(sfm.textdata)
    goodname = regexprep(sfm.textdata{i},'\W','');
    sfm.(goodname) = sfm.data(:,i);
end

% heres a reconstructed sparse disperity
surf_n_terp(sfm.QuerryX,sfm.QuerryY,sfm.Distance);

% i want to know the distribution of dy values so i can make a good
% threshold guess
dy= abs(sfm.QuerryY - sfm.TrainedY);
histogram(dy,50,'Normalization','probability'); % this looks good

% now change bins to simulate changing threshold, observe bin mass
he = zeros(1,49); hv = zeros(1,49);
for i=2:50
    % figure
    h = histogram(dy,i,'Normalization','probability');
    he(i-1) = h.BinEdges(2);
    hv(i-1) = h.Values(1);
end

% figure
plot(he,hv);
title('Match Probablility vs. Max Vertical Distance Threshold');
ylabel('Normalized Probablility');
xlabel('Vertical Distance (pixels)');

% do some analysis
% plot(sfm.Distance); hold on; plot(sfm.GroundTruth);
% truthAmp = max(sfm.GroundTruth) - min(sfm.GroundTruth);
% matchAmp = max(sfm.Distance) - min(sfm.Distance);
% scale = truthAmp/matchAmp;
% plot(scale*sfm.Distance); hold on; plot(sfm.GroundTruth);

% heres raw compare
figure
plot(sfm.GroundTruth); hold on; plot(abs(sfm.QuerryX-sfm.TrainedX))

% calculate dx from sift
sdx = abs(sfm.QuerryX - sfm.TrainedX);
plot(sdx); hold on; plot(sfm.GroundTruth);

figure
plot(abs(sfm.QuerryX-sfm.TrainedX)); hold on; plot(sfm.GroundTruth);hold off;
title('Raw SIFT Disparity Per Match Index')
figure
plot(abs(ffm.QuerryX-ffm.TrainedX)); hold on; plot(ffm.GroundTruth);hold off;
title('Raw FAST-FREAK Disparity Per Match Index')

% SIFT_RMS = MiddleburyRMS(abs(sfm.QuerryX-sfm.TrainedX),sfm.GroundTruth)

[sift_fit,sift_RMS] = fminsearch(@(s)MiddleburyRMS(abs(sfm.QuerryX-sfm.TrainedX)*s,sfm.GroundTruth),1)

[ffm_fit,ffm_RMS] = fminsearch(@(s)MiddleburyRMS(abs(ffm.QuerryX-ffm.TrainedX)*s,ffm.GroundTruth),1)

figure
plot(abs(sfm.QuerryX-sfm.TrainedX)*sift_fit); hold on; plot(sfm.GroundTruth);hold off;
title('Scaled SIFT Disparity Per Match Index')
figure
plot(abs(ffm.QuerryX-ffm.TrainedX)*ffm_fit); hold on; plot(ffm.GroundTruth);hold off;
title('Scaled FAST-FREAK Disparity Per Match Index')

sift_BADP = MiddleburyBadPixels(abs(sfm.QuerryX-sfm.TrainedX)*sift_fit,sfm.GroundTruth,40)
ffm_BADP = MiddleburyBadPixels(abs(ffm.QuerryX-ffm.TrainedX)*ffm_fit,ffm.GroundTruth,40)

d = zeros(1,100);
for i=100:-1:1
    
    d(i) = MiddleburyBadPixels(abs(sfm.QuerryX-sfm.TrainedX)*sift_fit,sfm.GroundTruth,i)
end

plot(d);
title('Percent Bad Pixels vs. Disparity Error Threshold');