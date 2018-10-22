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
[x,RMS] = fminsearch(@(s)MiddleburyRMS(abs(sfm.QuerryX-sfm.TrainedX)*s,sfm.GroundTruth),1)

figure
plot(x*abs(sfm.QuerryX-sfm.TrainedX)); hold on; plot(sfm.GroundTruth);hold off;
    