function [fraudDays] = initFraudperMatrix(data)
%Works on a 2D matrix
rate=rand(1); % Random rate
nRows=size(data,1);
nSample=floor(rate*size(data,1));
rndIDX=randperm(nRows);
newSample=rndIDX(1:nSample);
fraudDays=sort(newSample);
end
