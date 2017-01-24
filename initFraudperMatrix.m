function [dstart, fraudDays] = initFraudperMatrix(data)
%Works on a 2D matrix

rate=rand(1);
nRows=size(data,1);
nSample=floor(rate*size(data,1));
rndIDX=randperm(nRows);
newSample=rndIDX(1:nSample);
fraudDays=sort(newSample);

dstart=fraudDays(1);

end
