function [maxi,maxT,mini,minT, suma, av,stdev,lfactor,mintoav,mintomax, night,skew,kurt,varia, features]=extractFeatures(data)
% All features are size(data,1)x1
[maxi,maxT]=max(data,[],2);
[mini,minT]=min(data,[],2);
suma=sum(data,2);
av=mean(data,2);
stdev=std(data,0,2);
lfactor=av./maxi;
mintoav=mini./av;
mintomax=mini./maxi;
night=sum(data(:,1:5,:),2)./suma*100; % This makes sense for day vs hour matrix ONLY
skew=skewness(data,1,2);
kurt=kurtosis(data,1,2);
varia=var(data,0,2);
features=[maxi, maxT, mini, minT, suma, av, stdev, lfactor, mintoav, mintomax, night, skew, kurt, varia];
end
