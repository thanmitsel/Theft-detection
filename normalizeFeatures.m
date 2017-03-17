function [normalized, minval, maxval]=normalizeFeatures(X)
% can work on 3D Matrix
ena=ones(size(X));
minarray=ones(size(X));
maxarray=ones(size(X));
minval=min(X);
maxval=max(X);
for j=1:size(X,2)
    minarray(:,j,:)=ena(:,j,:)*minval(j);
    maxarray(:,j,:)=ena(:,j,:)*maxval(j);
end
normalized=(X-minarray)./(maxarray-minarray);
end