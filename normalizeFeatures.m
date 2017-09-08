function [normalized, minval, maxval]=normalizeFeatures(X)
% can work on 3D Matrix
minval=min(X);
maxval=max(X);

if sum(maxval==0)~=0 % if maxval=0 use 1 to keep the 0 on normalized
    columns=maxval==0;
    maxval(columns)=1;
end

ena=ones(size(X));
minarray=ones(size(X));
maxarray=ones(size(X));


for j=1:size(X,2)
    minarray(:,j,:)=ena(:,j,:)*minval(j);
    maxarray(:,j,:)=ena(:,j,:)*maxval(j);
end
normalized=(X-minarray)./(maxarray-minarray);
end