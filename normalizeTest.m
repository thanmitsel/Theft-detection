function [normalizedTest]=normalizeTest(Xtest, minval, maxval)
% can work on 3D Matrix
ena=ones(size(Xtest));
minarray=ones(size(Xtest));
maxarray=ones(size(Xtest));
for j=1:size(Xtest,2)
    minarray(:,j,:)=ena(:,j,:)*minval(j);
    maxarray(:,j,:)=ena(:,j,:)*maxval(j);
end
normalizedTest=(Xtest-minarray)./(maxarray-minarray);
end