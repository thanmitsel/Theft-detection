function [normalizedTest]=normalizeTest(Xtest, minval, maxval)
% can work on 3D Matrix
if sum(maxval==0)~=0 % if maxval=0 use 1 to keep the 0 on normalized
    columns=maxval==0;
    maxval(columns)=1;
end
ena=ones(size(Xtest));
minarray=ones(size(Xtest));
maxarray=ones(size(Xtest));
for j=1:size(Xtest,2)
    minarray(:,j,:)=ena(:,j,:)*minval(j);
    maxarray(:,j,:)=ena(:,j,:)*maxval(j);
end
normalizedTest=(Xtest-minarray)./(maxarray-minarray);
end