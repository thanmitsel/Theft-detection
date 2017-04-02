function [d]=convertDays3D(hh)
% if 3D matrix exists
% Converts 3D matrices to consumer x Days 2D matrix
% OR
% Converts 2D hh values to consumer x Days 2D matrix
if size(hh,3)>1
    conv=sum(hh,2);
    days(:,:)=conv(:,:,1:end);
    d=days';
else
d=zeros((size(hh,1)),365);
for i=1:365
    d(:,i)=sum(hh(:,(48*i-47):(48*i)),2);
end
end
end
