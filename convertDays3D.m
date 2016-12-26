function [d]=convertDays3D(hh)
d=zeros((size(hh,1)),365);
for i=1:365
    d(:,i)=sum(hh(:,(48*i-47):(48*i)),2);
end
end
