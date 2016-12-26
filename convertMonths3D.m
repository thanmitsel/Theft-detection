function [m, M]=convertMonths3D(hh)
m=zeros(size(hh,1),12);
M=zeros(12,730,size(hh,1));
h=zeros(size(hh,1),size(hh,2)/2);
% 17520/12=1460 hours on a month
for i=1:size(h,2)
    h(:,i)=sum(hh(:,(2*i-1):(2*i)),2);
end
for j=1:12
    m(:,j)=sum(hh(:,(1460*j-1459):(1460*j)),2);
end
for k=1:size(hh,1)
    M(:,:,k)=vec2mat(h(k,:),730);
end
end
