function [h, H]=convertHours3D(hh)
% size consumers*hours_consumption
h=zeros(size(hh,1),size(hh,2)/2);
H=zeros(365,24,size(hh,1));
for i=1:size(h,2)
    h(:,i)=hh(:,2*i-1)+hh(:,2*i);
end
for j=1:size(h,1)
    H(:,:,j)=vec2mat(h(j,:),24);
end
end

