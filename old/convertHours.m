function [h, H]=convertHours(data)
%data needs to range (0,8760*2)
h=zeros(8760,1);

hh=data(:,3);
for i=1:8760
    h(i)=hh(2*i-1)+hh(2*i);
end
[H]=vec2mat(h,24);
end