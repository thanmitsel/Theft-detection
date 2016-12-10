function [m, M]=convertMonths(data)
h=zeros(8760,1);

hh=data(:,3);
for i=1:8760
    h(i)=hh(2*i-1)+hh(2*i);
end
[M]=vec2mat(h(1:8640),720);
m=sum(M,2);
end