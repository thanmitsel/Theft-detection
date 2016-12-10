function [w, W]=convertWeeks(data)
h=zeros(8760,1);

hh=data(:,3);
for i=1:8760
    h(i)=hh(2*i-1)+hh(2*i);
end
[W]=vec2mat(h(1:8636),168);
w=sum(W,2);
end