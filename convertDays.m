function [d]= convertDays(data)
%data needs to range (0,8760*2)
hh=data(:,3);
[DxHH]=vec2mat(hh,48);
d =sum(DxHH,2);
end