function [hh, h, d, w, m, DxHH, DxH, WxH, MxH]=ConsumptionsConvert(data)
% DxHH matrix, days vs halh-hours size=365x48
% DxH matrix, days vs hours size=365x24
% ignore last day or 24 hours to create
% WxH matrix week vs hours size=52x158
% ignore last 5 days or 120 last hours to create
% MxH matrix month vs hours size=12x720 
% hh vector, size=48*365=17520 with kWh per half hour
% h vector, size=24*365=8760 with kWh per hour
% d vector, size=365 with kWh per day
% w vector, size=52 with kWh per week
% m vector, size=12 with kWh per month
h=zeros(8760,1);

hh=data(:,3);
for i=1:8760
    h(i)=hh(2*i-1)+hh(2*i);
end
[DxHH]=vec2mat(hh,48);
[DxH]=vec2mat(h,24);
[WxH]=vec2mat(h(1:8636),168);
[MxH]=vec2mat(h(1:8640),720);
d=sum(DxH,2);
w=sum(WxH,2);
m=sum(MxH,2);
end