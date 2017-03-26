function [f_data,y, F_data,Y] = type2_1Fraud (data, intensity)
% Fraud can be interrapted on days and hours
% with different intensity on days
F_data=data;
Y=zeros(size(data,1),1);
[fraudDays] = initFraudperMatrix(data);
for i=1:length(fraudDays)
  [tstart, duration] = initFraudperRow (data);
  F_data(fraudDays(i),tstart:(tstart+duration))=...
    intensity*data(fraudDays(i),tstart:(tstart+duration));
  Y(fraudDays(i),1)=1;
end
f_data_temp=F_data(:)';
f_data=f_data_temp';
y=1;
end
