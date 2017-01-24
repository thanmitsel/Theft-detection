function [f_data, y, F_data, Y] = type3Fraud (data)
% Fraud can be interrapted on days and hours
% with different intensity per hour and day
F_data=data;
Y=zeros(size(data,1),1);
[dstart, fraudDays] = initFraudperMatrix(data);
for i=1:length(fraudDays)
  [tstart, duration, intensity] = initFraudperRow (data); %intensity not needed
  for j=tstart:(tstart+duration)
    [patates, agouria, hour_intensity] = initFraudperRow (data); %patates-agouria not needed
    F_data(fraudDays(i),j)=...
      hour_intensity*data(fraudDays(i),j);
  end
  Y(fraudDays(i),1)=1;
end
f_data_temp=F_data(:)';
f_data=f_data_temp';
y=1;
end
