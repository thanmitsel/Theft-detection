function [factor] = type4Fraud (data)
% factor is the average of the intensity per fraud duration on day
[fraudDays] = initFraudperMatrix(data);
% dstart=fraudDays(1);
factor=zeros(length(fraudDays),1);
for i=1:length(fraudDays)
  [tstart, duration] = initFraudperRow(data); %intensity not needed
  if tstart~=24
    for j=tstart:(tstart+duration)
        hour_intensity=1-betarnd(6,3);     
        factor(i,1)=factor(i,1)+hour_intensity;    
     end
  else % when start is 24th hour
    duration=1; 
    hour_intensity=1-betarnd(6,3);     
    factor(i,1)=hour_intensity;
  end
  factor(i,1)=factor(i,1)/duration;
end
end
