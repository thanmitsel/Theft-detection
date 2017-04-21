function [factor] = type4_1Fraud (data, hour_intensity)
% factor is the average of the intensity per fraud duration on day
[fraudDays] = initFraudperMatrix(data);
factor=zeros(length(fraudDays),1);
for i=1:length(fraudDays)
  [tstart, duration] = initFraudperRow(data); %intensity not needed
  if tstart~=24
    for j=tstart:(tstart+duration)
        factor(i,1)=factor(i,1)+hour_intensity;    
     end
  else % when start is 24th hour
    duration=1; 
    factor(i,1)=hour_intensity;
  end
  factor(i,1)=factor(i,1)/duration;
end
end
