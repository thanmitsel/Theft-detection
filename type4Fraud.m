function [factor] = type4Fraud (data)
% factor is the average of the intensity per fraud duration on day
[fraudDays] = initFraudperMatrix(data);
% dstart=fraudDays(1);
for i=1:length(fraudDays)
  factor(i,1)=0;
  [tstart, duration, intensity] = initFraudperRow(data); %intensity not needed
  if tstart~=24
    for j=tstart:(tstart+duration)
      [patates, agouria, hour_intensity] = initFraudperRow(data); %patates-agouria not needed (expensive)
        factor(i,1)=factor(i,1)+hour_intensity;    
     end
  else % when start is 24th hour
    duration=1; 
    factor(i,1)=intensity;
  end
  factor(i,1)=factor(i,1)/duration;
end
end
