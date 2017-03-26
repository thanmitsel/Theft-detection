function [tstart, duration] = initFraudperRow (data)
% NOTE when tstart is 24 duration must be manually set to 1
duration=randi(24);
tstart=randi(24);
while (tstart+duration>size(data,2))
  duration=duration-1;
end
end
