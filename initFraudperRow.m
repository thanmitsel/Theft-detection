function [tstart, duration] = initFraudperRow (data)
% NOTE when tstart is 24 duration must be manually set to 1
% edit NOTE tstart is based on the colunms of matrix
duration=randi(size(data,2));
tstart=randi(size(data,2));
while (tstart+duration>size(data,2))
  duration=duration-1;
end
end
