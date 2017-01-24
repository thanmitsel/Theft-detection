function [sdata] = sortID(data)
% sort matrix according to second column (time)
sdata = sortrows(data,2);
% sort matrix according to first column (meter_id)
sdata = sortrows(sdata,1);
