function [f_data,y, F_data,Y]=type1Fraud(data)
% Input is a matrix (i.e days*hours)
% Consumer learns to steal at dstart 
% and keeps the stealing method unchanged for a whole days
F_data=data;
Y=zeros(size(data,1),1);
[dstart, fraudDays] = initFraudperMatrix(data);
[tstart, duration, intensity] = initFraudperRow (data);
for i=1:length(fraudDays)
    F_data(fraudDays(i),:)=intensity*data(fraudDays(i),:);
    Y(i,1)=1;
end
f_data_temp=F_data'(:);
f_data=f_data_temp';
y=1;
end


