function [f_data,y, F_data,Y]=type1_1Fraud(data, intensity)
% Created for ROC cureves and outside control on intensity
% Input is a matrix (i.e days*hours) and intensity
% Consumer learns to steal at dstart 
% and keeps the stealing method unchanged for the remaining of the year
% without stopping
F_data=data;
Y=zeros(size(data,1),1);
[fraudDays] = initFraudperMatrix(data);
% dstart=fraudDays(1);
% intensity=1-betarnd(6,3);
for i=1:length(fraudDays)
    F_data(fraudDays(i),:)=intensity*data(fraudDays(i),:);
    Y(i,1)=1;
end
f_data_temp=F_data(:)';
f_data=f_data_temp';
y=1;
end


