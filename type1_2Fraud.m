function [f_data,y, F_data,Y]=type1_2Fraud(data, intensity, dstart)
% Access full control over type1Fraud
% Input is a matrix (i.e days*hours) and intensity
% Consumer learns to steal at dstart 
% and keeps the stealing method unchanged for the remaining of the year
% without stopping

Y=zeros(size(data,1),1);

F_data_only=intensity*data(dstart:end,:);
F_data=[data(1:(dstart-1),:); F_data_only];
Y(dstart:end)=1;

f_data_temp=F_data(:)';
f_data=f_data_temp';
y=1;
end


