function [kWh_count, time_count, kWh_rate, time_rate] = frauDetails (Data, F_data)
kWh_count=sum(sum(Data-F_data));
time_count=sum(sum(Data~=F_data));
time_rate=time_count/(size(Data,1)*size(Data,2))*100;
kWh_rate=kWh_count/sum(sum(Data))*100; 
end
