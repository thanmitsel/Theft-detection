function plotConsumption(data)

figure; hold on;
data=data'; % Only for script1
[n, p]=size(data);
t= 1:n;
plot(t, data);
xlabel('Time'), ylabel('Consumption (kWh)');
title('Consumption recorded from Smart Meters');

hold off;
end
