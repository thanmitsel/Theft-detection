function plotConsumpion(data)

figure; hold on;

[n, p]=size(data);
t= 1:n;
plot(t, data);
xlabel('Time'), ylabel('Consumption (kWh)');
title('Consumption recorded from Smart Meters');

hold off;
end