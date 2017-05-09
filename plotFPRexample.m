function plotFPRexample(consumption)
% Consumption must be from a single consumer

plot(1:365,consumption);
title('Daily Consumption')
h1= gca;
h1.XLim=[1,365];
ylabel 'Consumption kWh';
xlabel 'Days';