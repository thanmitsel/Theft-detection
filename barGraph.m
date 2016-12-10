function barGraph(data)
% Bar graph for first consumer
figure; hold on;
x=1:size(data,1);
y=data(:,1);
bar(x,y);
xlabel('Time'), ylabel('Consumption (kWh)');
title('Consumption recorded from Smart Meters');
hold off;
end