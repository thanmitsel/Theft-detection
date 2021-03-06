function oneDayFraudvsNormal (Data, F_data, Y)
oneDay=find(Y);
figure; hold on;
[n, p]=size(Data(1,:));
t=1:p;
plot(t,F_data(oneDay(1),:),'color','r'); hold on;
plot(t,Data(oneDay(1),:),'color','b');
xlabel('Time'), ylabel('Consumption (kWh)');
title('Fraud data vs Genuine Data (Day)');
legend({'Fraud','Genuine'});
hold off;


end
