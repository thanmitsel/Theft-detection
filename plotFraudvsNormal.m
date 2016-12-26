function plotFraudvsNormal(data, fdata)

figure; hold on;
[n, p]=size(data);
t=1:p;
plot(t,data,'color','b'); hold on;
plot(t,fdata,'color','r');
xlabel('Time'), ylabel('Consumption (kWh)');
title('Fraud data vs Genuine Data (Year)');
legend({'Genuine','Fraud'});
hold off;
end
