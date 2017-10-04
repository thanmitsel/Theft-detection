%% Semi supervised algorithm with a twisted Anomaly Detection
  %  form cluster based on rule based system
    % pick some z vector 
    z=4500;
    r_cons=randi(size(hh,1),z,1);
    somehh=hh(r_cons,:);
    someID=ID(r_cons,:);

    % Convertions
    [h, H]=convertHours3D(somehh);

%% Fraud Initialization
% Create Fraud data
fprintf('Choose type of attack.\n');
prompt='\n1. Type 1\n2. Type 2\n3. Type 3\n4. Mixed\n';
attack_type= input(prompt);

fraud_rate=0.1; % Percentage of consumers who fraud
if attack_type==3
    [F_data3D, Y2D, f_data2D]=simulateType3Fraud(H,h,fraud_rate);
    % Details 
    [kWh_count, time_count, kWh_rate, time_rate] = frauDetails(H, F_data3D);
elseif attack_type==1
    [F_data3D, Y2D, f_data2D]=simulateType1Fraud(H,h, fraud_rate);
    % Details 
    [kWh_count, time_count, kWh_rate, time_rate] = frauDetails(H, F_data3D);
elseif attack_type==2
    [F_data3D, Y2D, f_data2D]=simulateType2Fraud(H,h,fraud_rate);  
    % Details 
    [kWh_count, time_count, kWh_rate, time_rate] = frauDetails(H, F_data3D);
elseif attack_type==4
    [F_data3D, Y2D, f_data2D]=simulateMixedFraud(H,h, fraud_rate);
    % Details 
    [kWh_count, time_count, kWh_rate, time_rate] = frauDetails(H, F_data3D);
end

fprintf('\nFraud Simulation done.\n');%% Form Cluster

daily_fraud3D=sum(F_data3D,2);
daily_fraud=permute(daily_fraud3D,[3 1 2]); % cons x 365 days    
daily_consumption3D=sum(H,2);
daily_consumption=permute(daily_consumption3D,[3 1 2]); % cons x 365 days    

Y=(sum(Y2D)>0)';

positive_fraud=daily_fraud(Y==1,:);
positive_consumption=daily_consumption(Y==1,:);
figure;
plot(positive_fraud(1,:),'r');
hold on;
plot(positive_consumption(1,:),'b');
h1 = gca;
h1.XLim = [0,365];
h1.XTick = 0:25:365;
if attack_type==1
    str='Απάτη Τύπου 1';
elseif attack_type==2
    str='Απάτη Τύπου 2';
elseif attack_type==3
    str='Απάτη Τύπου 3';
else
    str='Απάτη Μικτού Τύπου';
end
title(str);
    
legend('Προσομοίωση','Χρονοσειρά');
xlabel('Ημέρες');
ylabel('kWh');
hold off;
