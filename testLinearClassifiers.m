%% Preprocesing, Formating Data for many consumers
% In this script we get consumer data and add fraud values to their data.

% pick some z vector 
z=2000;
r_cons=randi(size(hh,1),z,1);
somehh=hh(r_cons,:);
HH=zeros(365,48,size(somehh,1));
for j=1:size(somehh,1)
    HH(:,:,j)=vec2mat(somehh(j,:),48);
end
someID=ID(r_cons,:);

% Convertions
[h, H]=convertHours3D(somehh);
[d]=convertDays3D(somehh);
%% Fraud Initialization
% Create Fraud data
fprintf('Choose type of attack.\n');
prompt='\n1. Type 1\n2. Type 2\n3. Type 3\n';
attack_type= input(prompt);

fprintf('Choose classification and press Enter.\n');
prompt='\n0. Days (Fast)\n1. Hours (Default)\n2. Half hours (Slow)\n';
x = input(prompt);
if x==2 % else simulation with hours
    H=HH;
    h=somehh;
end

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

fprintf('\nFraud Simulation done.\n');
%% Load time series
% Here we use SVM for many consumers

% No Feature extraction for cons x (time division)
% Needs threshold
ndays=0;
if x==0 % convert for days
	[Xn]=convertDays3D(F_data3D); % if more days than the threshold then fraud
else
	Xn=f_data2D;
end
Yn=(sum(Y2D)>ndays)';
    
    
fprintf('\nLoading of time series done.\n');
%% Create training and testing set
% Choose from every consumer sample

P=0.3; % Percent of Test
normalization=1;
[X_train, Y_train, X_test, Y_test, X_full, Y_full]=pickTrainTest(Xn, Yn, P, normalization);
Intr=sum(Y_full)/size(Y_full,1);% Probability of Intrusion based on Days

fprintf('\nSegmented Training and Testing.\n');
%% Test every method of libLinear

[Y_train, X_train]=get_sparse(Y_train, X_train);
[Y_test, X_test]=get_sparse(Y_test, X_test);

arguments=['-s ' num2str(2)];
tic
model=train(Y_train,X_train, arguments);
toc
prediction= predict(Y_test,X_test,model);
    
[precision, recall, in_recall, accuracy, F1score] = confusionMatrix (Y_test, prediction);
BDR=Intr*recall/(Intr*recall+(1-Intr)*in_recall) ; % Bayesian Detection Rate for days
fprintf('Option & DR  & FPR & Accuracy & F1 score & BDR \n');
fprintf('%d & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \n',mode(i),recall,in_recall,accuracy,F1score,BDR);

pause;
%% Explore Linear Classifiers 
mode=0:7;
fprintf('Option & DR  & FPR & Accuracy & F1 score & BDR \n');
for i=1:size(mode,2)
    arguments=['-s ' num2str(mode(i))];
    model=train(Y_train,X_train, arguments);
    prediction= predict(Y_test,X_test,model);
    
    [precision, recall, in_recall, accuracy, F1score] = confusionMatrix (Y_test, prediction);
    BDR=Intr*recall/(Intr*recall+(1-Intr)*in_recall) ; % Bayesian Detection Rate for days
    fprintf('%d & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \n',mode(i),recall,in_recall,accuracy,F1score,BDR);
end