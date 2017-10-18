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
prompt='\n1. Type 1\n2. Type 2\n3. Type 3\n4. Mixed\n';
attack_type= input(prompt);

fprintf('Choose classification and press Enter.\n');
prompt='\n0. Days (Fast)\n1. Hours (Default)\n2. Half hours (Slow)\n';
x = input(prompt);
if x==2 % else simulation with hours
    H=HH;
    h=somehh;
end

fraud_rate=0.5; % Percentage of consumers who fraud
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
prompt=('What kind of normalization?\n');
norm=input(prompt);

P=0.3; % Percent of Test
if norm==0
    normalization=0;
    [X_train, Y_train, X_test, Y_test, X_full, Y_full]=pickTrainTest(Xn, Yn, P, normalization);
elseif norm==1
    normalization=1;
    [X_train, Y_train, X_test, Y_test, X_full, Y_full]=pickTrainTest(Xn, Yn, P, normalization);   
elseif norm==2
    [train_idx, test_idx]=crossvalind('HoldOut',z,P);
    X_train=Xn(train_idx,:);
    Y_train=Yn(train_idx);
    [X_train, mu, sigma] = normalizeMinus_Plus(X_train);
    X_test=Xn(test_idx,:);
    Y_test=Yn(test_idx);
    [X_test] = normalizeMinus_PlusTest(X_test, mu, sigma);
    X_full=[X_train;X_test];
    Y_full=Yn;
elseif norm==3
    Xn_t=Xn';
    [Xn_t, mu, sigma] = normalizeMinus_Plus(Xn_t);
    Xn=Xn_t';
    [train_idx, test_idx]=crossvalind('HoldOut',z,P);
    X_train=Xn(train_idx,:);
    Y_train=Yn(train_idx);
    X_test=Xn(test_idx,:);
    Y_test=Yn(test_idx);
    X_full=[X_train;X_test];
    Y_full=Yn;
elseif norm==4
    Xn_t=Xn';
    [Xn_t, ~, ~] = normalizeFeatures(Xn_t);
    Xn=Xn_t';
    [train_idx, test_idx]=crossvalind('HoldOut',z,P);
    X_train=Xn(train_idx,:);
    Y_train=Yn(train_idx);
    X_test=Xn(test_idx,:);
    Y_test=Yn(test_idx);
    X_full=[X_train;X_test];
    Y_full=Yn;
end
Intr=sum(Y_full)/size(Y_full,1);% Probability of Intrusion based on Days
fprintf('\nSegmented Training and Testing.\n');
%% Choose type of test
prompt='Pick mode\n1. linear SVM\n2.libLinear\n3. Explore methods ';
test=input(prompt);
if test==1
    %% Test linear kernel SVM
    arguments=['-t ' num2str(0)]; % use linear kernel 
    tic
    model=svmtrain(double(Y_train),double(X_train),arguments);
    toc
    prediction= svmpredict(double(Y_test),double(X_test),model);
    [precision, recall, in_recall, accuracy, F1score] = confusionMatrix (Y_test, prediction);
    BDR=Intr*recall/(Intr*recall+(1-Intr)*in_recall) ; % Bayesian Detection Rate for days
    fprintf('Option & DR  & FPR & Accuracy & F1 score & BDR \n');
    fprintf('%d & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \n',mode(i),recall,in_recall,accuracy,F1score,BDR);

elseif test==2
    %% Test best method of libLinear

    [Y_train, X_train]=get_sparse(double(Y_train), double(X_train));
    [Y_test, X_test]=get_sparse(double(Y_test), double(X_test));

    arguments=['-s ' num2str(2)];
    tic
    model=train(Y_train,X_train, arguments);
    toc
    prediction= predict(Y_test,X_test,model);

    [precision, recall, in_recall, accuracy, F1score] = confusionMatrix (Y_test, prediction);
    BDR=Intr*recall/(Intr*recall+(1-Intr)*in_recall) ; % Bayesian Detection Rate for days
    fprintf('Option & DR  & FPR & Accuracy & F1 score & BDR \n');
    fprintf('%d & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \n',mode(i),recall,in_recall,accuracy,F1score,BDR);
elseif test==3
%% Explore Linear Classifiers 
    [Y_train, X_train]=get_sparse(double(Y_train), double(X_train));
    [Y_test, X_test]=get_sparse(double(Y_test), double(X_test));
    %[Y_train, X_train]=get_sparse(Y_train, X_train);
    %[Y_test, X_test]=get_sparse(Y_test, X_test);
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
end
delete libSVM.train;