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
[d]=convertDays3D(hh);
%% Fraud Initialization
% Create Fraud data
fprintf('Choose classification and press Enter.\n');
prompt='\n0. Days\n1. Hours\n2. Half hours\n';
x = input(prompt);

fraud_rate=0.35; % Percentage of consumers who fraud
if x==0 || x==1 
    F_data3D=H;
    f_data2D=h;
    Y2D=zeros(size(H,1),size(H,3));
    one_H=zeros(size(H(:,:,1)));

    
    [normal_idx, fraud_idx] = crossvalind('HoldOut', size(H,3), fraud_rate); % Keep in mind crossval floors the rate
    thiefs=find(fraud_idx==1);
    for i=1:size(thiefs,1)
        one_H=H(:,:,thiefs(i));
        [f_data, y, F_data,Y] = type3Fraud(one_H);
        f_data2D(thiefs(i),:)=f_data';
        F_data3D(:,:,thiefs(i))=F_data;
        Y2D(:,thiefs(i))=Y;
    end
    % Details 
    [kWh_count, time_count, kWh_rate, time_rate] = frauDetails(H, F_data3D);
elseif x==2
    F_data3D=HH;
    f_data2D=somehh;
    Y2D=zeros(size(HH,1),size(HH,3));
    one_HH=zeros(size(HH(:,:,1)));

    [normal_idx, fraud_idx] = crossvalind('HoldOut', size(H,3), fraud_rate); % Keep in mind crossval floors the rate
    thiefs=find(fraud_idx==1);
    for i=1:size(thiefs,1)
        one_HH=HH(:,:,thiefs(i));
        [f_data, y, F_data,Y] = type3Fraud(one_HH);
        f_data2D(thiefs(i),:)=f_data';
        F_data3D(:,:,thiefs(i))=F_data;
        Y2D(:,thiefs(i))=Y;
    end
    % Details 
    [kWh_count, time_count, kWh_rate, time_rate] = frauDetails(HH, F_data3D);
end

fprintf('\nFraud Simulation done.\n');
%% Load time series
% Here we use SVM for many consumers

% No Feature extraction for cons x (time division)
% Needs threshold
ndays=0;
if x==0
    [Xn]=convertDays3D(F_data3D); % if more days than the threshold then fraud
    ndays=0;
elseif x==1
    Xn=f_data2D;
    ndays=0;
elseif x==2
    Xn=f_data2D;
    ndays=0;    
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

%% SVM test and confusion matrices
% Test SVM
prompt='Select:\n0.libSVM(linear Kernel)\n1.libLinear\n';
x=input(prompt);
[Y_train, X_train]=get_sparse(Y_train, X_train);
[Y_test, X_test]=get_sparse(Y_test, X_test);

vfolds=5;
if x==0
    % arguments=['-t ' num2str(0) ' -v ' num2str(vfolds)]; % use linear kernel
    arguments=['-t ' num2str(0)]; % use linear kernel 
    model=svmtrain(Y_train,X_train,arguments);
    prediction= svmpredict(Y_test,X_test,model);
elseif x==1   
    %arguments=['-s ' num2str(0) ' -v ' num2str(vfolds)]; % not v-validation mode
    arguments=['-s ' num2str(1)]; % L2-Regularized L2-loss support vector classification (dual)
    model=train(Y_train,X_train, arguments);
    prediction= predict(Y_test,X_test,model);
end
pred_ID=prediction;

% Create confusion Matrix
% Detection Rate is Recall, False Positive Rate is Inverse recall 
[precision, recall, in_recall, accuracy, F1score] = confusionMatrix (Y_test, prediction);
BDR=Intr*recall/(Intr*recall+(1-Intr)*in_recall) ; % Bayesian Detection Rate for days

rouf_id=find(pred_ID==1);
roufianos=someID(rouf_id); % Keeps all the ID that contain intrusion

%% Printing Segment
fprintf('kWh Rate %4.2fper | Time Rate %4.2fper |\n',kWh_rate,time_rate);

% fprintf('Black List\n')
% disp(roufianos);

fprintf('\nClassification for Consumers\n');
fprintf('| Precision %4.2f | Recall %4.2f | Accuracy %4.2f | F1score %4.2f |\n',precision,recall,accuracy,F1score);
fprintf('| Actual Fraud %d IDs | Predicted Fraud Right %d IDs | Predicted Fraud Wrong %d IDs |\n',sum(Y_test==1),sum(prediction==1&Y_test==prediction),sum(prediction==1&Y_test~=prediction));
fprintf(' DR   FPR  BDR Accuracy F1score\n%4.2f %4.2f %4.2f %4.2f   %4.2f \n',recall,in_recall,BDR,accuracy,F1score);

prompt='\n-Delete train data?\n0.KEEP/n1.DELETE\n';
delete_data=input(prompt);
if delete_data==1
    delete libSVM.train;
end