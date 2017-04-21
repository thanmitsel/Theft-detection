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

%% Feature extraction
% Here we use SVM for many consumers

% No Feature extraction for cons x Days Data
% Needs threshold 
if x==0
    [Xn]=convertDays3D(F_data3D); % if more days than the threshold then fraud
    ndays=0;
    Yn=(sum(Y2D)>ndays)';
elseif x==1
    Xn=f_data2D;
    ndays=0;
    Yn=(sum(Y2D)>ndays)';
elseif x==2
    Xn=f_data2D;
    ndays=0;
    Yn=(sum(Y2D)>ndays)';
end
    
    
    
fprintf('\nFraud Data and features created.\n');
%% Create training and testing set
% Choose from every consumer sample

P=0.3; % Percent of Test
normalization=1;
[X_train, Y_train, X_test, Y_test, X_full, Y_full]=pickTrainTest(Xn, Yn, P, normalization);
Intr=sum(Y_full)/size(Y_full,1);% Probability of Intrusion based on Days

fprintf('\nSegmented Training and Testing.\n');

%% SVM test and confusion matrices
% Test SVM
[Y_full, X_full]=get_sparse(Y_full, X_full);

arg_v=[0; 0; 1; 3; 4; 5; 6];
accuracy_v=zeros(size(arg_v,1),1);
vfolds=5;
for i=1:size(arg_v,1)
    if i==1
        arguments=['-t ' num2str(arg_v(i)) ' -v ' num2str(vfolds)]; % use linear kernel
        accuracy_v(i,1)=svmtrain(Y_full,X_full,arguments);        
    else 
        arguments=['-s ' num2str(arg_v(i)) ' -v ' num2str(vfolds)]; % not v-validation mode      
        accuracy_v(i,1)=train(Y_full,X_full, arguments);
    end
end


%% Printing Segment
c = categorical({'libSVM\nlinear kernel','libLinear\nL2-RLR',...
    'libLinear\nL2-R, L2-LSVC', 'libLinear\nL2-R, L1-LSVC',...
    'libLinear\nCrammer Singer', 'libLinear\nL1-R, L2-LSVC', 'libLinear\nL1-RLR'});
bar(c,accuracy_v);
title('Accuracy on Cross Validation');