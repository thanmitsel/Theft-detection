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

fraud_rate=0.5; % Percentage of consumers who fraud
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

normalization=0;
P=0.3; % Percent of Test

[X_train, Y_train, X_test, Y_test, X_full, Y_full]=pickTrainTest(Xn, Yn, P, normalization);
Intr=sum(Y_full)/size(Y_full,1);% Probability of Intrusion based on Days

fprintf('\nSegmented Training and Testing.\n');
%% ===  PCA for Visualization ===
% Use PCA to project this cloud to 2D for visualization
% Subtract the mean to use PCA
[X_norm, mu, sigma] = normalizeMinus_Plus(X_full);

% PCA and project the data to 2D
% [U, S] = pca(X_norm);
% Z = projectData(X_norm, U, 2);

Z=X_full;

% plotClass(Z(:,:),Y(:));
% title('Classified examples');

% fprintf('Program paused. Press enter to continue.\n');
% pause;

%% ===  KMeans ===
%  One useful application of PCA is to use it to visualize high-dimensional
%  data. In the last K-Means exercise you ran K-Means on 3-dimensional 
%  pixel colors of an image. We first visualize this output in 3D, and then
%  apply PCA to obtain a visualization in 2D.

min_cost=zeros(5,1); % Carefull the size dependes on multiple initializations

max_iters = 10;
K=3;   

% 5 random initialazations
cost=1000000000;
for j=1:5
    initial_centroids = kMeansInitCentroids(Z, K); 
    [temp_cost, temp_centroids, temp_idx] = runkMeans(Z, initial_centroids, max_iters);
    if cost>temp_cost % Pick the lowest cost
        cost=temp_cost;
        centroids=temp_centroids;
        idx=temp_idx;
    end
end 

%fprintf('Program paused. Press enter to continue.\n');
%pause;

%% Apply anomalyDetection
% Estimate mu sigma2
precision_v=zeros(K,1);
recall_v=zeros(K,1);
in_recall_v=zeros(K,1);
accuracy_v=zeros(K,1);
F1score_v=zeros(K,1);
BDR_v=zeros(K,1);
for i=1:K
    [train, test]=crossvalind('HoldOut', size(Z(idx==i,:),1), 0.3);
    X_train=Z(train,:);
    Y_train=Y_full(train);
    X_test=Z(test,:);
    Y_test=Y_full(test);
    [mu, sigma2] = estimateGaussian(X_train);

    %  Training set 
    p = multivariateGaussian(X_train, mu, sigma2);

    %  Cross-validation set
    pval = multivariateGaussian(X_test, mu, sigma2);

    %  Find the best threshold
    [epsilon, F1] = selectThreshold(Y_test, pval);
    prediction=(pval<epsilon);

    % Create confusion Matrix
    % Detection Rate is Recall, False Positive Rate is Inverse recall 
    [precision, recall, in_recall, accuracy, F1score] = confusionMatrix (Y_test, prediction);
    if i==1
        Y_tf=Y_test;
        pred_f=prediction;
    else
        Y_tf=[Y_tf; Y_test];
        pred_f=[pred_f; prediction];
    end
    BDR=Intr*recall/(Intr*recall+(1-Intr)*in_recall) ; % Bayesian Detection Rate for days
    precision_v(i,1)=precision;
    recall_v(i,1)=recall;
    in_recall_v(i,1)=in_recall;
    accuracy_v(i,1)=accuracy;
    F1score_v(i,1)=F1score;
    BDR_v(i,1)=BDR;
end
precision=mean(precision_v);

recall=mean(recall_v);
in_recall=mean(in_recall_v);
accuracy=mean(accuracy_v);
F1score=mean(F1score_v);
BDR=mean(BDR_v);

rouf_id=find(prediction==1);
roufianos=someID(rouf_id); % Keeps all the ID that contain intrusion

fprintf('\nBest epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('Based on best F1\nDR=%4.2f FPR=%4.2f on Cross Validation Set\n', recall, in_recall);
fprintf('# Outliers found: %d\n', sum(p < epsilon));

%% Printing Segment
fprintf('kWh Rate %4.2fper | Time Rate %4.2fper |\n',kWh_rate,time_rate);

% fprintf('Black List\n')
% disp(roufianos); too many to print
Y_test=Y_tf;
prediction=pred_f;
fprintf('\nClassification for Days\n');
fprintf('| Precision %4.2f | Recall %4.2f | Accuracy %4.2f | F1score %4.2f |\n',precision,recall,accuracy,F1score);
fprintf('| Actual Fraud %d Days | Predicted Fraud Right %d Days | Predicted Fraud Wrong %d Days |\n',sum(Y_test==1),sum(prediction==1&Y_test==prediction),sum(prediction==1&Y_test~=prediction));
fprintf(' DR  FPR  BDR  Accuracy\n%4.2f %4.2f %4.2f %4.2f \n',recall,in_recall,BDR,accuracy);