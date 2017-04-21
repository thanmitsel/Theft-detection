% pick some z vector 
z=100;
r_cons=randi(size(hh,1),z,1);
somehh=hh(r_cons,:);
someID=ID(r_cons,:);

% Convertions
[h, H]=convertHours3D(somehh);

%% Fraud Initialization
% Create Fraud data
F_data3D=H;
Y2D=zeros(size(H,1),size(H,3));
one_H=zeros(size(H(:,:,1)));

% fraud_rate=0.05; % Percentage of consumers who fraud
fraud_rate=0.10; % Percentage of consumers who fraud
[normal_idx, fraud_idx] = crossvalind('HoldOut', size(H,3), fraud_rate); % Keep in mind crossval floors the rate
thiefs=find(fraud_idx==1);
for i=1:size(thiefs,1)
    one_H=H(:,:,thiefs(i));
    [f_data, y, F_data,Y] = type3Fraud(one_H);
    F_data3D(:,:,thiefs(i))=F_data;
    Y2D(:,thiefs(i))=Y;
end

% Details 
[kWh_count, time_count, kWh_rate, time_rate] = frauDetails(H, F_data3D);

%% Feature extraction
% Here we use SVM for many consumers

fprintf('Choose input data for anomaly detection.\n');
prompt='\n0. Raw Data per hour\n1. Features (#14)\n';
x = input(prompt);
fprintf('You pressed: %d\n', x) ;
count=0; % counts the errored data
if x==0
    X=F_data3D;
elseif x==1
    % Feature extraction
    % 14 Features 
    X=zeros(size(H,1),14,size(H,3));
    
    for i=1:size(H,3)
    [maxi,maxT,mini,minT,suma,av,stdev...
            , lfactor, mintoav,mintomax,night,skew,kurt,varia, features]=extractFeatures(F_data3D(:,:,i));
        X(:,:,i)=features; % X NEEDS TO BE 3D, so nothing crashes later
        if (sum(find(isnan(X(:,:,i))))~=0)
        fprintf('There is NaN on the %dst client\n',i);
        count=count+1;
        % if find NaN get the values of the previous consumer
        X(:,:,i)=X(:,:,(i-1));
        Y2D(:,i)=Y2D(:,(i-1));
        end
    end
end
fprintf('\nFraud Data and features created.\n');
%% Create training and testing set
% Choose from every consumer sample
% No normarlization neededs

% ndays=20; % if more days than the threshold then fraud

P=0.3; % Percent of Test
normalization=0;
[X_train, Y_train, X_test, Y_test, X_full, Y_full]=pickTrainTest(X, Y2D, P, normalization);
Intr=sum(Y_full)/size(Y_full,1);% Probability of Intrusion based on Days


fprintf('\nSegmented Training and Testing.\n');
%% ===  PCA for Visualization ===
% Use PCA to project this cloud to 2D for visualization
% Subtract the mean to use PCA
[X_norm, mu, sigma] = normalizeMinus_Plus(X_full);

% PCA and project the data to 2D
[U, S] = pca(X_norm);
Z = projectData(X_norm, U, 2);


plotClass(Z(:,:),Y(:));
title('Classified examples');

%% Run DBSCAN Clustering Algorithm

%epsilon=0.5;
epsilon=0.5;
MinPts=10;
[IDX, isnoise]=DBSCAN(Z,epsilon,MinPts);

%% Plot Results

PlotClusterinResult(Z, IDX);
title(['DBSCAN Clustering (\epsilon = ' num2str(epsilon) ', MinPts = ' num2str(MinPts) ')']);

ndays=0;
pred_table=vec2mat(isnoise, size(X,1))';
pred_ID=(sum(pred_table==1)>ndays)'; % fraud if more than "ndays" days

Y_table=vec2mat(Y_full, size(X,1))';
class_ID=(sum(Y_table==1)>ndays)'; % fraud if more than "ndays" days

% Create confusion Matrix
% Detection Rate is Recall, False Positive Rate is Inverse recall 
[precision, recall, in_recall, accuracy, F1score] = confusionMatrix (Y_full,isnoise);
BDR=Intr*recall/(Intr*recall+(1-Intr)*in_recall) ; % Bayesian Detection Rate for days
[precision_t, recall_t, in_recall_t, accuracy_t, F1score_t] = confusionMatrix (class_ID, pred_ID);

BDR_t=fraud_rate*recall_t/(fraud_rate*recall_t+ (1-fraud_rate)*in_recall_t); % Bayesian Detection Rate for Consumers

rouf_id=find(pred_ID==1);
roufianos=someID(rouf_id); % Keeps all the ID that contain intrusion

%% Printing Segment
fprintf('\nThere are %d consumers with corrupted features!\n',count);
fprintf('kWh Rate %4.2fper | Time Rate %4.2fper |\n',kWh_rate,time_rate);

fprintf('\nClassification for IDs\n');
fprintf('| Precision %4.2f | Recall %4.2f | Accuracy %4.2f | F1score %4.2f |\n',precision_t,recall_t,accuracy_t,F1score_t);
fprintf('| Actual Fraud %d IDs | Predicted Fraud Right %d IDs | Predicted Fraud Wrong %d IDs |\n',sum(class_ID==1),sum(pred_ID==1&pred_ID==class_ID),sum(pred_ID==1&class_ID~=pred_ID));
fprintf(' DR  FPR BDR Accuracy\n%4.2f %4.2f % 4.2f %4.2f \n',recall_t,in_recall_t,BDR_t, accuracy_t);
% fprintf('Black List\n')
% disp(roufianos);

fprintf('\nClassification for Days\n');
fprintf('| Precision %4.2f | Recall %4.2f | Accuracy %4.2f | F1score %4.2f |\n',precision,recall,accuracy,F1score);
fprintf('| Actual Fraud %d Days | Predicted Fraud Right %d Days | Predicted Fraud Wrong %d Days |\n',sum(Y_full==1),sum(isnoise==1&Y_full==isnoise),sum(isnoise==1&Y_full~=isnoise));
fprintf(' DR  FPR  BDR  Accuracy\n%4.2f %4.2f %4.2f %4.2f \n',recall,in_recall,BDR,accuracy);