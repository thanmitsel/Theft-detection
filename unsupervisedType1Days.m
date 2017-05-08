    % pick some z vector 
    z=2000;
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
fraud_rate=0.1; % Percentage of consumers who fraud
[normal_idx, fraud_idx] = crossvalind('HoldOut', size(H,3), fraud_rate); % Keep in mind crossval floors the rate
thiefs=find(fraud_idx==1);
for i=1:size(thiefs,1)    
    intensity=1-betarnd(6,3); % beta distribution
    dstart=floor(normrnd(size(one_H,1)/2,size(one_H,1)/6.5)); % normal distribution
    while dstart<1 || dstart>(size(one_H,1)-1)
        dstart=floor(normrnd(size(one_H,1)/2,size(one_H,1)/6.5)); % normal distribution
    end
    one_H=H(:,:,thiefs(i));
    [f_data, y, F_data,Y] = type1_2Fraud(one_H, intensity,dstart);
    F_data3D(:,:,thiefs(i))=F_data;
    Y2D(:,thiefs(i))=Y;
end

% Details 
[kWh_count, time_count, kWh_rate, time_rate] = frauDetails(H, F_data3D);

%% Feature extraction
% Here we use SVM for many consumers
prompt=('Get sparce vectors for features?\n 1. Yes 0. No\n');
sparce=input(prompt);
% Feature extraction
av_per_dif=0.6;
std_per_dif=0.6;
symmetric_av=0.6;
symmetric_std=0.6;
[X_1]=ConsumerFeaturesDays(F_data3D, av_per_dif, std_per_dif, symmetric_av, symmetric_std, sparce); %includes ONLY 3 features
X_cons=mean(X_1,1);
X_cons=permute(X_cons, [3 2 1]);
[X_2,Y]=unrollto2D(X_1, Y2D);
Y_cons=(sum(Y2D)>1)'; % if more than one hour consider fraud

% Obtain features related to neighbors
K=3;
av_threshold=0.6;
std_threshold=0.6;
[X_neighborhood]=NeighborFeaturesDays(X_1, X_cons, K, av_threshold, std_threshold, sparce);
X=[X_2 X_neighborhood];
fprintf('\nFraud Data and features created.\n');
%% ===  PCA for Visualization ===
% Use PCA to project this cloud to 2D for visualization
% Subtract the mean to use PCA
prompt=('Apply normalization?\n 0 w/o norm, 1 with norm\n');
x=input(prompt);
if x==0
    X_norm=X;
elseif x==1
    [X_norm, mu, sigma] = normalizeMinus_Plus(X);
end
prompt=('Apply PCA?\n 0 w/o PCA, 1 with PCA\n');
x=input(prompt);
if x==0
    Z=X_norm;
elseif x==1
    % PCA and project the data to 2D
    [U, S] = pca(X_norm);
    Z = projectData(X_norm, U, 2);
end
%% Create training and testing set
% Choose from every consumer sample
% No normarlization needed here

[X3D]=reshapeto3D(Z,size(X_1,3));
ndays=10;
P=0.3; % Percent of Test
normalization=0;
[X_train, Y_train, X_test, Y_test, X_full, Y_full]=pickTrainTest(X3D, Y2D, P, normalization);
Intr=sum(Y_full)/size(Y_full,1);% Probability of Intrusion based on Days

Y_table=vec2mat(Y_test, floor(P*size(X_1,1)))';
class_ID=(sum(Y_table==1)>ndays)'; % fraud if more than 1 days

fprintf('\nSegmented Training and Testing.\n');
%% Apply anomalyDetection
% Estimate mu sigma2
[mu, sigma2] = estimateGaussian(X_train);

%  Training set 
p = multivariateGaussian(X_train, mu, sigma2);

%  Cross-validation set
pval = multivariateGaussian(X_test, mu, sigma2);

%  Find the best threshold
[epsilon, F1] = selectThreshold(Y_test, pval);
prediction=(pval<epsilon);

pred_table=vec2mat(prediction, floor(P*size(X_1,1)))';
pred_ID=(sum(pred_table==1)>ndays)'; % fraud if more than "ndays" days


% Create confusion Matrix
% Detection Rate is Recall, False Positive Rate is Inverse recall 
[precision, recall, in_recall, accuracy, F1score] = confusionMatrix (Y_test, prediction);
BDR=Intr*recall/(Intr*recall+(1-Intr)*in_recall) ; % Bayesian Detection Rate for days

[precision_t, recall_t, in_recall_t, accuracy_t, F1score_t] = confusionMatrix (class_ID, pred_ID);
BDR_t=fraud_rate*recall_t/(fraud_rate*recall_t+ (1-fraud_rate)*in_recall_t); % Bayesian Detection Rate for Consumers

rouf_id=find(pred_ID==1);
roufianos=someID(rouf_id); % Keeps all the ID that contain intrusion

fprintf('\nBest epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('Based on best F1\nDR=%4.2f FPR=%4.2f on Cross Validation Set\n', recall, in_recall);
fprintf('# Outliers found: %d\n', sum(p < epsilon));

%% Printing Segment
fprintf('kWh Rate %4.2fper | Time Rate %4.2fper |\n',kWh_rate,time_rate);
fprintf('\nClassification for IDs\n');
fprintf('| Precision %4.2f | Recall %4.2f | Accuracy %4.2f | F1score %4.2f |\n',precision_t,recall_t,accuracy_t,F1score_t);
fprintf('| Actual Fraud %d IDs | Predicted Fraud Right %d IDs | Predicted Fraud Wrong %d IDs |\n',sum(class_ID==1),sum(pred_ID==1&pred_ID==class_ID),sum(pred_ID==1&class_ID~=pred_ID));
fprintf(' DR  FPR BDR Accuracy\n%4.2f %4.2f % 4.2f %4.2f \n',recall_t,in_recall_t,BDR_t, accuracy_t);
% fprintf('Black List\n')
% disp(roufianos);

fprintf('\nClassification for Days\n');
fprintf('| Precision %4.2f | Recall %4.2f | Accuracy %4.2f | F1score %4.2f |\n',precision,recall,accuracy,F1score);
fprintf('| Actual Fraud %d Days | Predicted Fraud Right %d Days | Predicted Fraud Wrong %d Days |\n',sum(Y_test==1),sum(prediction==1&Y_test==prediction),sum(prediction==1&Y_test~=prediction));
fprintf(' DR  FPR  BDR  Accuracy\n%4.2f %4.2f %4.2f %4.2f \n',recall,in_recall,BDR,accuracy);
