%%  Semi-supervised script for ROC curves
% pick some z vector 
    z=2000;
    r_cons=randi(size(hh,1),z,1);
    somehh=hh(r_cons,:);
    someID=ID(r_cons,:);

    % Convertions
    [h, H]=convertHours3D(somehh);
    %% vectors and matrices for ROC curves
    % 3 curves, high medium and low intensity
    % 10 points on every curve, threshold

    intensity=[0.20 0.50 0.80];
    thresh=0:10:100; 

DR_days=zeros(length(thresh), length(intensity));
FPR_days=zeros(length(thresh), length(intensity));
DR_IDs=zeros(length(thresh), length(intensity));
FPR_IDs=zeros(length(thresh), length(intensity));

prompt=('Apply normalization?\n 0 w/o norm, 1 with norm\n');
normalization=input(prompt);

prompt=('Apply PCA?\n 0 w/o PCA, 1 with PCA\n');
apply_pca=input(prompt);
for id_i=1:length(intensity) 
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
    dstart=floor(normrnd(size(one_H,1)/2,size(one_H,1)/6.5)); % normal distribution
    while dstart<1 || dstart>(size(one_H,1)-1)
        dstart=floor(normrnd(size(one_H,1)/2,size(one_H,1)/6.5)); % normal distribution
    end
    one_H=H(:,:,thiefs(i));
    [f_data, y, F_data,Y] = type1_2Fraud(one_H, intensity(id_i),dstart);
    F_data3D(:,:,thiefs(i))=F_data;
    Y2D(:,thiefs(i))=Y;
end

% Details 
[kWh_count, time_count, kWh_rate, time_rate] = frauDetails(H, F_data3D);

%% Feature extraction
% Here we use SVM for many consumers
    
% Feature extraction
av_per_dif=0.6;
std_per_dif=0.6;
symmetric_av=0.6;
symmetric_std=0.6;
[X_1]=ConsumerFeaturesDays(F_data3D, av_per_dif, std_per_dif, symmetric_av, symmetric_std); %includes ONLY 3 features
X_cons=mean(X_1,1);
X_cons=permute(X_cons, [3 2 1]);
[X_2,Y]=unrollto2D(X_1, Y2D);
Y_cons=(sum(Y2D)>1)'; % if more than one hour consider fraud

% Obtain features related to neighbors
K=3;
av_threshold=0.6;
std_threshold=0.6;
[X_neighborhood]=NeighborFeaturesDays(X_1, X_cons, K, av_threshold, std_threshold);
X=[X_2 X_neighborhood];
fprintf('\nFraud Data and features created.\n');
%% ===  PCA for Visualization ===
% Use PCA to project this cloud to 2D for visualization
% Subtract the mean to use PCA

if normalization==0
    X_norm=X;
elseif normalization==1
    [X_norm, mu, sigma] = normalizeMinus_Plus(X);
end

if apply_pca==0
    Z=X_norm;
elseif apply_pca==1
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
for id_th=1:length(thresh) 
pred_ID=(sum(pred_table==1)>thresh(id_th))'; % fraud if more than "ndays" days
class_ID=(sum(Y_table==1)>thresh(id_th))'; % h if more than 1 days


% Create confusion Matrix
% Detection Rate is Recall, False Positive Rate is Inverse recall 
[precision, recall, in_recall, accuracy, F1score] = confusionMatrix (Y_test, prediction);
BDR=Intr*recall/(Intr*recall+(1-Intr)*in_recall) ; % Bayesian Detection Rate for days

[precision_t, recall_t, in_recall_t, accuracy_t, F1score_t] = confusionMatrix (class_ID, pred_ID);
BDR_t=fraud_rate*recall_t/(fraud_rate*recall_t+ (1-fraud_rate)*in_recall_t); % Bayesian Detection Rate for Consumers

rouf_id=find(pred_ID==1);
roufianos=someID(rouf_id); % Keeps all the ID that contain intrusion

DR_IDs(id_th, id_i)=recall_t;
FPR_IDs(id_th,id_i)=in_recall_t;
end
end
plotCurves(FPR_IDs,DR_IDs,z,fraud_rate,thresh);