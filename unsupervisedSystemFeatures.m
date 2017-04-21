% pick some z vector 
z=1000;
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
fraud_rate=0.35; % Percentage of consumers who fraud
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
ndays=80;
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
%[U, S] = pca(X_norm);
%Z = projectData(X_norm, U, 2);


%plotClass(Z(:,:),Y(:));
%title('Classified examples');

%fprintf('Program paused. Press enter to continue.\n');
%pause;

%% ===  KMeans ===
%  One useful application of PCA is to use it to visualize high-dimensional
%  data. In the last K-Means exercise you ran K-Means on 3-dimensional 
%  pixel colors of an image. We first visualize this output in 3D, and then
%  apply PCA to obtain a visualization in 2D.

max_iters = 10;
K=3;   

% 5 random initialazations
cost=1000000000;
for j=1:5
    initial_centroids = kMeansInitCentroids(X_full, K); 
    [temp_cost, temp_centroids, temp_idx] = runkMeans(X_full, initial_centroids, max_iters);
    if cost>temp_cost % Pick the lowest cost
        cost=temp_cost;
        centroids=temp_centroids;
        idx=temp_idx;
    end
end 

%sel = floor(rand(1000, 1) * size(Z, 1)) + 1;
%  Setup Color Palette
%palette = hsv(K);
%colors = palette(idx(sel), :);

% Plot in 2D
%figure;
%plotDataPoints(Z(:, :), idx(:), K);
%title('2D plot of features, using PCA for dimensionality reduction');

%fprintf('Program paused. Press enter to continue.\n');
%pause;

%% Apply AnomalyDetection
% Estimate mu sigma2
precision_v=zeros(K,1);
recall_v=zeros(K,1);
in_recall_v=zeros(K,1);
accuracy_v=zeros(K,1);
F1score_v=zeros(K,1);
BDR_v=zeros(K,1);

test_rate=0.3;
Y_tf=zeros(test_rate*size(Y_full,1),1);
pred_f=zeros(test_rate*size(Y_full,1),1);
for i=1:K
    [train, test]=crossvalind('HoldOut', size(X_norm(idx==i,:),1), test_rate);
    X_train=X_full(train,:);
    Y_train=Y_full(train);
    X_test=X_full(test,:);
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

Y_table=vec2mat(Y_tf, floor(P*size(X,1)))';
class_ID=(sum(Y_table==1)>ndays)'; % fraud if more than ndays days

% Create confusion Matrix
% Detection Rate is Recall, False Positive Rate is Inverse recall 
pred_table=vec2mat(pred_f, floor(P*size(X,1)))';
pred_ID=(sum(pred_table==1)>ndays)'; % fraud if more than 'ndays' days
[precision_t, recall_t, in_recall_t, accuracy_t, F1score_t] = confusionMatrix (class_ID, pred_ID);
BDR_t=fraud_rate*recall_t/(fraud_rate*recall_t+ (1-fraud_rate)*in_recall_t); % Bayesian Detection Rate for Consumers

rouf_id=find(pred_ID==1);
%roufianos=someID(rouf_id); % Keeps all the ID that contain intrusion

fprintf('\nBest epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('Based on best F1\nDR=%4.2f FPR=%4.2f on Cross Validation Set\n', recall, in_recall);
fprintf('# Outliers found: %d\n', sum(p < epsilon));

%% Printing Segment
fprintf('\nThere are %d consumers with corrupted features!\n',count);
fprintf('kWh Rate %4.2fper | Time Rate %4.2fper |\n',kWh_rate,time_rate);

fprintf('\nClassification for IDs\n');
fprintf('| Precision %4.2f | Recall %4.2f | Accuracy %4.2f | F1score %4.2f |\n',precision_t,recall_t,accuracy_t,F1score_t);
fprintf('| Actual Fraud %d IDs | Predicted Fraud Right %d IDs | Predicted Fraud Wrong %d IDs |\n',sum(class_ID==1),sum(pred_ID==1&pred_ID==class_ID),sum(pred_ID==1&class_ID~=pred_ID));
fprintf(' DR  FPR BDR Accuracy\n%4.2f %4.2f % 4.2f %4.2f \n',recall_t,in_recall_t,BDR_t, accuracy_t);
fprintf('Black List\n')
% disp(roufianos); too many to print

fprintf('\nClassification for Days\n');
fprintf('| Precision %4.2f | Recall %4.2f | Accuracy %4.2f | F1score %4.2f |\n',precision,recall,accuracy,F1score);
fprintf('| Actual Fraud %d Days | Predicted Fraud Right %d Days | Predicted Fraud Wrong %d Days |\n',sum(Y_test==1),sum(prediction==1&Y_test==prediction),sum(prediction==1&Y_test~=prediction));
fprintf(' DR  FPR  BDR  Accuracy\n%4.2f %4.2f %4.2f %4.2f \n',recall,in_recall,BDR,accuracy);
