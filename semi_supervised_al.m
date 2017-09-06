  %% Semi-supervised algorithm with 
  %  form cluster
    % pick some z vector 
    z=4500;
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
%% Form Cluster
daily_consumption3D=sum(F_data3D,2);
daily_consumption=permute(daily_consumption3D,[3 1 2]); % cons x 365 days    
daily_consumption_t=daily_consumption';    
prompt=('Apply horizontal normalization?\n 0 w/o norm, 1 with norm [-1,1], 2 with norm [0,1]\n');    
apply_normalization=input(prompt);   
if apply_normalization==0    
    daily_norm_t=daily_consumption_t;   
elseif apply_normalization==1    
    [daily_norm_t, mu, sigma] = normalizeMinus_Plus(daily_consumption_t);    
elseif apply_normalization==2    
    [daily_norm_t,~,~]=normalizeFeatures(daily_consumption_t);    
end

daily_norm=daily_norm_t';

% K-Means
    
K_clusters=2;    
max_iters=10;        
cluster_input=daily_norm;        
cost=1000000000;          
% 5 random initializations        
for j=1:5            
    initial_centroids = kMeansInitCentroids(cluster_input, K_clusters);             
    [temp_cost, temp_centroids, temp_idx] = runkMeans(cluster_input, initial_centroids, max_iters);        
    if cost>temp_cost % Pick the lowest cost        
        cost=temp_cost;                
        centroids=temp_centroids;                
        idx=temp_idx;            
    end    
end
%% Feature extraction
prompt=('Choose fast or sophisticated features\n0. fast 1. sofisticated (KMEANS) 2. mixed (KMEANS) 3. sofisticated (Fuzzy) 4. sofisticated (SOM) 5. Euclidean\n');
sophisticated=input(prompt);

ndays=1;
Y1D=(sum(Y2D)>ndays)';
Y=Y1D;

av_per_dif=0.8;
std_per_dif=0.6;
% Feature extraction

if sophisticated==0
    symmetric_av=0.4;
    symmetric_std=0.5;
    av_threshold=0.7;
    std_threshold=0.5;
    [X_1]=ConsumerFeatures(F_data3D, av_per_dif, std_per_dif, symmetric_av, symmetric_std); %includes ONLY 3 features
    X_2=mean(X_1,1);
    X_2=permute(X_2, [3 2 1]);

    % Obtain features related to neighbors
    K=3;
    [X_3]=NeighborFeatures(X_1, X_2, K, av_threshold, std_threshold);
    %X=X_3(:,1:(end));
    [trend_coef]=get_Trend(X_1);
    X=[X_2 X_3(:,9:end) trend_coef];
elseif sophisticated==1
    av_per_dif=0.7;
    std_per_dif=0.7;
    av_cut_per=0.1; % 0.7
    std_cut_per=0.1;% 0.7
    neigh_av_cut_per=0.3; % 0.1
    neigh_std_cut_per=0.4; % 0.1
   [X]=sophisticatedFeatures(F_data3D, av_per_dif, std_per_dif, ...
       av_cut_per, std_cut_per, neigh_av_cut_per, neigh_std_cut_per);
elseif sophisticated==2
    av_per_dif=0.7;
    std_per_dif=0.7;
    symmetric_av=0.6; 
    symmetric_std=0.6;
    neigh_av_cut_per=0.3;
    neigh_std_cut_per=0.4;
   [X]=mixedFeatures(F_data3D, av_per_dif, std_per_dif, ...
       symmetric_av, symmetric_std, neigh_av_cut_per, neigh_std_cut_per);
elseif sophisticated==3
    av_cut_per=0.1; % 0.8
    std_cut_per=0.1;% 0.6
    neigh_av_cut_per=0.1; % 0.6
    neigh_std_cut_per=0.1;
   [X]=sophFuzzyFeatures(F_data3D, av_per_dif, std_per_dif, ...
       av_cut_per, std_cut_per, neigh_av_cut_per, neigh_std_cut_per);
elseif sophisticated==4
    av_cut_per=0.1; % 0.8
    std_cut_per=0.1;% 0.6
    neigh_av_cut_per=0.1; % 0.6
    neigh_std_cut_per=0.1;
    [X]=sophSOMfeatures(F_data3D, av_per_dif, std_per_dif, ...
       av_cut_per, std_cut_per, neigh_av_cut_per, neigh_std_cut_per);
elseif sophisticated==5
    av_per_dif=0.7;
    std_per_dif=0.7;
    av_cut_per=0.3; % 0.8
    std_cut_per=0.1;% 0.6
    neigh_av_cut_per=0.1; % 0.6
    neigh_std_cut_per=0.1;
   [X]=sophisticatedFeaturesEucl(F_data3D, av_per_dif, std_per_dif, ...
       av_cut_per, std_cut_per, neigh_av_cut_per, neigh_std_cut_per);
elseif sophisticated==6
    av_per_dif=0.7;
    std_per_dif=0.7;
    av_cut_per=0.3; % 0.8
    std_cut_per=0.1;% 0.6
    neigh_av_cut_per=0.1; % 0.6
    neigh_std_cut_per=0.1;
   [X]=sophSOMEuclfeatures(F_data3D, av_per_dif, std_per_dif, ...
       av_cut_per, std_cut_per, neigh_av_cut_per, neigh_std_cut_per);
elseif sophisticated==7
    av_per_dif=0.7;
    std_per_dif=0.7;
    av_cut_per=0.3; % 0.8
    std_cut_per=0.1;% 0.6
    neigh_av_cut_per=0.1; % 0.6
    neigh_std_cut_per=0.1;
   [X]=sophFuzzyEuclfeatures(F_data3D, av_per_dif, std_per_dif, ...
       av_cut_per, std_cut_per, neigh_av_cut_per, neigh_std_cut_per);
end
fprintf('\nFraud Data and features created.\n');
%% Fast cluster prediction
cluster_pred=zeros(size(Y2D,2),1);
if sum(idx==1)>sum(idx==2) % Cluster 1 is greater (negative results)
    cluster_pred(idx==2,1)=1;
    binary_table=X(idx==2,3:end)~=0;
    binary_vector=sum(binary_table,2)<3;
    cluster_pred(idx==2,1)=not(binary_vector);
else % Cluster 2 is greater (negative results)
    cluster_pred(idx==1,1)=1;
    binary_table=X(idx==1,3:end)~=0;
    binary_vector=sum(binary_table,2)<3;
    cluster_pred(idx==1,1)=not(binary_vector);
end
%% ===  Normalization ===
prompt=('Apply normalization?\n 0 w/o norm, 1 with norm [-1,1], 2 with norm [0,1]\n');
apply_normalization=input(prompt);
if apply_normalization==0
    X_norm=X;
elseif apply_normalization==1
    [X_norm, mu, sigma] = normalizeMinus_Plus(X);
elseif apply_normalization==2
    [X_norm,~,~]=normalizeFeatures(X);
end

%% == PCA for Visualization ==
% Use PCA to project this cloud to 2D for visualization
% Subtract the mean to use PCA
prompt=('Apply PCA?\n 0 w/o PCA, 1 with PCA\n');
apply_pca=input(prompt);
if apply_pca==0
    Z=X_norm;
elseif apply_pca==1
    % PCA and project the data to 2D
    [U, S] = pca(X_norm);
    Z = projectData(X_norm, U, 2);


    plotClass(Z(:,:),Y(:));
    title('Classified examples');
end

%% Create training and testing set
% Choose from every consumer sample
% No normarlization needed here

Intr=sum(Y)/size(Y,1);% Probability of Intrusion based on Days
consumers=size(X,1);
Kfolds=5;
binary_test_table=zeros(consumers,Kfolds); % cons x Kfolds
prediction_Kfolds=zeros(consumers/Kfolds,Kfolds);
% p | ep | pr | rec | in | acc | F1 | BDR
result_table=zeros(Kfolds, 8);
Indices=crossvalind('Kfold', consumers, Kfolds);
prompt=('Choose fit of Distribution\n0. percent of instances 1. all negative examples\n');
fit_on=input(prompt);
for i=1:Kfolds
    test_idx=(Indices==i); train_idx= ~test_idx;
    binary_test_table(:,i)=test_idx; 
    fprintf('\nSegmented Training and Testing.\n');
    %% Apply anomalyDetection
    if fit_on==0
        % Estimate mu sigma2
        [mu, sigma2] = estimateGaussian(Z(train_idx,:));
        %  Training set percent of data
        p = multivariateGaussian(Z(train_idx,:), mu, sigma2);
    elseif fit_on==1
        % Estimate mu sigma2
        [mu, sigma2] = estimateGaussian(Z(Y==0,:));
        
        %  Training set  all negative examples
        p = multivariateGaussian(Z(Y==0,:), mu, sigma2);
    end
    
    %  Cross-validation set
    pval = multivariateGaussian(Z(test_idx,:), mu, sigma2);

    %  Find the best threshold
    [epsilon, F1] = selectThreshold(Y(test_idx), pval);
    prediction=(pval<epsilon);
    prediction_Kfolds(:,i)=prediction;
    
    % Operations between two predictions
    cluster_pred_test=cluster_pred(test_idx,1);
    prediction=or(cluster_pred_test, prediction);
    
    % Create confusion Matrix
    % Detection Rate is Recall, False Positive Rate is Inverse recall 
    [precision, recall, in_recall, accuracy, F1score] = confusionMatrix (Y(test_idx), prediction);
    BDR=Intr*recall/(Intr*recall+(1-Intr)*in_recall) ; % Bayesian Detection Rate for days
    result_table(i,:)=[ sum(p < epsilon) epsilon precision recall in_recall accuracy F1score BDR];
    
    rouf_id=find(prediction==1);
    roufianos=someID(rouf_id); % Keeps all the ID that contain intrusion
end
if apply_pca==1
   %plot(Z(:,1),Z(:,2), 'bx');
   %axis([0 30 0 30]);
   %xlabel('Principal Component 1');
   %ylabel('Principal Component 2');
   visualizeFit(Z, mu, sigma2, apply_normalization);
   xlabel('Principal Component 1');
   ylabel('Principal Component 2');
end


[outliers, epsilon, precision, recall, in_recall, accuracy, F1score, BDR]=MeanResults(result_table);

fprintf('\nBest epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('Based on best F1\nDR=%4.2f FPR=%4.2f on Cross Validation Set\n', recall, in_recall);
fprintf('# Outliers found: %d\n', outliers);

%% Printing Segment
fprintf('kWh Rate %4.2fper | Time Rate %4.2fper |\n',kWh_rate,time_rate);
fprintf('\nClassification for IDs\n');
fprintf('| Precision %4.2f | Recall %4.2f | Accuracy %4.2f | F1score %4.2f |\n',precision,recall,accuracy,F1score);
fprintf('| Actual Fraud %d IDs | Predicted Fraud Right %d IDs | Predicted Fraud Wrong %d IDs |\n',sum(Y(test_idx)==1),sum(prediction==1&Y(test_idx)==prediction),sum(prediction==1&Y(test_idx)~=prediction));
fprintf(' DR  FPR  BDR  Accuracy F1score\n%4.2f %4.2f %4.2f %4.2f %4.2f\n',recall,in_recall,BDR,accuracy, F1score);  