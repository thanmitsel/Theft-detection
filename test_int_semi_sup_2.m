%% Semi supervised algorithm with a twisted Anomaly Detection
  %  form cluster based on rule based system
    % pick some z vector 
    z=4500;
    r_cons=randi(size(hh,1),z,1);
    somehh=hh(r_cons,:);
    someID=ID(r_cons,:);

    % Convertions
    [h, H]=convertHours3D(somehh);

    %% Fraud Initialization
    % Create Fraud data
    intensity_vector=0:0.01:1;
for int_idx=1:length(intensity_vector)
    F_data3D=H;
    Y2D=zeros(size(H,1),size(H,3));
    one_H=zeros(size(H(:,:,1)));

% fraud_rate=0.05; % Percentage of consumers who fraud
fraud_rate=0.1; % Percentage of consumers who fraud
[normal_idx, fraud_idx] = crossvalind('HoldOut', size(H,3), fraud_rate); % Keep in mind crossval floors the rate
thiefs=find(fraud_idx==1);
for i=1:size(thiefs,1)    
    intensity=1-intensity_vector(1,int_idx); % beta distribution
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
apply_normalization=1;   
if apply_normalization==0    
    daily_norm_t=daily_consumption_t;   
elseif apply_normalization==1    
    [daily_norm_t, mu, sigma] = normalizeMinus_Plus(daily_consumption_t);    
elseif apply_normalization==2    
    [daily_norm_t,~,~]=normalizeFeatures(daily_consumption_t);    
end

daily_norm=daily_norm_t';


form_cluster=0;
     
K_clusters=2;    
max_iters=10;            
cluster_input=daily_norm;
if form_cluster==0
    % K-Means   
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
elseif form_cluster==1
    divider=5;
    smaller_cluster_input=zeros(size(daily_consumption,1),365/divider);
    for w=1:365/divider
        smaller_cluster_input(:,w)=sum(daily_consumption(:,(divider*w-4):divider*w),2);
    end
    % Fuzzy C-Means
    options=[3 NaN NaN 0];
    [~,U,objfnc] = fcm(cluster_input,K_clusters, options);
    maxU=max(U);
    idx=zeros(size(daily_consumption,1),1);
    for i=1:K_clusters
        temp_idx=U(i,:)==maxU;
        idx(temp_idx,1)=i;
    end
elseif form_cluster==2
    net=selforgmap([K_clusters/2 K_clusters/2]);
    [net, tr]=train(net, cluster_input');
    outputs=net(cluster_input'); % K x consumers

    idx=zeros(size(daily_consumption,1),1);

    for i=1:K_clusters
        idx(logical(outputs(:,i)),1)=i;
    end
    idx=idx';

end
%% Feature extraction
sophisticated=5;

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
%% == PCA for Visualization ==
% Use PCA to project this cloud to 2D for visualization
% Subtract the mean to use PCA
apply_pca=0;
if apply_pca==0
    X=X;
elseif apply_pca==1
    % PCA and project the data to 2D
    [U, S] = pca(X);
    X = projectData(X, U, 2);


    plotClass(X(:,:),Y(:));
end
%% Segment Consumers based on Form    
X_part1=X(idx==1, :);    
Y_part1=Y(idx==1);    
X_part2=X(idx==2, :);   
Y_part2=Y(idx==2);    
if size(X_part1,1)>size(X_part2,1)      
    X_big=X_part1;
    Y_big=Y_part1;
    X_small=X_part2;
    Y_small=Y_part2;  
else 
    X_big=X_part2;
    Y_big=Y_part2;
    X_small=X_part1;
    Y_small=Y_part1; 
end

%% Prediction
pred_big=zeros(size(Y_big));
%binary_table=X_big(:,3:end)~=0;
%pred_big=sum(binary_table,2)>=3;

pred_small=ones(size(Y_small));
binary_table=X_small(:,3:end)~=0; % how many features
binary_vector=sum(binary_table,2)<3; % less than 3 features consider negative
%%%binary_vector=sum(X_small(:,3:end),2)==0;
pred_small(binary_vector)=0;
%% Anomaly Detection with random test train
%First segmentation for training
[train_idx, remain_idx]=crossvalind('Holdout', size(X_big,1),0.3);
X_train=X_big(train_idx,:);
[X_train, minval, maxval]=normalizeFeatures(X_train);
X_remain=X_big(remain_idx,:);
Y_remain=Y_big(remain_idx,:);
pred_big_remain=pred_big(remain_idx,:);

%Second segmentation for test and cross validation
pred_shuffled=[pred_big_remain; pred_small];
X_shuffled=[X_remain; X_small];
[X_shuffled]=normalizeTest(X_shuffled, minval, maxval);
Y_shuffled=[Y_remain; Y_small];

[test_idx, val_idx]=crossvalind('Holdout', size(X_shuffled,1), 0.5);
X_test=X_shuffled(test_idx,:);
Y_test=Y_shuffled(test_idx,:);
unsup_pred_test=pred_shuffled(test_idx,:);
X_val=X_shuffled(val_idx,:);
Y_val=Y_shuffled(val_idx,:);
unsup_pred_val=pred_shuffled(val_idx,:);

[train_idx, remain_idx]=crossvalind('Holdout', size(X_big,1),0.5);

%  Estimate mu and sigma2
[mu, sigma2] = estimateGaussian(X_train);

%  Returns the density of the multivariate normal at each data point (row) 
%  of X val
p = multivariateGaussian(X_val, mu, sigma2);


ptest = multivariateGaussian(X_test, mu, sigma2);

[epsilon, F1] = selectThreshold(Y_test, ptest);

%  Find the outliers in the training set
semi_pred_val = (p < epsilon);

% Logic operations

logic_operation=0;
if logic_operation==0
    prediction=and(semi_pred_val, unsup_pred_val);
elseif logic_operation==1
    prediction=or(semi_pred_val, unsup_pred_val);
end

[precision, recall, in_recall, accuracy, F1score] = confusionMatrix (Y_val, prediction);
BDR=fraud_rate*recall/(fraud_rate*recall+(1-fraud_rate)*in_recall) ; % Bayesian Detection Rate for days
fprintf(' DR  FPR Accuracy F1score BDR \n%4.2f & %4.2f  & %4.2f & %4.2f & %4.2f\n',recall,in_recall,accuracy, F1score,BDR);
result_table(int_idx,:)=[intensity_vector(int_idx) recall in_recall  accuracy BDR F1score];
end
figure;
plot(result_table(:,1),result_table(:,2))
xlabel('Ένταση');
ylabel('DR');
hold on;
figure;
plot(result_table(:,1),result_table(:,3))
xlabel('Ένταση');
ylabel('FPR')
hold off;