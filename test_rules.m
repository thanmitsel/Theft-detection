%% Fully unsupervised algorithm with 
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

prompt=('Form Clustering.\n 0. K-Means 1. Fuzzy\n');
form_cluster=input(prompt);
     
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
end
%% Feature extraction
prompt=('Choose fast or sophisticated features\n0. fast 1. sofisticated (KMEANS) 2. mixed (KMEANS) 3. sofisticated (Fuzzy) 4. sofisticated (SOM) 5. Euclidian\n');
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
end
fprintf('\nFraud Data and features created.\n');
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
%pred_big=zeros(size(Y_big));
binary_table=X_big(:,3:end)~=0;
pred_big=sum(binary_table,2)>=3;


pred_small=ones(size(Y_small));
binary_table=X_small(:,3:end)~=0; % how many features
binary_vector=sum(binary_table,2)<3; % less than 3 features consider negative
%%%binary_vector=sum(X_small(:,3:end),2)==0;
pred_small(binary_vector)=0;
%% Results
tp=sum((pred_big==Y_big)&(pred_big==1)); % Predicted yes, actual yes
tn=sum((pred_big==Y_big)&(pred_big==0)); % Predicted no , actual no 
fp=sum((pred_big==1)&(Y_big==0)); % Predicted yes, actual no
fn=sum((pred_big==0)&(Y_big==1)); % Predicted no, actual yes 

tp=tp+sum((pred_small==Y_small)&(pred_small==1));
tn=tn+sum((pred_small==Y_small)&(pred_small==0));
fp=fp+sum((pred_small==1)&(Y_small==0));
fn=fn+sum((pred_small==0)&(Y_small==1));

precision=tp/(tp+fp)*100; 
recall=tp/(tp+fn)*100; % True Positive Rate-Detection Rate
in_recall=fp/(fp+tn)*100; % False Positive Rate
accuracy=((tp+tn)/(tp+tn+fp+fn))*100;
F1score=2*precision*recall/(precision+recall); % Already a percent value    
BDR=fraud_rate*recall/(fraud_rate*recall+(1-fraud_rate)*in_recall) ; % Bayesian Detection Rate for days

%% Printing Segment
fprintf('kWh Rate %4.2fper | Time Rate %4.2fper |\n',kWh_rate,time_rate);
fprintf('\nClassification for IDs\n');
fprintf('| Precision %4.2f | Recall %4.2f | Accuracy %4.2f | F1score %4.2f |\n',precision,recall,accuracy,F1score);
fprintf('| Actual Fraud %d IDs | Predicted Fraud Right %d IDs | Predicted Fraud Wrong %d IDs |\n',sum(Y==1), tp, fp);
fprintf(' DR  FPR  BDR  Accuracy F1score\n%4.2f %4.2f %4.2f %4.2f %4.2f\n',recall,in_recall,BDR,accuracy, F1score);
%% Apply Confidence on results
if form_cluster==1
looking_for=100;
confident_idx=zeros(looking_for,1);
maxU_indexed=[1:z ; maxU]';
sort_max=sortrows(maxU_indexed, 2);
confident=sort_max((end-looking_for+1):end,:);
confident_idx(:,1)=confident(:,1);
confident_prediction=ones(looking_for,1);

% This segment doesn't really help get lower FP
%binary_table_conf=X(confident_idx,3:end)~=0; % how many features
%binary_vector_conf=sum(binary_table_conf,2)<2; % less than 3 features consider negative
%%binary_vector=sum(X_small(:,3:end),2)==0;
%confident_prediction(binary_vector_conf)=0;

right=sum(Y(confident_idx,1)==confident_prediction);
wrong=sum(Y(confident_idx,1)~=confident_prediction);

tp=sum((confident_prediction==Y(confident_idx,1))&(confident_prediction==1)); % Predicted yes, actual yes
tn=sum((confident_prediction==Y(confident_idx,1))&(confident_prediction==0)); % Predicted no , actual no 
fp=sum((confident_prediction==1)&(Y(confident_idx,1)==0)); % Predicted yes, actual no
fn=sum((confident_prediction==0)&(Y(confident_idx,1)==1)); % Predicted no, actual yes

fprintf('\nLooking for %d, right predicions %d, wrong predictions %d\n', looking_for, right, wrong);
fprintf('| TP=%d | TN=%d | FP=%d | FN=%d |\n', tp, tn, fp, fn);

for i=1:fraud_rate*z
    looking_for=i;
    confident_idx=zeros(looking_for,1);
    maxU_indexed=[1:z ; maxU]';
    sort_max=sortrows(maxU_indexed, 2);
    confident=sort_max((end-looking_for+1):end,:);
    confident_idx(:,1)=confident(:,1);
    confident_prediction=ones(looking_for,1);

    % This segment doesn't really help get lower FP
    %binary_table_conf=X(confident_idx,3:end)~=0; % how many features
    %binary_vector_conf=sum(binary_table_conf,2)<2; % less than 3 features consider negative
    %%binary_vector=sum(X_small(:,3:end),2)==0;
    %confident_prediction(binary_vector_conf)=0;

    right(looking_for,1)=sum(Y(confident_idx,1)==confident_prediction);
    wrong(i,1)=sum(Y(confident_idx,1)~=confident_prediction);
end
plot(1:fraud_rate*z,wrong(:,1));
end