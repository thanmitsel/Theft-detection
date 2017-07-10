%% Full unsupervised algorithm with 
  %  form cluster
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
fraud_rate=0.3; % Percentage of consumers who fraud
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
prompt=('Pick Form Clustering\n 0. NOP 1. Large Cluster 2. Small Cluster');
pickForm=input(prompt);

if pickForm~=0
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
    X_part1=F_data3D(:, :, idx==1);
    Y_part1=Y2D(:, idx==1);
    X_part2=F_data3D(:, :, idx==2);
    Y_part2=Y2D(:, idx==2);
    if pickForm==1
       if size(X_part1,3)>size(X_part2,3)
           F_data3D=X_part1;
           Y2D=Y_part1;
       else
           F_data3D=X_part2;
           Y2D=Y_part2;
       end
    elseif pickForm==2
        if size(X_part1,3)>size(X_part2,3)
           F_data3D=X_part2;
           Y2D=Y_part2;
       else
           F_data3D=X_part1;
           Y2D=Y_part1;
       end
    end
end

% check if even so K-Folds won't break
modFlag=mod(size(F_data3D,3),2);
if modFlag~=0
    F_data3D=F_data3D(:, :, 1:(end-modFlag));
    Y2D=Y2D(:,1:(end-modFlag));
end

%% Feature extraction
prompt=('Choose fast or sophisticated features\n0. fast 1. sofisticated (KMEANS) 2. mixed (KMEANS) 3. sofisticated (Fuzzy) 4. sofisticated (SOM)\n');
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