%% Apply K-Means horizontaly and verticaly 
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

daily_consumption3D=sum(F_data3D,2);
daily_consumption=permute(daily_consumption3D,[3 1 2]); % cons vs days
average_consumption=mean(daily_consumption,2);

%% K-Means will cluster consumers based on yearly average consumptions
random_init=5; % random initializations
max_iters=10;
K=4;

cost=1000000000;
% 5 random initializations
for j=1:random_init
    initial_centroids = kMeansInitCentroids(daily_consumption, K); 
    [temp_cost, temp_centroids, temp_idx] = runkMeans(daily_consumption, initial_centroids, max_iters);
    if cost>temp_cost % Pick the lowest cost
        cost=temp_cost;
        centroids=temp_centroids;
        idx=temp_idx;
    end
end 
%% Transpose daily consumption
daily_consumption_transposed=daily_consumption';
K_t=7;
idx_t=zeros(size(daily_consumption_transposed,1),K);
cost_t=zeros(1,K);
%centroids_t=zeros(K_t*K,size(daily_consumption_transposed,2));
%centroids_t=zeros(K_t,size(daily_consumption_transposed,2),K);
for i=1:K
    some_consumption_transposed=daily_consumption_transposed(:,idx==i);
    for j=1:random_init
        initial_centroids = kMeansInitCentroids(some_consumption_transposed, K_t); 
        [temp_cost, temp_centroids, temp_idx] = runkMeans(some_consumption_transposed, initial_centroids, max_iters);
        if cost>temp_cost % Pick the lowest cost
            cost_t(1,i)=temp_cost;
            %centroids_t((K_t*(i-1)+1):(K_t*i),:)=temp_centroids;
            %centroids_t(:,:,i)=temp_centroids;
            idx_t(:,i)=temp_idx;
        end
    end   
end