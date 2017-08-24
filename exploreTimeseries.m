%% Preprocesing, Formating Data for many consumers
% We try to explore consumption based on form and energy levels

% pick some z vector 
z=4500;
r_cons=randi(size(hh,1),z,1);
somehh=hh(r_cons,:);
HH=zeros(365,48,size(somehh,1));
for j=1:size(somehh,1)
    HH(:,:,j)=vec2mat(somehh(j,:),48);
end
someID=ID(r_cons,:);

% Convertions
[h, H]=convertHours3D(somehh);
[daily_consumption]=convertDays3D(somehh);
%% Form Clustering
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
    
K_clusters=6;    
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
%% Print size of clusters
fprintf('|Cluster  | Members    |\n');
for i=1:K_clusters
    fprintf('|  %d   |   %d  |\n',i, sum(idx==i));
end
%% Plot clusters
T=size(daily_consumption,2);
figure; hold on;
for i=1:K_clusters
    ax1=subplot(K_clusters/2,2,i);
    daily_cluster=daily_consumption(idx==i,:);
    plot(daily_cluster(2,:));
    h1 = gca;
    h1.XLim = [0,T];
    str=sprintf('Παράδειγμα Κατανάλωσης Συστάδας %d',i);
    %str=sprintf('Example of Cluster %d',i);
    title(str);
    ylabel 'Κατανάλωση (kWh)';
    %ylabel 'Consumption (kWh)';
    xlabel 'Μέρες';
    %xlabel 'Days';
end
hold off;

%% Consumption clusteringdaily_consumption_t=daily_consumption';    
% K-Means
    
K_clusters=6;    
max_iters=10;        
cluster_input=daily_consumption;        
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
%% Print size of clusters
fprintf('|Cluster  | Members    |\n');
for i=1:K_clusters
    fprintf('  %d   &   %d  \\ \n',i, sum(idx==i));
end
%% Plot clusters
T=size(daily_consumption,2);
figure; hold on;
for i=1:K_clusters
    ax1=subplot(K_clusters/2,2,i);
    daily_cluster=daily_consumption(idx==i,:);
    plot(daily_cluster(2,:));
    h1 = gca;
    h1.XLim = [0,T];
    str=sprintf('Παράδειγμα Κατανάλωσης Συστάδας %d',i);
    %str=sprintf('Example of Cluster %d',i);
    title(str);
    ylabel 'Κατανάλωση (kWh)';
    %ylabel 'Consumption (kWh)';
    xlabel 'Μέρες';
    %xlabel 'Days';
end
hold off;
%% Print histogram with Mean of all consumptions
meandayscons=mean(daily_consumption,1);
meancustomer=mean(daily_consumption,2);
stddayscons=std(daily_consumption,0,1);
stdcustomer=std(daily_consumption,0,2);
nbins=100;
figure; hold on;
histogram(meandayscons,nbins);
ylabel 'Μέρες';
xlabel 'Kατανάλωση Ημέρας (kWh)'
figure;
histogram(meancustomer,nbins);
ylabel 'Καταναλωτές';
xlabel 'Ετήσια κατανάλωση (kWh)'
figure;
histogram(stddayscons,nbins);
ylabel 'Μέρες';
xlabel 'Kατανάλωση Ημερήσια (kWh)'
figure;
histogram(stdcustomer,nbins);
ylabel 'Καταναλωτές';
xlabel 'Ετήσια τυπική απόκλιση (kWh)'
hold off;