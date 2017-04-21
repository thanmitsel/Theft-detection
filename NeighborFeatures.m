function [X_3]=NeighborFeatures(X_2)
% K Means to Obtain values of same type of consumers
average_dif=zeros(size(X_2,1),1);
std_dif=zeros(size(X_2,1),1);
max_iters = 10;
K=3;   

% 5 random initialazations
cost=1000000000;
for j=1:5
    initial_centroids = kMeansInitCentroids(X_2(:,1:2), K); 
    [temp_cost, temp_centroids, temp_idx] = runkMeans(X_2(:,1:2), initial_centroids, max_iters);
    if cost>temp_cost % Pick the lowest cost
        cost=temp_cost;
        centroids=temp_centroids;
        idx=temp_idx;
    end
end 

for i=1:size(X_2,1)
    cluster=idx(i);
    average_dif(i)=centroids(cluster,1)-X_2(i,1);
    std_dif(i)=centroids(cluster,2)-X_2(i,2);
end

X_3=[X_2 average_dif std_dif];