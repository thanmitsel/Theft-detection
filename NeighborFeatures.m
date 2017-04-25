function [X_3]=NeighborFeatures(X_1, X_2, K, av_threshold, std_threshold)
% K Means to Obtain values of same type of consumers
temp_av_matrix=zeros(73,5,size(X_1,3));
temp_std_matrix=zeros(73,5,size(X_1,3));
temp_av_t=zeros(5,73,size(X_1,3));
temp_std_t=zeros(5,73,size(X_1,3));

divided_av=zeros(5,1,size(X_1,3));
divided_std=zeros(5,1,size(X_1,3));
neighborhood_av=zeros(size(X_1,1),K);
neighborhood_std=zeros(size(X_1,1),K);
d_neighborhood_av=zeros(5,K);
d_neighborhood_std=zeros(5,K);
average_dif=zeros(size(X_2,1),1);
std_dif=zeros(size(X_2,1),1);
max_dif_av=zeros(size(X_2,1),1);
max_dif_std=zeros(size(X_2,1),1);
max_iters = 10;

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

% Loop over consumers
for i=1:size(X_2,1) 
    cluster=idx(i);
    average_dif(i)=centroids(cluster,1)-X_2(i,1);
    std_dif(i)=centroids(cluster,2)-X_2(i,2);
    neighborhood_av(:,cluster)=neighborhood_av(:,cluster)+X_1(:,1,i);
    neighborhood_std(:,cluster)=neighborhood_std(:,cluster)+X_1(:,2,i);
end

% Get means
for i=1:K
    neighborhood_av(:,i)=neighborhood_av(:,i)/sum(idx==i);
    temp_neigh_av_matrix=reshape(neighborhood_av(:,i), [73, 5]);
    temp_neigh_av_t=temp_neigh_av_matrix';
    d_neighborhood_av(:,i)=mean(temp_neigh_av_t,2);
    neighborhood_std(:,i)=neighborhood_std(:,i)/sum(idx==i);
    temp_neigh_std_matrix=reshape(neighborhood_std(:,i), [73, 5]);
    temp_neigh_std_t=temp_neigh_std_matrix(:,i)';
    d_neighborhood_std(:,i)=mean(temp_neigh_std_t,2);
end


for i=1:size(X_1,3)
    cluster=idx(i);
    temp_av_matrix(:,:,i)=reshape(X_1(:,1,i), [73, 5]);
    temp_av_t(:,:,i)=temp_av_matrix(:,:,i)';
    divided_av(:,:,i)=mean(temp_av_t(:,:,i),2);
    if max((d_neighborhood_av(:,cluster)-divided_av(:,:,i))./d_neighborhood_av(:,cluster))>av_threshold    
        max_dif_av(i,1)=max(d_neighborhood_av(:,cluster)-divided_av(:,:,i));
    end
    temp_std_matrix(:,:,i)=reshape(X_1(:,2,i), [73, 5]);
    temp_std_t(:,:,i)=temp_std_matrix(:,:,i)';
    divided_std(:,:,i)=mean(temp_std_t(:,:,i),2);
    if max((d_neighborhood_std(:,cluster)-divided_std(:,:,i))./d_neighborhood_std(:,cluster))>std_threshold    
        max_dif_std(i,1)=max(d_neighborhood_std(:,cluster)-divided_std(:,:,i));
    end
end
X_3=[X_2 average_dif std_dif max_dif_av max_dif_std];