function [X_3]=NeighborFeatures(X_1, X_2, K, av_threshold, std_threshold)
% K Means to Obtain values of same type of consumers
neighborhood_av=zeros(size(X_1,1),K);
neighborhood_std=zeros(size(X_1,1),K);
d_neighborhood_av=zeros(5,K);
d_neighborhood_std=zeros(5,K);
max_dif_av=zeros(size(X_2,1),1);
max_dif_av3D=zeros(size(X_1,1),1,size(X_1,3));
max_dif_std=zeros(size(X_2,1),1);
max_dif_std3D=zeros(size(X_1,1),1,size(X_1,3));
max_iters = 10;


cost=1000000000;
for j=1:1
    initial_centroids = kMeansInitCentroids(X_2(:,1:2), K); 
    [temp_cost, temp_centroids, temp_idx] = runkMeans(X_2(:,1:2), initial_centroids, max_iters);
    if cost>temp_cost % Pick the lowest cost
        cost=temp_cost;
        centroids=temp_centroids;
        idx=temp_idx;
    end
end 

% Loop over Days*Consumers
for i=1:size(X_2,1) 
    cluster=idx(i);
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
    temp_av_matrix=reshape(X_1(:,1,i), [73, 5]);
    temp_av_t=temp_av_matrix';
    divided_av=mean(temp_av_t,2);
    if max((d_neighborhood_av(:,cluster)-divided_av)./d_neighborhood_av(:,cluster))>av_threshold   
        [max_dif_av(i,1), max_dif_av_idx]=max(d_neighborhood_av(:,cluster)-divided_av);
        if max_dif_av_idx>4
            max_dif_av3D(:,1,i)=...
                [max_dif_av3D(1:(292),1,i); repmat(max_dif_av(i,1),size(X_1(293:end,:,:),1),1)];    
        elseif max_dif_av_idx<2
            max_dif_av3D(:,1,i)=...
                [repmat(max_dif_av(i,1),size(X_1(1:73,:,:),1),1); max_dif_av3D(74:end,1,i) ];
        else
            max_dif_av3D(:,1,i)=...
                [max_dif_av3D(1:((max_dif_av_idx-1)*73),1,i);...
                repmat(max_dif_av(i,1),73,1); max_dif_av3D(((max_dif_av_idx*73+1):end),1,i)];
        end
    end
    temp_std_matrix=reshape(X_1(:,2,i), [73, 5]);
    temp_std_t=temp_std_matrix';
    divided_std=mean(temp_std_t,2);
    if max((d_neighborhood_std(:,cluster)-divided_std)./d_neighborhood_std(:,cluster))>std_threshold    
        [max_dif_std(i,1), max_dif_std_idx]=max(d_neighborhood_std(:,cluster)-divided_std);
        if max_dif_std_idx>4
            max_dif_std3D(:,1,i)=...
                [max_dif_std3D(1:(292),1,i); repmat(max_dif_std(i,1),size(X_1(293:end,:,:),1),1)];
        elseif max_dif_std_idx<2
            max_dif_std3D(:,1,i)=...
                [repmat(max_dif_std(i,1),size(X_1(1:73,:,:),1),1); max_dif_std3D(74:end,1,i) ];
        else
            max_dif_std3D(:,1,i)=...
                [max_dif_std3D(1:((max_dif_std_idx-1)*73),1,i);...
                repmat(max_dif_std(i,1),73,1); max_dif_std3D(((max_dif_std_idx*73+1):end),1,i)];
        end
    end
end
Y=zeros(size(X_2,1),1);
[max_dif_av2D,Y]=unrollto2D(max_dif_av3D,Y);
[max_dif_std2D,Y]=unrollto2D(max_dif_std3D,Y);

X_3=[max_dif_av2D max_dif_std2D];