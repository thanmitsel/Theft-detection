function [features]=sophisticatedFeatures(data, av_per_dif, std_per_dif, av_cut_per, std_cut_per, neigh_av_cut_per, neigh_std_cut_per )

max_av_difference3D=zeros(size(data,1),1,size(data,3));
max_std_difference3D=zeros(size(data,1),1,size(data,3));
av_cut_dif=zeros(size(data,3),1);
std_cut_dif=zeros(size(data,3),1);
neigh_av_cut_dif=zeros(size(data,3),1);
neigh_std_cut_dif=zeros(size(data,3),1);

average3D=mean(data,2); % 1st Feature average consumption
standard_dev3D=std(data,0,2); % 2nd Feature std of consumption

for i=1:size(data,3)
    % 3rd Feature std difference
    % Moving average of average consumption of every month
    temp_av=reshape(average3D(1:360,1,i), [30,12]); % Discard 5 last days
    month_av_data=temp_av';
    m_av=mean(month_av_data,2);
    
    max_av=0;
    for j=2:(size(m_av,1)-1)
    %for j=4:(size(m_av,1)-3)
        if ((mean(m_av(1:j,1))-mean(m_av(j+1:end,1)))/mean(m_av(1:j,1))> av_per_dif) % at least some per difference
           dif_av=mean(m_av(1:j,1))-mean(m_av(j+1:end,1));
           if max_av<dif_av
            max_av=dif_av;
           end
        end
    end
    max_av_difference3D(:,1,i)=repmat(max_av, size(average3D,1),1);
    
    % 4th Feature std difference
    % Moving average of std consumption of every month
    temp_std=reshape(standard_dev3D(1:360,1,i), [30,12]); % Discard 5 last days
    month_std_data=temp_std';
    m_std=mean(month_std_data,2);
    
    max_std=0;
    for j=2:(size(m_std,1)-1)
    %for j=4:(size(m_av,1)-3)
        if ((mean(m_std(1:j,1))-mean(m_std(j+1:end,1)))/mean(m_std(1:j,1))> std_per_dif) % at least some per difference
           dif_std=mean(m_std(1:j,1))-mean(m_std(j+1:end,1));
           if max_std<dif_std
            max_std=dif_std;
           end
        end
    end
    max_std_difference3D(:,1,i)=repmat(max_std, size(standard_dev3D,1),1);
end

basic_features3D=[average3D standard_dev3D max_av_difference3D max_std_difference3D];
temp_basic_feat=mean(basic_features3D,1);
basic_features=permute(temp_basic_feat, [3 2 1]); % cons x 4 features

% Sophisticated part
daily_consumption3D=sum(data,2);
daily_consumption=permute(daily_consumption3D,[3 1 2]); % cons x 365 days
average_consumption=mean(daily_consumption,2); % cons x 1 average of year
std_consumption=std(daily_consumption,0,2);
daily_std3D=basic_features3D(:,2,:);
daily_std=permute(daily_std3D,[3 1 2]);

K=4;
max_iters=10;
cluster_input=[average_consumption std_consumption];
% Fuzzy C-Means
[centers,U] = fcm(cluster_input,K);
maxU=max(U);
idx=zeros(size(daily_consumption,1),1);
for i=1:K
    temp_idx=U(i,:)==maxU;
    idx(temp_idx,1)=i;
end



% Create daily consumptions based on clusters
cluster_consumption=zeros(K,size(daily_consumption,2));
cluster_std=zeros(K,size(daily_std,2));
sum_clusters=zeros(K,1);

for i=1:K
    cluster_consumption(i,:)=mean(daily_consumption(idx==i,:),1); % K x 365 days 
    cluster_std(i,:)=mean(daily_std(idx==i,:),1);
    sum_clusters(i)=sum(idx==i);
end

% Fit quadratic trend
% Fit the second degree polynomial to the observed series.
tH=zeros(size(cluster_consumption,2),K);
T = size(cluster_consumption,2);
t = (1:T)';
X = [ones(T,1) t t.^2];

for i=1:K
    y=cluster_consumption(i,:)';
    b = X\y;
    tH(:,i) = X*b;  % 365 days x K
end
% Indicates when the minimum consumption happened
[min_tH, min_tH_idx]=min(tH,[],1); % 1 x K

for i=1:size(daily_consumption,1) % loop consumers
    cluster=idx(i);
    cut=min_tH_idx(1, cluster);
    after_cut=daily_consumption(i,cut:end);
    after_cut_std=daily_std(i,cut:end);
    after_cut_size=size(after_cut,2);
    % Need to check if after cut is larger than before cut
    if (cut-1-after_cut_size)<1
        before_cut=daily_consumption(i,1:(cut-1));
        before_cut_std=daily_std(i,1:(cut-1));
    else
        before_cut=daily_consumption(i,(cut-1-after_cut_size):(cut-1));
        before_cut_std=daily_std(i,(cut-1-after_cut_size):(cut-1));
    end
    % 5th feature
    if (mean(before_cut)-mean(after_cut))/mean(before_cut) > av_cut_per
        av_cut_dif(i)=mean(before_cut)-mean(after_cut);   
    end
    % 6th feature
    if (mean(before_cut_std)-mean(after_cut_std))/mean(before_cut_std) > std_cut_per
        std_cut_dif(i)=mean(before_cut_std)-mean(after_cut_std);   
    end
    neighbor_cut=cluster_consumption(cluster, cut:end);
    % 7th feature
    if (mean(neighbor_cut)-mean(after_cut))/mean(neighbor_cut) > neigh_av_cut_per
        neigh_av_cut_dif(i)=mean(neighbor_cut)-mean(after_cut);
    end
    neighbor_cut_std=cluster_std(cluster, cut:end);
    % 8th feature
    if (mean(neighbor_cut_std)-mean(after_cut_std))/mean(neighbor_cut_std) > neigh_std_cut_per
        neigh_std_cut_dif(i)=mean(neighbor_cut_std)-mean(after_cut_std);
    end  
end


features=[basic_features av_cut_dif std_cut_dif neigh_av_cut_dif neigh_std_cut_dif];
