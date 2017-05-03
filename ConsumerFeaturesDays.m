function [X_1]=ConsumerFeaturesDays(data, av_per_dif, std_per_dif, symmetric_av, symmetric_std)
% gets 3 features for consumers
month_av_data=zeros(12,30,size(data,3));
month_std_data=zeros(12,30,size(data,3));
temp_av=zeros(30,12,size(data,3));
temp_std=zeros(30,12,size(data,3));
m_av=zeros(12,1,size(data,3));
m_std=zeros(12,1,size(data,3));
max_av_difference3D=zeros(size(data,1),1,size(data,3));
max_std_difference3D=zeros(size(data,1),1,size(data,3));
symmetric_av_difference3D=zeros(size(data,1),1,size(data,3));
symmetric_std_difference3D=zeros(size(data,1),1,size(data,3));

average3D=mean(data,2); % 1st Feature average consumption
standard_dev3D=std(data,0,2); % 2nd Feature std of consumption

for i=1:size(data,3)
    % 3rd Feature std difference
    temp_av(:,:,i)=reshape(average3D(1:360,1,i), [30,12]); % Discard 5 last days
    month_av_data(:,:,i)=temp_av(:,:,i)';
    m_av(:,:,i)=mean(month_av_data(:,:,i),2);
    
    max_av=0;
    for j=2:(size(m_av,1)-1)
    %for j=4:(size(m_av,1)-3)
        if ((mean(m_av(1:j,1,i))-mean(m_av(j+1:end,1,i)))/mean(m_av(1:j,1,i))> av_per_dif) % at least some per difference
           dif_av=mean(m_av(1:j,1,i))-mean(m_av(j+1:end,1,i));
           if max_av<dif_av
            max_av=dif_av;
            max_av_idx=j;
           end
        end
    end
    if max_av~=0
        max_av_difference3D(:,1,i)=...
            [max_av_difference3D(1:(max_av_idx-1),1,i); repmat(max_av, size(average3D(max_av_idx:end,:,:),1),1)];
    end
    
    % 4th Feature std difference
    temp_std(:,:,i)=reshape(standard_dev3D(1:360,1,i), [30,12]); % Discard 5 last days
    month_std_data(:,:,i)=temp_std(:,:,i)';
    m_std(:,:,i)=mean(month_std_data(:,:,i),2);
    
    max_std=0;
    for j=2:(size(m_std,1)-1)
    %for j=4:(size(m_av,1)-3)
        if ((mean(m_std(1:j,1,i))-mean(m_std(j+1:end,1,i)))/mean(m_std(1:j,1,i))> std_per_dif) % at least some per difference
           dif_std=mean(m_std(1:j,1,i))-mean(m_std(j+1:end,1,i));
           if max_std<dif_std
            max_std=dif_std;
            max_std_idx=j;
           end
        end
    end
    if max_std~=0
        max_std_difference3D(:,1,i)=...
            [max_std_difference3D(1:(max_std_idx-1),1,i); repmat(max_std, size(standard_dev3D(max_std_idx:end,:,:),1),1)];
    end
    
    % 5th feature
    max_sym_av=0;
    temp_sym_av_table=reshape(average3D(1:360,1,i), [90, 4]);
    temp_sym_av_table_t=temp_sym_av_table';
    temp_sym_av_vector=mean(temp_sym_av_table_t,2);
    for j=1:size(temp_sym_av_vector)/2
        if (temp_sym_av_vector(j,1)-temp_sym_av_vector(end+1-j,1))/temp_sym_av_vector(j,1)>symmetric_av
            if max_sym_av<(temp_sym_av_vector(j,1)-temp_sym_av_vector(end+1-j,1))/temp_sym_av_vector(j,1)
                max_sym_av=temp_sym_av_vector(j,1)-temp_sym_av_vector(end+1-j,1);
                max_sym_av_idx=j;
            end
        end
    end
    if max_sym_av~=0
        if max_sym_av_idx<2
        symmetric_av_difference3D(:,1,i)=...
            [symmetric_av_difference3D(1:270,1,i); repmat(max_sym_av,size(average3D(271:end,:,:),1),1)];
        else
            symmetric_av_difference3D(:,1,i)=...
            [symmetric_av_difference3D(1:(180),1,i); ...
            repmat(max_sym_av,size(average3D(181:270,:,:),1),1); symmetric_av_difference3D(271:end,1,i)];
        end
    end
    
    %6th feature
    max_sym_std=0;
    temp_sym_std_table=reshape(standard_dev3D(1:360,1,i), [90, 4]);
    temp_sym_std_table_t=temp_sym_std_table';
    temp_sym_std_vector=mean(temp_sym_std_table_t,2);
    for j=1:size(temp_sym_std_vector)/2
        if (temp_sym_std_vector(j,1)-temp_sym_std_vector(end+1-j,1))/temp_sym_std_vector(j,1)>symmetric_std
            if max_sym_std<(temp_sym_std_vector(j,1)-temp_sym_std_vector(end+1-j,1))/temp_sym_std_vector(j,1)
                max_sym_std=temp_sym_std_vector(j,1)-temp_sym_std_vector(end+1-j,1);
                max_sym_std_idx=j;
            end
        end
    end
    if max_sym_std~=0
        if max_sym_std_idx<2
        symmetric_std_difference3D(:,1,i)=...
            [symmetric_std_difference3D(1:(270),1,i); repmat(max_sym_std,size(average3D(271:end,:,:),1),1)];
        else
            symmetric_std_difference3D(:,1,i)=...
            [symmetric_std_difference3D(1:(180),1,i); ...
            repmat(max_sym_std,size(average3D(181:270,:,:),1),1); symmetric_std_difference3D(271:end,1,i)];
        end
    end
end
X_1=[average3D standard_dev3D max_av_difference3D max_std_difference3D symmetric_av_difference3D symmetric_std_difference3D];