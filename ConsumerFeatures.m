function [X_plot]=ConsumerFeatures(data, av_per_dif, std_per_dif)
% gets 3 features for consumers
month_av_data=zeros(12,30,size(data,3));
month_std_data=zeros(12,30,size(data,3));
temp_av=zeros(30,12,size(data,3));
temp_std=zeros(30,12,size(data,3));
m_av=zeros(12,1,size(data,3));
m_std=zeros(12,1,size(data,3));
max_av_difference3D=zeros(size(data,1),1,size(data,3));
max_std_difference3D=zeros(size(data,1),1,size(data,3));

average3D=mean(data,2); % 1st Feature average consumption
standard_dev3D=std(data,0,2); % 2nd Feature std of consumption

for i=1:size(data,3)
    % 3rd Feature std difference
    temp_av(:,:,i)=reshape(average3D(1:360,1,i), [30,12]); % Discard 5 last days
    month_av_data(:,:,i)=temp_av(:,:,i)';
    m_av(:,:,i)=mean(month_av_data(:,:,i),2);
    
    max_av=0;
    for j=2:(size(m_av,1)-1)        
        if ((mean(m_av(1:j,1,i))-mean(m_av(j+1:end,1,i)))/mean(m_av(1:j,1,i))> av_per_dif) % at least some per difference
           dif_av=mean(m_av(1:j,1,i))-mean(m_av(j+1:end,1,i));
           if max_av<dif_av
            max_av=dif_av;
           end
        end
    end
    max_av_difference3D(:,1,i)=repmat(max_av, size(average3D,1),1);
    
    % 4th Feature std difference
    temp_std(:,:,i)=reshape(standard_dev3D(1:360,1,i), [30,12]); % Discard 5 last days
    month_std_data(:,:,i)=temp_std(:,:,i)';
    m_std(:,:,i)=mean(month_std_data(:,:,i),2);
    
    max_std=0;
    for j=2:(size(m_std,1)-1)        
        if ((mean(m_std(1:j,1,i))-mean(m_std(j+1:end,1,i)))/mean(m_std(1:j,1,i))> std_per_dif) % at least some per difference
           dif_std=mean(m_std(1:j,1,i))-mean(m_std(j+1:end,1,i));
           if max_std<dif_std
            max_std=dif_std;
           end
        end
    end
    max_std_difference3D(:,1,i)=repmat(max_std, size(standard_dev3D,1),1);
end
X_plot=[average3D standard_dev3D max_av_difference3D max_std_difference3D];