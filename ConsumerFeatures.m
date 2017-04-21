function [X_plot]=ConsumerFeatures(data)
% gets 2 features for consumers
month_data=zeros(12,30,size(data,3));
temp_data=zeros(30,12,size(data,3));
m_av=zeros(12,1,size(data,3));
max_difference3D=zeros(size(data,1),1,size(data,3));

average3D=mean(data,2);
standard_dev3D=std(data,0,2);
for i=1:size(data,3)
    temp_data(:,:,i)=reshape(average3D(1:360,1,i), [30,12]); % Discard 5 last days
    month_data(:,:,i)=temp_data(:,:,i)';
    m_av(:,:,i)=mean(month_data(:,:,i),2);
    
    max=0;
    for j=2:(size(m_av,1)-1)        
        if (mean(0.9*m_av(1:j,1,i))>mean(m_av(j+1:end,1,i))) % atleast 10 per difference
           dif=mean(m_av(1:j,1,i))-mean(m_av(j+1:end,1,i));
           if max<dif
            max=dif;
           end
        end
    end
    max_difference3D(:,1,i)=repmat(max, size(average_vector,1),1);
end
X_plot=[average3D standard_dev3D max_difference3D];