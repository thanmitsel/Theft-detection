function [F_data3D, Y2D, f_data2D]=simulateMixedFraud(H,h, fraud_rate)  
F_data3D=H;
f_data2D=h;
Y2D=zeros(size(H,1),size(H,3));
one_H=zeros(size(H(:,:,1)));
    
[normal_idx, fraud_idx] = crossvalind('HoldOut', size(H,3), fraud_rate); % Keep in mind crossval floors the rate
thiefs=find(fraud_idx==1);
for i=1:size(thiefs,1)
    if size(thiefs)/i<= 0.7
        intensity=1-betarnd(6,3); % beta distribution
        dstart=floor(normrnd(size(one_H,1)/2,size(one_H,1)/6.5)); % normal distribution
        while dstart<1 || dstart>(size(one_H,1)-1)
            dstart=floor(normrnd(size(one_H,1)/2,size(one_H,1)/6.5)); % normal distribution
        end
        one_H=H(:,:,thiefs(i));
        [f_data, y, F_data,Y] = type1_2Fraud(one_H, intensity,dstart);
        f_data2D(thiefs(i),:)=f_data;
    elseif size(thiefs)/i> 0.7 & size(thiefs)/i <=0.9
        one_H=H(:,:,thiefs(i));
        [f_data, y, F_data,Y] = type2Fraud(one_H);
        %f_data2D(thiefs(i),:)=f_data';
        f_data2D(thiefs(i),:)=f_data;
    else
        one_H=H(:,:,thiefs(i));
        [f_data, y, F_data,Y] = type3Fraud(one_H);
        %f_data2D(thiefs(i),:)=f_data';
        f_data2D(thiefs(i),:)=f_data;
    end    
    F_data3D(:,:,thiefs(i))=F_data;
	Y2D(:,thiefs(i))=Y;
end