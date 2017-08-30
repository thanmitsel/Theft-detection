function [F_data3D, Y2D, f_data2D]=simulateType2Fraud(H,h,fraud_rate)  
F_data3D=H;        
f_data2D=h;
Y2D=zeros(size(H,1),size(H,3));
one_H=zeros(size(H(:,:,1)));

[normal_idx, fraud_idx] = crossvalind('HoldOut', size(H,3), fraud_rate); % Keep in mind crossval floors the rate
thiefs=find(fraud_idx==1);
for i=1:size(thiefs,1)
    one_H=H(:,:,thiefs(i));
	[f_data, y, F_data,Y] = type2Fraud(one_H);
	%f_data2D(thiefs(i),:)=f_data';
	f_data2D(thiefs(i),:)=f_data;
    F_data3D(:,:,thiefs(i))=F_data;
	Y2D(:,thiefs(i))=Y;
end