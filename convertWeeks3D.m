function [w, W]=convertWeeks3D(hh)
h=zeros(size(hh,1),size(hh,2)/2);
w=zeros(size(hh,1),52);
W=zeros(52,168,size(hh,1));
for i=1:size(h,2)
    h(:,i)=sum(hh(:,(2*i-1):(2*i)),2);
end
for j=1:52
    w(:,j)=sum(hh(:,(336*j-335):(336*j)),2);
end
for k=1:size(hh,1)
    W(:,:,k)=vec2mat(h(k,1:8736),168);
end
end
