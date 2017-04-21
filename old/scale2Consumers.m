function [temph, tempH]=scale2Consumers(h, H, temph, tempH, indx)
if(indx==1)
    temph=h;
    tempH=H;
else
    temph=[temph h];
    tempH=[tempH H];
end     
end