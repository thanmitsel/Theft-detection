function [yearData, yearID]=pickConsumers(sData)
% Creates one matrix (consumers x HalfHours)
% one vector with the corresponding IDs

[row,col]=find(sData==73048);
indx=1;
people=floor(length(row));
for i=1:people
    if (sData(row(i)-17519,2)==36601)
        yearData(indx,:)=sData(row(i)-17519:row(i),3)';
        yearID(indx,1)=sData(row(i),1);
        indx=indx+1;
     end
end

