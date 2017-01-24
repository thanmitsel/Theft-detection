function [trainidx, testidx] = crossValind (initial_size, test_rate)
% Outputs logical vectors with 1 when used in training or testing 
trainidx=zeros(initial_size,1);
testidx=zeros(initial_size,1);

test_size=round(test_rate*initial_size);
test=randperm(initial_size,test_size);
testidx(test)=1;

testidx=logical(testidx);
trainidx=~testidx;
end
