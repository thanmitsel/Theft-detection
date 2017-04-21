function [X_train, Y_train, X_test, Y_test, X_full, Y_full]=pickTrainTest(X, Y2D, P, normalization)
% It separates data to 2 matrices Train and Test
% Normalizes with the same factor 
% Input 3D arrays, output 2D array

N=size(X,1); % No. of observations
Xtrain3D=zeros(ceil((1-P)*size(X,1)),size(X,2),size(X,3));
Ytrain2D=zeros(ceil((1-P)*size(X,1)),size(X,3));
Xtest3D=zeros(floor(P*size(X,1)),size(X,2),size(X,3));
Ytest2D=zeros(floor(P*size(X,1)),size(X,3));
for i=1:size(X,3)
    [Train, Test] = crossvalind('HoldOut', N, P); %Train is ceiled, Test is floored
    % Train set
    Xtrain3D(:,:,i)= X(Train,:,i);
    Ytrain2D(:,i)= Y2D(Train,i);
    % Test set 
    Xtest3D(:,:,i)=X(Test,:,i);
    Ytest2D(:,i)=Y2D(Test,i);
end
X_train= permute(Xtrain3D,[1 3 2]);
X_train= reshape(X_train,[],size(Xtrain3D,2),1);
if normalization==1
    [X_train, minval, maxval]=normalizeFeatures(X_train); % Normalize Training Set
end
Y_train=Ytrain2D(:);
X_test= permute(Xtest3D,[1 3 2]);
X_test= reshape(X_test,[],size(Xtest3D,2),1);
if normalization==1
    [X_test]=normalizeTest(X_test, minval, maxval); % Normalize Test set based to these values
end
Y_test=Ytest2D(:);
X_full=[X_train;X_test];
Y_full=[Y_train;Y_test];