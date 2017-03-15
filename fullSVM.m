F_data3D=zeros(size(H));
Y2D=zeros(size(H,1),size(H,3));
one_H=zeros(size(H(:,:,1)));
for i=1:size(H,3)
    one_H=H(:,:,i);
    [f_data, y, F_data,Y] = type3Fraud(one_H);
    F_data3D(:,:,i)=F_data;
    Y2D(:,i)=Y;
end

[maxi,maxT,mini,minT,suma,av,stdev...
, lfactor, mintoav,mintomax,night,skew,kurt,varia, X]=extractFeatures(F_data3D);
X_full= permute(X,[1 3 2]);
X_full= reshape(X_full,[],size(X,2),1);
Y_full=Y2D(:);

N=size(X,1); % No. of observations
P=0.3; % Percent of Test
Xtrain3D=zeros(ceil((1-P)*size(X,1)),size(X,2),size(X,3));
Ytrain2D=zeros(ceil((1-P)*size(X,1)),size(X,3));
Xtest3D=zeros(floor(P*size(X,1)),size(X,2),size(X,3));
Ytest2D=zeros(floor(P*size(X,1)),size(X,3));
for i=1:size(H,3)
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
Y_train=Ytrain2D(:);
X_test= permute(Xtest3D,[1 3 2]);
X_test= reshape(X_test,[],size(Xtest3D,2),1);
Y_test=Ytest2D(:);

%[C_opt, sigma_opt,minobjfn] = tuneSVM(X_full, Y_full,'kernel','rbf','numFolds',5);
%min=100;
%for i=1:10
    %[C_tuned, rbf_sigma,minobjfn] = tuneSVM(X_full, Y_full,'kernel','rbf','numFolds',5);
    %if min > minobjfn
        %min=minobjfn;
        %C_opt=C_tuned;
        %sigma_opt=rbf_sigma;
    %end
%end
%gamma_opt=1/(2*sigma_opt^2);

[C, gamma]=findSVMparams(X_train, Y_train, X_test, Y_test);
% Get best fitted arguments
arguments=['-t ' num2str(2) ' -g ' num2str(gamma) ' -c ' num2str(C)]; 
model=svmtrain(Y_train,X_train,arguments);
prediction= svmpredict(Y_test,X_test,model);

% Create confusion Matrix
[precision, recall, accuracy, F1score] = confusionMatrix (Y_test, prediction);