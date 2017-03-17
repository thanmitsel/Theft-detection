%% Creating Fraud Data and extraction Features
% This script needs script2 before so it has H
F_data3D=zeros(size(H));
Y2D=zeros(size(H,1),size(H,3));
one_H=zeros(size(H(:,:,1)));
for i=1:size(H,3)
    one_H=H(:,:,i);
    [f_data, y, F_data,Y] = type3Fraud(one_H);
    F_data3D(:,:,i)=F_data;
    Y2D(:,i)=Y;
end

% Feature extraction
% 14 Features
X=zeros(size(H,1),14,size(H,3));
count=0;
for i=1:size(H,3)
[maxi,maxT,mini,minT,suma,av,stdev...
        , lfactor, mintoav,mintomax,night,skew,kurt,varia, features]=extractFeatures(F_data3D(:,:,i));
    X(:,:,i)=features; % X NEEDS TO BE 3D, so nothing crashes later
    if (sum(find(isnan(X(:,:,i))))~=0)
    fprintf('There is NaN on the %dst client\n',i);
    count=count+1;
    % if find NaN get the values of the previous consumer
    X(:,:,i)=X(:,:,(i-1));
    Y2D(:,i)=Y2D(:,(i-1));
    end
end
fprintf('Fraud Data and features created.\n');
fprintf('Program paused. Press enter to continue.\n');
pause;
%% Normalize
% Unroll 3D and 2D
X_full= permute(X,[1 3 2]);
X_full= reshape(X_full,[],size(X,2),1);
Y_full=Y2D(:);

% Normalize
[Xn_full]=normalizeFeatures(X_full);

% Reshape to 3D
[row,col] = size(Xn_full);
nlay  = size(H,3);
Xn   = permute(reshape(Xn_full',[col,row/nlay,nlay]),[2,1,3]);

fprintf('Data Normalized.\n');
fprintf('Program paused. Press enter to continue.\n');
pause;
%% Create training and testing set
% Choose from every consumer sample
N=size(X,1); % No. of observations
P=0.3; % Percent of Test
Xtrain3D=zeros(ceil((1-P)*size(X,1)),size(X,2),size(X,3));
Ytrain2D=zeros(ceil((1-P)*size(X,1)),size(X,3));
Xtest3D=zeros(floor(P*size(X,1)),size(X,2),size(X,3));
Ytest2D=zeros(floor(P*size(X,1)),size(X,3));
for i=1:size(H,3)
    [Train, Test] = crossvalind('HoldOut', N, P); %Train is ceiled, Test is floored
    % Train set
    Xtrain3D(:,:,i)= Xn(Train,:,i);
    Ytrain2D(:,i)= Y2D(Train,i);
    % Test set 
    Xtest3D(:,:,i)=Xn(Test,:,i);
    Ytest2D(:,i)=Y2D(Test,i);
end
X_train= permute(Xtrain3D,[1 3 2]);
X_train= reshape(X_train,[],size(Xtrain3D,2),1);
Y_train=Ytrain2D(:);
X_test= permute(Xtest3D,[1 3 2]);
X_test= reshape(X_test,[],size(Xtest3D,2),1);
Y_test=Ytest2D(:);

fprintf('Segmented Training and Testing.\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

%% Parameter fitting
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

% Test SVM
model=svmtrain(Y_train,X_train,arguments);
prediction= svmpredict(Y_test,X_test,model);

% Create confusion Matrix
[precision, recall, accuracy, F1score] = confusionMatrix (Y_test, prediction);
fprintf('\n| Precision %4.2f | Recall %4.2f | Accuracy %4.2f | F1score %4.2f |\n',precision,recall,accuracy,F1score);
%fprintf('| Actual %d Days | Predicted Right %d Days | kWh Rate %4.2fper | Time Rate %4.2fper |\n',sum(Y_test==1),sum(prediction==1&Y_test==prediction),kWh_rate,time_rate);
fprintf('| Actual %d Days | Predicted Right %d Days |\n',sum(Y_test==1),sum(prediction==1&Y_test==prediction));
fprintf(' There are %d client with corrupted values!\n',count);