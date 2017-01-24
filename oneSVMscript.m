% This a script for an SVM for one consumer
% script1 should give all the input arguments
% for these functions


[f_data,y, F_data,Y] = type3Fraud(H);
[maxi,maxT,mini,minT,suma,av,stdev...
, lfactor, mintoav,mintomax,night,skew,kurt,varia, X]=extractFeatures(F_data);
P=size(X,1);
% Details for Fraud
[kWh_count, time_count, kWh_rate, time_rate] = frauDetails(H, F_data);

% Cross-Validation
[trainidx, testidx]=crossValind(P,0.3);
% Train set
Xtrain= X(trainidx,:);
Ytrain= Y(trainidx);
% Test set 
Xval=X(testidx,:);
Yval=Y(testidx);

% Iteratively seek parameters
[C, gamma]=findSVMparams(Xtrain, Ytrain, Xval, Yval);

% Get best fitted arguments
arguments=['-t ' num2str(2) ' -g ' num2str(gamma) ' -c ' num2str(C)]; 
model=svmtrain(Ytrain,Xtrain,arguments);
prediction= svmpredict(Yval,Xval,model);

% Create confusion Matrix
[precision, recall, accuracy, F1score] = confusionMatrix (Yval, prediction);

