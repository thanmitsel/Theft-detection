% This a script for an SVM for one consumer
% script1 should give all the input arguments
% for these functions

Intr=sum(Y)/size(Y,1);% Probability of Intrusion based on Days

% Cross-Validation
P=size(X,1);
[trainidx, testidx]=crossValind(P,0.3);

% Train set
Xtrain= X(trainidx,:);
Ytrain= Y(trainidx);
% Normalize Training Set 
[Xtn, minval, maxval]=normalizeFeatures(Xtrain);

% Test set 
Xval=X(testidx,:);
Yval=Y(testidx);
% Normalize Test set based to these values
[Xvn]=normalizeTest(Xval, minval, maxval);


% Iteratively seek parameters
[C, gamma]=findSVMparams(Xtn, Ytrain, Xvn, Yval);
%[C, gamma]=naiveGridSearch(X,Y);

% Get best fitted arguments
%arguments=['-t ' num2str(2) ' -g ' num2str(gamma) ' -c ' num2str(C)]; 
arguments=['-t ' num2str(2) ' -h ' num2str(0) ' -g ' num2str(0.1) ' -c ' num2str(10)]; 
model=svmtrain(Ytrain,Xtn,arguments);
prediction= svmpredict(Yval,Xvn,model);

% Create confusion Matrix
[precision, recall, in_recall, accuracy, F1score] = confusionMatrix (Yval, prediction);
BDR=Intr*recall/(Intr*recall+(1-Intr)*in_recall) ; % Bayesian Detection Rate for days
fprintf('| kWh Rate %4.2fper | Time Rate %4.2fper |\n', kWh_rate, time_rate);

fprintf('\nClassification for Days\n');
fprintf('| Precision %4.2f | Recall %4.2f | Accuracy %4.2f | F1score %4.2f |\n',precision,recall,accuracy,F1score);
fprintf('| Actual Fraud %d Days | Predicted Fraud Right %d Days | Predicted Fraud Wrong %d Days |\n',sum(Yval==1),sum(prediction==1&Yval==prediction),sum(prediction==1&Yval~=prediction));
fprintf(' DR  FPR  BDR  Accuracy\n%4.2f %4.2f %4.2f %4.2f \n',recall,in_recall,BDR,accuracy);