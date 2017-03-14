% Optimized for parameter search
% Needs X, Y from the script oneSVMscript

[C_tuned, rbf_sigma,minobjfn] = tuneSVM(X, Y);
rbf_gamma=1/(2*rbf_sigma^2);

% Get best fitted arguments
arguments=['-t ' num2str(2) ' -g ' num2str(rbf_gamma) ' -c ' num2str(C_tuned)]; 
model=svmtrain(Ytrain,Xtrain,arguments);
prediction= svmpredict(Yval,Xval,model);

% Create confusion Matrix
[precision, recall, accuracy, F1score] = confusionMatrix (Yval, prediction);