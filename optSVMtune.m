% Optimized for parameter search
% Needs X, Y from the script oneSVMscript

min=100;
for i=1:10
    [C_tuned, rbf_sigma,minobjfn] = tuneSVM(X, Y,'kernel','rbf','numFolds',5);
    if min > minobjfn
        min=minobjfn;
        C_opt=C_tuned;
        sigma_opt=rbf_sigma;
    end
end
gamma_opt=1/(2*sigma_opt^2);
% Get best fitted arguments
arguments=['-t ' num2str(2) ' -g ' num2str(gamma_opt) ' -c ' num2str(C_opt)]; 
model=svmtrain(Ytrain,Xtrain,arguments);
prediction= svmpredict(Yval,Xval,model);

% Create confusion Matrix
[precision, recall, accuracy, F1score] = confusionMatrix (Yval, prediction);