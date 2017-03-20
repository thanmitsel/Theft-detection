%% Create training and testing set
% Choose from every consumer sample

P=0.3; % Percent of Test
[X_train, Y_train, X_test, Y_test, X_full, Y_full]=pickTrainTest(X, Y2D, P);

fprintf('\nSegmented Training and Testing.\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

%% Parameter fitting
% "Optimized" 
%[C_opt, sigma_opt,minobjfn] = tuneSVM(X_full, Y_full,'kernel','rbf','numFolds',5);
min=100;
for i=1:10
    [C_tuned, rbf_sigma,minobjfn] = tuneSVM(X_full, Y_full,'kernel','rbf','numFolds',5);
    if min > minobjfn
        min=minobjfn;
        C=C_tuned;
        sigma_opt=rbf_sigma;
    end
end
gamma=1/(2*sigma_opt^2);
% fast iterative
%[C, gamma]=findSVMparams(X_train, Y_train, X_test, Y_test);

% naive grid search
%[X_full, Y_full]=unrollto2D(X,Y2D);
%[C, gamma]=naiveGridSearch(X_full,Y_full);

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
fprintf(' There are %d consumres with corrupted features!\n',count);