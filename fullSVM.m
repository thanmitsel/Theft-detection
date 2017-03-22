%% Create training and testing set
% Choose from every consumer sample

P=0.3; % Percent of Test
[X_train, Y_train, X_test, Y_test, X_full, Y_full]=pickTrainTest(X, Y2D, P);

fprintf('\nSegmented Training and Testing.\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

%% Parameter fitting
fprintf('Choose method for parameter search and press Enter.\n');
promt='1. Optimized\n2. Fast\n3. Naive Grid Search\n';
x = input(prompt);
fprintf('You pressed: %d', x) ;

if w==1
    % "Optimized" 
    [C_opt, sigma_opt,minobjfn] = tuneSVM(X_full, Y_full,'kernel','rbf','numFolds',5);
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
elseif w==2
    % fast iterative
    [C, gamma]=findSVMparams(X_train, Y_train, X_test, Y_test);
elseif w==3
    % naive grid search
    % no normalization for input
     K=5; % no of folds
     [C, gamma]=naiveGridSearch(X,Y2D,K);
end
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