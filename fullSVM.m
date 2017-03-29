%% Create training and testing set
% Choose from every consumer sample
ndays=10; % if more days than the threshold then fraud
P=0.3; % Percent of Test
normalization=1;
[X_train, Y_train, X_test, Y_test, X_full, Y_full]=pickTrainTest(X, Y2D, P, normalization);
Intr=sum(Y_full)/size(Y_full,1);% Probability of Intrusion based on Days
Y_table=vec2mat(Y_test, floor(P*size(X,1)))';
class_ID=(sum(Y_table==1)>ndays)'; % fraud if more than 20 days

fprintf('\nSegmented Training and Testing.\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

%% Parameter fitting
fprintf('Choose method for parameter search and press Enter.\n');
prompt='\n0. Optimized\n1. Fully Optimized\n2. Fast\n3. Naive Grid Search\n4. No search\n';
x = input(prompt);
fprintf('You pressed: %d\n', x) ;

if x==0
    % "Optimized" 
    [C, sigma_opt,minobjfn] = tuneSVM(X_full, Y_full,'kernel','rbf','numFolds',5);
    gamma=1/(2*sigma_opt^2);
elseif x==1
    % "Optimized full search" 
    nloop=10; % no of repeating the process of parameter search
    min=100;
    for i=1:nloop
        [C_tuned, rbf_sigma,minobjfn] = tuneSVM(X_full, Y_full,'kernel','rbf','numFolds',5);
        if min > minobjfn
            min=minobjfn;
            C=C_tuned;
            sigma_opt=rbf_sigma;
        end
    end
    gamma=1/(2*sigma_opt^2);
elseif x==2
    % fast iterative
    [C, gamma]=findSVMparams(X_train, Y_train, X_test, Y_test);
elseif x==3
    % naive grid search
    % no normalization for input
     K=5; % no of folds
     [C, gamma]=naiveGridSearch(X,Y2D,K);
elseif x==4
    gamma=0.01;
    C=10;
end

% Get best fitted arguments
arguments=['-t ' num2str(2) ' -h ' num2str(0) ' -g ' num2str(gamma) ' -c ' num2str(C)];
%% SVM test and confusion matrices
% Test SVM
model=svmtrain(Y_train,X_train,arguments);
prediction= svmpredict(Y_test,X_test,model);

pred_table=vec2mat(prediction, floor(P*size(X,1)))';
pred_ID=(sum(pred_table==1)>ndays)'; % fraud if more than 20 days

% Create confusion Matrix
% Detection Rate is Recall, False Positive Rate is Inverse recall 
[precision, recall, in_recall, accuracy, F1score] = confusionMatrix (Y_test, prediction);
BDR=Intr*recall/(Intr*recall+(1-Intr)*in_recall) ; % Bayesian Detection Rate for days
[precision_t, recall_t, in_recall_t, accuracy_t, F1score_t] = confusionMatrix (class_ID, pred_ID);

BDR_t=fraud_rate*recall_t/(fraud_rate*recall_t+ (1-fraud_rate)*in_recall_t); % Bayesian Detection Rate for Consumers

rouf_id=find(pred_ID==1);
roufianos=someID(rouf_id); % Keeps all the ID that contain intrusion

%% Printing Segment
fprintf('\nThere are %d consumers with corrupted features!\n',count);
fprintf('kWh Rate %4.2fper | Time Rate %4.2fper |\n',kWh_rate,time_rate);

fprintf('\nClassification for IDs\n');
fprintf('| Precision %4.2f | Recall %4.2f | Accuracy %4.2f | F1score %4.2f |\n',precision_t,recall_t,accuracy_t,F1score_t);
fprintf('| Actual Fraud %d IDs | Predicted Fraud Right %d IDs | Predicted Fraud Wrong %d IDs |\n',sum(class_ID==1),sum(pred_ID==1&pred_ID==class_ID),sum(pred_ID==1&class_ID~=pred_ID));
fprintf(' DR  FPR BDR Accuracy\n%4.2f %4.2f % 4.2f %4.2f \n',recall_t,in_recall_t,BDR_t, accuracy_t);
% fprintf('Black List\n')
% disp(roufianos);

fprintf('\nClassification for Days\n');
fprintf('| Precision %4.2f | Recall %4.2f | Accuracy %4.2f | F1score %4.2f |\n',precision,recall,accuracy,F1score);
fprintf('| Actual Fraud %d Days | Predicted Fraud Right %d Days | Predicted Fraud Wrong %d Days |\n',sum(Y_test==1),sum(prediction==1&Y_test==prediction),sum(prediction==1&Y_test~=prediction));
fprintf(' DR  FPR  BDR  Accuracy\n%4.2f %4.2f %4.2f %4.2f \n',recall,in_recall,BDR,accuracy);