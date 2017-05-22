%% This script is supposed to test the accuracy of every feature
% most parameters that follow a distribution are now fixed
% pick some z vector 
    z=2000;
    r_cons=randi(size(hh,1),z,1);
    somehh=hh(r_cons,:);
    someID=ID(r_cons,:);

    % Convertions
    [h, H]=convertHours3D(somehh);

    %% Fraud Initialization
    % Create Fraud data
    prompt='Choose fixed fraud values (0) or random (1)\n';
    random_values=input(prompt);
    F_data3D=H;
    Y2D=zeros(size(H,1),size(H,3));
    one_H=zeros(size(H(:,:,1)));

fraud_rate=0.5; % Percentage of consumers who fraud
[normal_idx, fraud_idx] = crossvalind('HoldOut', size(H,3), fraud_rate); % Keep in mind crossval floors the rate
thiefs=find(fraud_idx==1);
for i=1:size(thiefs,1)    
    if random_values==0
        intensity=0.2; % fixed intensity
        dstart=182; % fixed day of staring fraud
    else
        intensity=1-betarnd(6,3); % beta distribution
        dstart=floor(normrnd(size(one_H,1)/2,size(one_H,1)/6.5)); % normal distribution
    end
    while dstart<1 || dstart>(size(one_H,1)-1)
        dstart=floor(normrnd(size(one_H,1)/2,size(one_H,1)/6.5)); % normal distribution
    end
    one_H=H(:,:,thiefs(i));
    [f_data, y, F_data,Y] = type1_2Fraud(one_H, intensity,dstart);
    F_data3D(:,:,thiefs(i))=F_data;
    Y2D(:,thiefs(i))=Y;
end

% Details 
[kWh_count, time_count, kWh_rate, time_rate] = frauDetails(H, F_data3D);
%% Feature extraction
prompt=('Choose fast or sophisticated features\n0. fast 1. sofisticated (KMEANS)\n');
sophisticated=input(prompt);

ndays=1;
Y1D=(sum(Y2D)>ndays)';
Y=Y1D;


% Feature extraction

if sophisticated==0
    av_per_dif=0.7;
    std_per_dif=0.7;
    symmetric_av=0.6;
    symmetric_std=0.6;
    av_threshold=0.6;
    std_threshold=0.7;
    [X_1]=ConsumerFeatures(F_data3D, av_per_dif, std_per_dif, symmetric_av, symmetric_std); %includes ONLY 3 features
    X_2=mean(X_1,1);
    X_2=permute(X_2, [3 2 1]);

    % Obtain features related to neighbors
    K=3;
    [X_3]=NeighborFeatures(X_1, X_2, K, av_threshold, std_threshold);
    %X=X_3(:,1:(end));
    [trend_coef]=get_Trend(X_1);
    X=[X_2 X_3(:,9:end) trend_coef];
elseif sophisticated==1
    av_per_dif=0.7;
    std_per_dif=0.7;
    av_cut_per=0.1; % 0.8
    std_cut_per=0.1;% 0.6
    neigh_av_cut_per=0.3; % 0.6
    neigh_std_cut_per=0.4;
   [X]=sophisticatedFeatures(F_data3D, av_per_dif, std_per_dif, ...
       av_cut_per, std_cut_per, neigh_av_cut_per, neigh_std_cut_per);
end
%% 
prompt=('Choose which feature u wanna test 3-8.\n');
nofeature=input(prompt); % number of feature

feature_columns=1:size(X,2);
binary_choice=feature_columns==nofeature;

X_reduced=X(:,binary_choice);
prediction=X_reduced~=0;

[precision, recall, in_recall, accuracy, F1score]=confusionMatrix(Y, prediction);
Intr=sum(Y)/size(Y,1);
BDR=Intr*recall/(Intr*recall+(1-Intr)*in_recall) ; % Bayesian Detection Rate for days

%% Printing Segment
fprintf('kWh Rate %4.2fper | Time Rate %4.2fper |\n',kWh_rate,time_rate);
fprintf('\nClassification for IDs\n');
fprintf('| Precision %4.2f | Recall %4.2f | Accuracy %4.2f | F1score %4.2f |\n',precision,recall,accuracy,F1score);
fprintf('| Actual Fraud %d IDs | Predicted Fraud Right %d IDs | Predicted Fraud Wrong %d IDs |\n',sum(Y==1),sum(prediction==1&Y==prediction),sum(prediction==1&Y~=prediction));
fprintf(' DR  FPR  BDR  Accuracy F1\n%4.2f %4.2f %4.2f %4.2f %4.2f\n',recall,in_recall,BDR,accuracy,F1score);
fprintf('\nProgramm Paused, if u need FPR analysis press any key\nelse press Ctrl+C\n');
