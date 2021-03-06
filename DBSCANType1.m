% pick some z vector 
z=1000;
r_cons=randi(size(hh,1),z,1);
somehh=hh(r_cons,:);
someID=ID(r_cons,:);

% Convertions
[h, H]=convertHours3D(somehh);

%% Fraud Initialization
% Create Fraud data
F_data3D=H;
Y2D=zeros(size(H,1),size(H,3));
one_H=zeros(size(H(:,:,1)));

% fraud_rate=0.05; % Percentage of consumers who fraud
fraud_rate=0.3; % Percentage of consumers who fraud
[normal_idx, fraud_idx] = crossvalind('HoldOut', size(H,3), fraud_rate); % Keep in mind crossval floors the rate
thiefs=find(fraud_idx==1);
for i=1:size(thiefs,1)    
    intensity=1-betarnd(6,3); % beta distribution
    dstart=floor(normrnd(size(one_H,1)/2,size(one_H,1)/6.5)); % normal distribution
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
% Here we use SVM for many consumers
count=0; % counts the errored data
    
% Feature extraction
ndays=1;
per_dif=0.6;
[X_1]=ConsumerFeatures(F_data3D,per_dif); %includes ONLY 3 features
X_2=mean(X_1,1);
X_2=permute(X_2, [3 2 1]); %
[X_3]=NeighborFeatures(X_2);
Y1D=(sum(Y2D)>ndays)';
X=X_3(:,3:end);
Y=Y1D;

fprintf('\nFraud Data and features created.\n');
%% ===  PCA for Visualization ===
% Use PCA to project this cloud to 2D for visualization
% Subtract the mean to use PCA
prompt=('Apply normalization?\n 0 w/o norm, 1 with norm\n');
x=input(prompt);
if x==0
    X_norm=X;
elseif x==1
    [X_norm, mu, sigma] = normalizeMinus_Plus(X);
end
prompt=('Apply PCA?\n 0 w/o PCA, 1 with PCA\n');
x=input(prompt);
if x==0
    Z=X_norm;
elseif x==1
    % PCA and project the data to 2D
    [U, S] = pca(X_norm);
    Z = projectData(X_norm, U, 2);


    plotClass(Z(:,:),Y(:));
    title('Classified examples');
end
fprintf('Program paused. Press enter to continue.\n');
%% Create training and testing set
% Choose from every consumer sample
% No normarlization needed here

P=0.3; % Percent of Test
normalization=0;
[X_train, Y_train, X_test, Y_test, X_full, Y_full]=pickTrainTest(Z, Y, P, normalization);
Intr=sum(Y_full)/size(Y_full,1);% Probability of Intrusion based on Days

fprintf('\nSegmented Training and Testing.\n');
%% Run DBSCAN Clustering Algorithm

epsilon=0.5;
ep_v=[2^(-10) 2^(-10) 2^(-9) 2^(-8) 2^(-7) 2^(-6) 2^(-5) 2^(-4) 2^(-3) 2^(-2) 2^(-1)];
pts_v=10:50;
max=0;
for j=1:length(pts_v)
    for i=1:length(ep_v)
        [IDX_t, isnoise_t]=DBSCAN(Z, ep_v(i), pts_v(j));
        [F1_temp]=getF1score(isnoise_t,Y_full);
        if F1_temp>max && isnan(F1_temp)==0
            max=F1_temp;
            MinPts=pts_v(j);
            epsilon=ep_v(i);
            IDX=IDX_t;
            isnoise=isnoise_t;
        end
    end
end
PlotClusterinResult(Z, IDX);
prediction=isnoise;

% Create confusion Matrix
% Detection Rate is Recall, False Positive Rate is Inverse recall 
[precision, recall, in_recall, accuracy, F1score] = confusionMatrix (Y_full, prediction);
BDR=Intr*recall/(Intr*recall+(1-Intr)*in_recall) ; % Bayesian Detection Rate for days

rouf_id=find(prediction==1);
roufianos=someID(rouf_id); % Keeps all the ID that contain intrusion

%% Printing Segment
fprintf('kWh Rate %4.2fper | Time Rate %4.2fper |\n',kWh_rate,time_rate);
fprintf('\nClassification for IDs\n');
fprintf('| Precision %4.2f | Recall %4.2f | Accuracy %4.2f | F1score %4.2f |\n',precision,recall,accuracy,F1score);
fprintf('| Actual Fraud %d IDs | Predicted Fraud Right %d IDs | Predicted Fraud Wrong %d IDs |\n',sum(Y_full==1),sum(prediction==1&Y_full==prediction),sum(prediction==1&Y_full~=prediction));
fprintf(' DR  FPR  BDR  Accuracy\n%4.2f %4.2f %4.2f %4.2f \n',recall,in_recall,BDR,accuracy);
