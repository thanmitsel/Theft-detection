%% Preprocesing, Formating Data for many consumers
% In this script we get consumer data and add fraud values to their data.

% pick some z vector 
z=200;
r_cons=randi(size(hh,1),z,1);
somehh=hh(r_cons,:);
HH=zeros(365,48,size(somehh,1));
for j=1:size(somehh,1)
    HH(:,:,j)=vec2mat(somehh(j,:),48);
end
someID=ID(r_cons,:);

% Convertions
[h, H]=convertHours3D(somehh);
[d]=convertDays3D(hh);
%% Fraud Initialization
% Create Fraud data
fprintf('Choose classification and press Enter.\n');
prompt='\n0. Days\n1. Hours\n2. Half hours\n';
x = input(prompt);

fraud_rate=0.35; % Percentage of consumers who fraud
if x==0 || x==1 
    F_data3D=H;
    f_data2D=h;
    Y2D=zeros(size(H,1),size(H,3));
    one_H=zeros(size(H(:,:,1)));

    
    [normal_idx, fraud_idx] = crossvalind('HoldOut', size(H,3), fraud_rate); % Keep in mind crossval floors the rate
    thiefs=find(fraud_idx==1);
    for i=1:size(thiefs,1)
        one_H=H(:,:,thiefs(i));
        [f_data, y, F_data,Y] = type3Fraud(one_H);
        f_data2D(thiefs(i),:)=f_data';
        F_data3D(:,:,thiefs(i))=F_data;
        Y2D(:,thiefs(i))=Y;
    end
    % Details 
    [kWh_count, time_count, kWh_rate, time_rate] = frauDetails(H, F_data3D);    
elseif x==2
    F_data3D=HH;
    f_data2D=somehh;
    Y2D=zeros(size(HH,1),size(HH,3));
    one_HH=zeros(size(HH(:,:,1)));
    [normal_idx, fraud_idx] = crossvalind('HoldOut', size(H,3), fraud_rate); % Keep in mind crossval floors the rate
    thiefs=find(fraud_idx==1);
    for i=1:size(thiefs,1)
        one_HH=HH(:,:,thiefs(i));
        [f_data, y, F_data,Y] = type3Fraud(one_HH);
        f_data2D(thiefs(i),:)=f_data';
        F_data3D(:,:,thiefs(i))=F_data;
        Y2D(:,thiefs(i))=Y;
    end
    % Details 
    [kWh_count, time_count, kWh_rate, time_rate] = frauDetails(HH, F_data3D);
end

%% Feature extraction
% Here we use SVM for many consumers

% No Feature extraction for cons x Days Data
% Needs threshold 
if x==0
    [Xn]=convertDays3D(F_data3D); % if more days than the threshold then fraud
    ndays=0;
    Yn=(sum(Y2D)>ndays)';
elseif x==1
    Xn=f_data2D;
    ndays=0;
    Yn=(sum(Y2D)>ndays)';
    % Get rid of a day
    X_temp=Xn(:,1:(8760-24));
    X_week=reshape(X_temp', [24*7,size(X_temp,1)*52])';
    Y_temp=Y2D(1:364,:);
    Y_week=reshape(Y_temp,[7,52*size(Y_temp,2)]);
elseif x==2
    Xn=f_data2D;
    ndays=0;
    Yn=(sum(Y2D)>ndays)';
    % Get rid of a day
    X_temp=f_data2D(:,1:(17520-48));
    % Convert to [cons x week]x[hh of week]
    X_week=reshape(X_temp', [48*7,size(X_temp,1)*52])';
    Y_temp=Y2D(1:364,:);
    Y_week=reshape(Y_temp,[7,52*size(Y_temp,2)]);
end
    Y_vector=(sum(Y_week)>ndays)';
    
    
fprintf('\nFraud Data and features created.\n');
%% Create training and testing set
% Choose from every consumer sample

normalization=0;
P=0.3; % Percent of Test

[X_train, Y_train, X_test, Y_test, X_full, Y_full]=pickTrainTest(X_week, Y_vector, P, normalization);
Intr=sum(Y_full)/size(Y_full,1);% Probability of Intrusion based on Days

fprintf('\nSegmented Training and Testing.\n');
%% ===  PCA for Visualization ===
% Use PCA to project this cloud to 2D for visualization
% Subtract the mean to use PCA
[X_norm, mu, sigma] = normalizeMinus_Plus(X_full);

% PCA and project the data to 2D
[U, S] = pca(X_norm);
Z = projectData(X_norm, U, 2);


plotClass(Z(:,:),Y(:));
title('Classified examples');



%% Run DBSCAN Clustering Algorithm

epsilon=0.5;
ep_v=[2^(-10) 2^(-10) 2^(-9) 2^(-8) 2^(-7) 2^(-6) 2^(-5) 2^(-4) 2^(-3)];
pts_v=10:50;
max=0;
for j=1:length(pts_v)
    for i=1:length(ep_v)
        [IDX_t, isnoise_t]=DBSCAN(Z,ep_v(i),pts_v(j));
        [F1_temp]=getF1score(isnoise,Y_full);
        if F1_temp>max
            max=F1_temp;
            MinPts=pts_v(j);
            epsilon=ep_v(i);
            IDX=IDX_t;
            isnoise=isnoise_t;
        end
    end
end
%% Plot Results

PlotClusterinResult(Z, IDX);
title(['DBSCAN Clustering (\epsilon = ' num2str(epsilon) ', MinPts = ' num2str(MinPts) ')']);
