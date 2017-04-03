%% Preprocesing, Formating Data for many consumers
% In this script we get consumer data and add fraud values to their data.

% pick some z vector 
z=2000;
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
elseif x==2
    Xn=f_data2D;
    ndays=0;
    Yn=(sum(Y2D)>ndays)';
end
    
    
    
fprintf('\nFraud Data and features created.\n');
%% Create training and testing set
% Choose from every consumer sample

P=0.3; % Percent of Test
normalization=1;
[X_train, Y_train, X_test, Y_test, X_full, Y_full]=pickTrainTest(Xn, Yn, P, normalization);
Intr=sum(Y_full)/size(Y_full,1);% Probability of Intrusion based on Days

fprintf('\nSegmented Training and Testing.\n');
%% ===  KMeans & PCA ===
%  One useful application of PCA is to use it to visualize high-dimensional
%  data. In the last K-Means exercise you ran K-Means on 3-dimensional 
%  pixel colors of an image. We first visualize this output in 3D, and then
%  apply PCA to obtain a visualization in 2D.

clusters = 1:20;
cost_v=zeros(length(clusters),1);
min_cost=zeros(10,1); % Carefull the size dependes on multiple initializations

max_iters = 10;
for i=1:length(clusters)
    K=i;
    initial_centroids = kMeansInitCentroids(X_full, K);
    [cost, centroids, idx] = runkMeans(X_full, initial_centroids, max_iters);
    if i>1 && cost_v(i)<cost_v(i-1) % meybe we got stuck at bad local optima
        for j=1:10
            initial_centroids = kMeansInitCentroids(X_full, K);
            [cost, centroids, idx] = runkMeans(X_full, initial_centroids, max_iters);
            min_cost(j,1)=cost;
        end
        cost=min(min_cost);
    end
    cost_v(i,1)=cost;
end
[value, index]=min(cost_v);
K=clusters(index);

%  Sample 1000 random indexes (since working with all the data is
%  too expensive. If you have a fast computer, you may increase this.
% sel = floor(rand(1000, 1) * size(X_full, 1)) + 1;

%  Setup Color Palette
palette = hsv(K);
colors = palette(idx(:), :);

%  Visualize the data and centroid memberships in 3D
%figure;
%scatter3(X(sel, 1), X(sel, 2), X(sel, 3), 10, colors);
%title('Pixel dataset plotted in 3D. Color shows centroid memberships');
%fprintf('Program paused. Press enter to continue.\n');
%pause;

%% ===  PCA for Visualization ===
% Use PCA to project this cloud to 2D for visualization
% Subtract the mean to use PCA
[X_norm, mu, sigma] = normalizeMinus_Plus(X_full);

% PCA and project the data to 2D
[U, S] = pca(X_norm);
Z = projectData(X_norm, U, 2);

% Plot in 2D
figure;
plotDataPoints(Z(:, :), idx(:), K);
title('2D plot of features, using PCA for dimensionality reduction');

plotClass(Z(sel,:),Y(sel));
title('Classified examples');
fprintf('Program paused. Press enter to continue.\n');
pause;