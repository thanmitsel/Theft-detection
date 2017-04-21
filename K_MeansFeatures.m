% pick some z vector 
z=600;
r_cons=randi(size(hh,1),z,1);
somehh=hh(r_cons,:);
someID=ID(r_cons,:);

% Convertions
[h, H]=convertHours3D(somehh);
[d]=convertDays3D(hh);
%% Fraud Initialization
% Create Fraud data
F_data3D=H;
Y2D=zeros(size(H,1),size(H,3));
one_H=zeros(size(H(:,:,1)));

fraud_rate=0.35; % Percentage of consumers who fraud
[normal_idx, fraud_idx] = crossvalind('HoldOut', size(H,3), fraud_rate); % Keep in mind crossval floors the rate
thiefs=find(fraud_idx==1);
for i=1:size(thiefs,1)
    one_H=H(:,:,thiefs(i));
    [f_data, y, F_data,Y] = type3Fraud(one_H);
    F_data3D(:,:,thiefs(i))=F_data;
    Y2D(:,thiefs(i))=Y;
end

% Details 
[kWh_count, time_count, kWh_rate, time_rate] = frauDetails(H, F_data3D);

%% Feature extraction
% Here we use SVM for many consumers

% No Feature extraction for cons x Days Data
% Needs threshold

[days]=convertDays3D(F_data3D); % if more days than the threshold then fraud


% Feature extraction
% 14 Features 
X=zeros(size(H,1),14,size(H,3));
count=0; % counts the errored data
for i=1:size(H,3)
[maxi,maxT,mini,minT,suma,av,stdev...
        , lfactor, mintoav,mintomax,night,skew,kurt,varia, features]=extractFeatures(F_data3D(:,:,i));
    X(:,:,i)=features; % X NEEDS TO BE 3D, so nothing crashes later
    if (sum(find(isnan(X(:,:,i))))~=0)
    fprintf('There is NaN on the %dst client\n',i);
    count=count+1;
    % if find NaN get the values of the previous consumer
    X(:,:,i)=X(:,:,(i-1));
    Y2D(:,i)=Y2D(:,(i-1));
    end
end
fprintf('\nFraud Data and features created.\n');
fprintf('Program paused. Press enter to continue.\n');
pause;
%% Create training and testing set
% Choose from every consumer sample

P=0.3; % Percent of Test
normalization=0;
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
% Carefull the size dependes on multiple initializations
rnd_init=5;
min_cost=zeros(rnd_init,1); 

max_iters = 10;
for i=1:length(clusters)
    K=i;
    initial_centroids = kMeansInitCentroids(X_full, K);
    [cost, centroids, idx] = runkMeans(X_full, initial_centroids, max_iters);
    if i>1 && cost_v(i)<cost_v(i-1) % meybe we got stuck at bad local optima
        for j=1:rnd_init
            initial_centroids = kMeansInitCentroids(X_full, K);
            [cost, centroids, idx] = runkMeans(X_full, initial_centroids, max_iters);
            min_cost(j,1)=cost;
        end
        cost=min(min_cost);
    end
    cost_v(i,1)=cost;
end
prompt='Choose number of K\n';
plotElbow(clusters, cost_v);
x = input(prompt);
K=x;

%  Sample 1000 random indexes (since working with all the data is
%  too expensive. If you have a fast computer, you may increase this.
% sel = floor(rand(1000, 1) * size(X_full, 1)) + 1;

%  Setup Color Palette
palette = hsv(K);
colors = palette(:, :);

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
plotDataPoints(Z(:, :), idx, K);
title('2D plot of features, using PCA for dimensionality reduction');

plotClass(Z(:,:),Y_full(:));
title('Classified examples');
fprintf('Program paused. Press enter to continue.\n');
pause;