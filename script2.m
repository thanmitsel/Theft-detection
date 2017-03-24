%% Preprocesing, Formating Data for many consumers
% In this script we get consumer data and add fraud values to their data.

% Initialization
clear; close all; clc

% Load Data
 %cd ..\'CER Data'\Data\; %windows
 cd ../CER' Data'/Data/; %Matlab Linux
%cd '../CER Data/Data/'; % Linux
data=load('File1.txt');
%cd '../../code/'; % Linux
cd ../../Thesis/; %Matlab Linux
%cd ..\..\code\; % windows

% Sorting file
[sData] = sortID(data);

% Create matrix of consumersxhh 
% and vector with their IDs
[hh, ID]=pickConsumers(sData);

% pick some z vector 
z=20;
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

% Feature extraction
% 14 Features 
X=zeros(size(H,1),14,size(H,3));
count=0;
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