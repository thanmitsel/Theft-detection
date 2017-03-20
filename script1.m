%% Preprocesing, Formating Data for one consumer.
% In this script we get consumer data and add fraud values to their data.

% Initialization
clear; close all; clc

% Load Data
 %cd ..\'CER Data'\Data\; %windows
 cd ../CER' Data'/Data/; %Matlab Linux
%cd '../CER Data/Data/'; % Linux
data=load('File1.txt');
%cd '../../code/'; % Linux
cd ../../code/; %Matlab Linux
%cd ..\..\code\; % windows

% Sorting file
[sData] = sortID(data);

% Create matrix of consumersxhh 
% and vector with their IDs
[hh, ID]=pickConsumers(sData);

% pick one
r_cons=randi(size(hh,1));
onehh=hh(r_cons,:);
oneID=ID(r_cons,:);

% Convertions
[h, H]=convertHours3D(onehh);
[m, M]=convertMonths3D(onehh);
[d]=convertDays3D(onehh);
plotConsumption(d);

 % Test type 1 Fraud
[f_data, y, F_data, Y]=type1Fraud(H);
plotFraudvsNormal(h,f_data);
oneDayFraudvsNormal(H, F_data, Y);

% Test type 2 Fraud
[f_data, y, F_data,Y] = type2Fraud(H);
oneDayFraudvsNormal(H, F_data, Y);

% Test type 3 Fraud
[f_data, y, F_data,Y] = type3Fraud(H);
oneDayFraudvsNormal(H, F_data, Y);

% Details for Fraud
[kWh_count, time_count, kWh_rate, time_rate] = frauDetails(H, F_data);

% Features 
[maxi,maxT,mini,minT,suma,av,stdev...
, lfactor, mintoav,mintomax,night,skew,kurt,varia, X]=extractFeatures(F_data);