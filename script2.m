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

% Convertions
[h, H]=convertHours3D(hh);
[m, M]=convertMonths3D(hh);
[d]=convertDays3D(hh);

% Apply Fraud values
F_data3D=zeros(size(H));
Y2D=zeros(size(H,1),size(H,3));
one_H=zeros(size(H(:,:,1)));
for i=1:size(H,3)
    one_H=H(:,:,i);
    [f_data, y, F_data,Y] = type3Fraud(one_H);
    F_data3D(:,:,i)=F_data;
    Y2D(:,i)=Y;
end

% Features
[maxi,maxT,mini,minT,suma,av,stdev...
, lfactor, mintoav,mintomax,night,skew,kurt,varia, X]=extractFeatures(H);
