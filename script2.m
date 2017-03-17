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