%initialization
clear; close all; clc

% Load Data
% 1st col Meter_ID, 2nd col Day_Time_Code, 3rd col kWh/30min
fprintf('Loading data...\n');
data1 = load('File1.txt');
data2 = load('File2.txt');
data3 = load('File3.txt');
data4 = load('File4.txt');
data5 = load('File5.txt');
data6 = load('File6.txt');
fprintf('Done loading!\n');

% Need to check if IDs have continuous data
% Sort matrix according according to columns
fprintf('Sorting data...\n');
Bdata= [data1; data2; data3; data4; data5; data6];
[Sdata] = sortID(Bdata);
%[A] = sortID(data1);
%[B] = sortID(data2);
%[C] = sortID(data3);
%[D] = sortID(data4);
%[E] = sortID(data5);
%[F] = sortID(data6);
fprintf('Done sorting!\n');

% Count occurances of each ID
IDv=Sdata(:,1);
[occ, ID]=hist(IDv,unique(IDv));
