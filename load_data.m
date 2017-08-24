% Initialization
clear; close all; clc

% Load Data
cd ../CER' Data'/Data/; %Matlab Linux
data1=load('File1.txt');
data2=load('File2.txt');
data3=load('File3.txt');
data4=load('File4.txt');
data5=load('File5.txt');
data6=load('File6.txt');
data=[data1;data2;data3;data4;data5;data6];
cd ../../Thesis/; %Matlab Linux


% Sorting file
[sData] = sortID(data);

% Create matrix of consumersxhh 
% and vector with their IDs
[hh, ID]=pickConsumers(sData);
