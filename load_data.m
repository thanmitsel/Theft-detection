% Initialization
clear; close all; clc

% Load Data
cd ../CER' Data'/Data/; %Matlab Linux
data1=load('File1.txt');
data2=load('File2.txt');
data3=load('File3.txt');
data=[data1;data2;data3];
cd ../../Thesis/; %Matlab Linux
