%initialization
clear; close all; clc

% Load Data
% 1st col Meter_ID, 2nd col Day_Time_Code, 3rd col kWh/30min
fprintf('Loading data...\n');
cd ..\'CER Data'\Data\; % Windows Folders
cd '../CER Data/Data/'; % Linux Dirs
data1 = load('File1.txt');
cd ..\..\code\; % Windows Folders
cd '../../code/; % Linux Dirs
fprintf('Done loading!\n');

fprintf('Sorting data...\n');
[sData] = sortID(data1);
fprintf('Done sorting!\n');

% All convertions in one function
%[hh, h, d, w, m, DxHH, DxH, WxH, MxH]=ConsumptionsConvert(yearCons);

% Index the end of year 2010
[row,col]=find(sData==73048);
% Intialize array and vectors for scaling
[t_h, t_H, t_d, t_w, t_W, t_m, t_M]=initScale;

% Pick an amount of consumers automatically 
% Check if they have data whole year
for i=1:floor(length(row)/10)
    if(sData(row(i)-17519,2)==36601) % enough data for a full year? 
        year=sData((row(i)-17519):row(i),:);
        [h, DxH]=convertHours(year);
        [t_h, t_H]=scale2Consumers(h, DxH, t_h, t_H, i);
    end
end
% if scale is in same loop for diffent array size, it crashes!
% solution might be to create many scale functions
for i=1:floor(length(row)/10)
    if(sData(row(i)-17519,2)==36601) % enough data for a full year? 
        year=sData((row(i)-17519):row(i),:);
        [m, MxH]=convertMonths(year);
        [t_m, t_H]=scale2Consumers(m, MxH, t_m, t_H, i);
    end
end

plotConsumpion(t_m);
barGraph(t_m);
bar3Graph(t_m); % Matlab only




    
