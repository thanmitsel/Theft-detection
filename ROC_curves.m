% Initialization
clear; close all; clc

% 3 curves, high medium and low intensity
% 10 points on every curve, threshold

intensity=[20 50 80];
%thresh=[10 20 50 70 100 120 150 170 180 200];
thresh=[10 20 30 40 50 60 70 80 90 100]; % ndays
fraud_rate=0.9; % Percentage of consumers who fraud

DR_days=zeros(length(thresh), length(intensity));
FPR_days=zeros(length(thresh), length(intensity));
DR_IDs=zeros(length(thresh), length(intensity));
FPR_IDs=zeros(length(thresh), length(intensity));

%% Preprocesing, Formating Data for many consumers
% In this script we get consumer data and add fraud values to their data.


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
z=50; % Needs to be 300>
r_cons=randi(size(hh,1),z,1);
somehh=hh(r_cons,:);
someID=ID(r_cons,:);

% Convertions
[h, H]=convertHours3D(somehh);

for id_i=1:length(intensity)
    for id_th=1:length(thresh)
            %% Fraud Initialization
            % Create Fraud data
            F_data3D=H;
            Y2D=zeros(size(H,1),size(H,3));
            one_H=zeros(size(H(:,:,1)));
            
            [normal_idx, fraud_idx] = crossvalind('HoldOut', size(H,3), fraud_rate); % Keep in mind crossval floors the rate
            thiefs=find(fraud_idx==1);
            for i=1:size(thiefs,1)
                one_H=H(:,:,thiefs(i));
                [f_data, y, F_data,Y] = type3_1Fraud(one_H, intensity(id_i));
                F_data3D(:,:,thiefs(i))=F_data;
                Y2D(:,thiefs(i))=Y;
            end
            %% Feature extraction
            % Here we use SVM for many consumers

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
            %% Create training and testing set
            % Choose from every consumer sample
            
            P=0.3; % Percent of Test
            normalization=1;
            [X_train, Y_train, X_test, Y_test, X_full, Y_full]=pickTrainTest(X, Y2D, P, normalization);
            Intr=sum(Y_full)/size(Y_full,1);% Probability of Intrusion based on Days
            Y_table=vec2mat(Y_test, floor(P*size(X,1)))';
            class_ID=(sum(Y_table==1)>thresh(id_th))'; % fraud if more than threshold fraud

            fprintf('\nSegmented Training and Testing.\n');
            % Get best fitted arguments
            arguments=['-t ' num2str(2) ' -g ' num2str(0.1) ' -c ' num2str(10)]; 
            %% SVM test and confusion matrices
            % Test SVM
            model=svmtrain(Y_train,X_train,arguments);
            prediction= svmpredict(Y_test,X_test,model);
            pred_table=vec2mat(prediction, floor(P*size(X,1)))';
            pred_ID=(sum(pred_table==1)>thresh(id_th))'; % fraud if more than threshold

            % Create confusion Matrix
            % Detection Rate is Recall, False Positive Rate is Inverse recall 
            [precision, recall, in_recall, accuracy, F1score] = confusionMatrix (Y_test, prediction);
            
            [precision_t, recall_t, in_recall_t, accuracy_t, F1score_t] = confusionMatrix (class_ID, pred_ID);
            DR_IDs(id_th, id_i)=recall_t;
            FPR_IDs(id_th,id_i)=in_recall_t;
            
            rouf_id=find(pred_ID==1);
            roufianos=someID(rouf_id); % Keeps all the ID that contain intrusion
    end
end

plotCurves(FPR_IDs,DR_IDs);
