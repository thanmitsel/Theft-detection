%% Preprocesing, Formating Data for many consumers
% In this script we get consumer data and add fraud values to their data.

% pick some z vector 
z=2000;
r_cons=randi(size(hh,1),z,1);
somehh=hh(r_cons,:);
HH=zeros(365,48,size(somehh,1));
for j=1:size(somehh,1)
    HH(:,:,j)=vec2mat(somehh(j,:),48);
end
someID=ID(r_cons,:);

% Convertions
[h, H]=convertHours3D(somehh);
[d]=convertDays3D(somehh);
%% Fraud Initialization
% Create Fraud data
%fprintf('Choose type of attack.\n');
%prompt='\n1. Type 1\n2. Type 2\n3. Type 3\n4. Mixed\n';
%attack_type= input(prompt);

%fprintf('Choose classification and press Enter.\n');
%prompt='\n0. Days (Fast)\n1. Hours (Default)\n2. Half hours (Slow)\n';
%x = input(prompt);
%if x==2 % else simulation with hours
%    H=HH;
%    h=somehh;
%end

intensity=0:0.001:1;
result_table=zeros(size(intensity,2),6);
for int_idx=1:size(intensity,2)
fraud_rate=0.1; % Percentage of consumers who fraud
F_data3D=H;
f_data2D=h;
Y2D=zeros(size(H,1),size(H,3));
one_H=zeros(size(H(:,:,1)));
    
[normal_idx, fraud_idx] = crossvalind('HoldOut', size(H,3), fraud_rate); % Keep in mind crossval floors the rate
thiefs=find(fraud_idx==1);
for i=1:size(thiefs,1)    
	dstart=floor(normrnd(size(one_H,1)/2,size(one_H,1)/6.5)); % normal distribution
	while dstart<1 || dstart>(size(one_H,1)-1)
        dstart=floor(normrnd(size(one_H,1)/2,size(one_H,1)/6.5)); % normal distribution
    end
    one_H=H(:,:,thiefs(i));
    [f_data, y, F_data,Y] = type1_2Fraud(one_H, intensity(1,int_idx),dstart);
    F_data3D(:,:,thiefs(i))=F_data;
    Y2D(:,thiefs(i))=Y;
    f_data2D(thiefs(i),:)=f_data;
end

fprintf('\nFraud Simulation done.\n');
%% Load time series
% Here we use SVM for many consumers

% No Feature extraction for cons x (time division)
% Needs threshold
ndays=0;
if x==0 % convert for days
	[Xn]=convertDays3D(F_data3D); % if more days than the threshold then fraud
else
	Xn=f_data2D;
end
Yn=(sum(Y2D)>ndays)';
    
    
fprintf('\nLoading of time series done.\n');
%% Create training and testing set
% Choose from every consumer sample
%prompt=('What kind of normalization?\n');
%norm=input(prompt);
norm=1;
P=0.3; % Percent of Test
if norm==0
    normalization=0;
    [X_train, Y_train, X_test, Y_test, X_full, Y_full]=pickTrainTest(Xn, Yn, P, normalization);
elseif norm==1
    normalization=1;
    [X_train, Y_train, X_test, Y_test, X_full, Y_full]=pickTrainTest(Xn, Yn, P, normalization);   
elseif norm==2
    [X_norm, mu, sigma] = normalizeMinus_Plus(Xn);
    [train_idx, test_idx]=crossvalind('HoldOut',z,P);
    X_train=X_norm(train_idx,:);
    Y_train=Yn(train_idx);
    X_test=X_norm(test_idx,:);
    Y_test=Yn(test_idx);
    X_full=X_norm;
    Y_full=Yn;
end
Intr=sum(Y_full)/size(Y_full,1);% Probability of Intrusion based on Days
fprintf('\nSegmented Training and Testing.\n');
%% Test linear kernel
arguments=['-t ' num2str(0)]; % use linear kernel 
model=svmtrain(double(Y_train),double(X_train),arguments);
prediction= svmpredict(double(Y_test),double(X_test),model);
[precision, recall, in_recall, accuracy, F1score] = confusionMatrix (Y_test, prediction);
BDR=Intr*recall/(Intr*recall+(1-Intr)*in_recall) ; % Bayesian Detection Rate for days
%fprintf('Option & DR  & FPR & Accuracy & F1 score & BDR \n');
fprintf('%d & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \n',intensity(1,int_idx),recall,in_recall,accuracy,F1score,BDR);
result_table(int_idx, :)=[intensity(1,int_idx) recall in_recall accuracy F1score BDR];
end